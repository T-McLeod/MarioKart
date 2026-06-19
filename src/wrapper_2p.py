"""Two-player Super Mario Kart environment for Phase 2.

`TwoPlayerMarioEnv` wraps a `stable_retro.make(..., players=2)` env and presents
it as a single multi-agent Gymnasium env:

  * observation: the native split-screen frame is cropped (top half = Player 1,
    bottom half = Player 2), each half greyscaled + resized to 84x84 and stacked
    over `num_stack` frames -> Box(2, num_stack, 84, 84), float32 in [0, 1].
  * action: MultiDiscrete([N, N]) -- one discrete index per player, de-multiplexed
    through `action_map` (12-button arrays) and concatenated into the 24-bit
    MultiBinary action the players=2 core expects.
  * frame skip: each `step` advances the core `skip` frames (action repeated).
  * reward: always 0.0 here -- per-player reward shaping is owned by the Python
    reward wrappers (see RelativeProgressReward etc.). Raw per-player state is
    exposed under info["p0"] / info["p1"] for those wrappers to consume.

The greyscale/resize mirrors `MarioResize` in wrapper.py so both halves match the
single-player observation pipeline.
"""
import collections

import cv2
import numpy as np
import gymnasium as gym

from .agents.ppo_nature.agent import DISCOVERY_ACTIONS
from .wrapper import get_checkpoint


def _gray84(rgb_half):
    """RGB (H, W, 3) -> uint8 (84, 84) greyscale, matching MarioResize."""
    gray = cv2.cvtColor(rgb_half, cv2.COLOR_RGB2GRAY)
    return cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)


def split_player_info(info):
    """Split a flat retro info dict into per-player (p0, p1) sub-dicts.

    Keys mirror those the single-player helpers expect (``current_checkpoint``,
    ``lapsize``, ``lap``) so get_checkpoint() works on each sub-dict directly.
    """
    lapsize = info.get("lapsize", 0)
    p0 = {
        "current_checkpoint": info.get("current_checkpoint", 0),
        "lapsize": lapsize,
        "lap": info.get("lap", 128),
        "speed": info.get("kart1_speed", 0),
        "rank": info.get("rank", 0),
        "surface": info.get("surface", 0),
        "direction": info.get("kart1_direction", 0),
        "X": info.get("kart1_X", 0),
        "Y": info.get("kart1_Y", 0),
    }
    p1 = {
        "current_checkpoint": info.get("kart2_checkpoint", 0),
        "lapsize": lapsize,
        "lap": info.get("kart2_lap", 128),
        "speed": info.get("kart2_speed", 0),
        "rank": info.get("kart2_rank", 0),
        "surface": info.get("kart2_surface", 0),
        "direction": info.get("kart2_direction", 0),
        "X": info.get("kart2_X", 0),
        "Y": info.get("kart2_Y", 0),
    }
    return p0, p1


class TwoPlayerMarioEnv(gym.Wrapper):
    def __init__(self, env, action_map=None, skip=4, num_stack=4):
        super().__init__(env)

        if action_map is None:
            # Default to the agent's discrete action table.
            action_map = DISCOVERY_ACTIONS
        self.action_map = [np.asarray(a, dtype=np.int8) for a in action_map]
        self.num_actions = len(self.action_map)
        self._buttons_per_player = self.action_map[0].shape[0]

        self.skip = skip
        self.num_stack = num_stack
        self._frames = [
            collections.deque(maxlen=num_stack),
            collections.deque(maxlen=num_stack),
        ]

        self.action_space = gym.spaces.MultiDiscrete([self.num_actions, self.num_actions])
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(2, num_stack, 84, 84), dtype=np.float32
        )

    # -- observation helpers -------------------------------------------------
    def _split(self, frame):
        """Return (p0_gray84, p1_gray84) from a full split-screen RGB frame."""
        half = frame.shape[0] // 2
        top = frame[:half]
        bottom = frame[half:2 * half]
        return _gray84(top), _gray84(bottom)

    def _stacked_obs(self):
        o0 = np.asarray(self._frames[0], dtype=np.float32) / 255.0  # (num_stack,84,84)
        o1 = np.asarray(self._frames[1], dtype=np.float32) / 255.0
        return np.stack([o0, o1], axis=0)                          # (2,num_stack,84,84)

    # -- per-player info -----------------------------------------------------
    def _player_info(self, info):
        return split_player_info(info)

    # -- gym API -------------------------------------------------------------
    def reset(self, **kwargs):
        frame, info = self.env.reset(**kwargs)
        g0, g1 = self._split(frame)
        for _ in range(self.num_stack):
            self._frames[0].append(g0)
            self._frames[1].append(g1)
        info["p0"], info["p1"] = self._player_info(info)
        return self._stacked_obs(), info

    def step(self, action):
        a0, a1 = int(action[0]), int(action[1])
        joint = np.concatenate([self.action_map[a0], self.action_map[a1]])

        terminated = truncated = False
        info = {}
        frame = None
        for _ in range(self.skip):
            frame, _, terminated, truncated, info = self.env.step(joint)
            if terminated or truncated:
                break

        g0, g1 = self._split(frame)
        self._frames[0].append(g0)
        self._frames[1].append(g1)

        info["p0"], info["p1"] = self._player_info(info)
        # Reward is owned by the Python reward wrappers; emit 0.0 here.
        return self._stacked_obs(), 0.0, terminated, truncated, info


# ===========================================================================
# Per-player reward wrappers
#
# These wrap TwoPlayerMarioEnv and maintain a length-2 reward vector under
# info["rewards"] (index 0 = Player 1 / learner, index 1 = Player 2). The
# scalar reward each wrapper returns is the sum of that vector (for monitoring /
# gym-vector aggregation); the training loop reads the per-player vector.
# ===========================================================================

_PLAYER_KEYS = ("p0", "p1")


def _player_rewards(info):
    """Get (or lazily init) the length-2 per-player reward vector in info."""
    rewards = info.get("rewards")
    if rewards is None:
        rewards = np.zeros(2, dtype=np.float32)
        info["rewards"] = rewards
    return rewards


def _progress_delta(pinfo, prev_cp):
    """Global-checkpoint value and its delta vs prev, ignoring reload glitches."""
    cp = get_checkpoint(pinfo)
    delta = cp - prev_cp
    lapsize = pinfo.get("lapsize", 0)
    if lapsize and abs(delta) >= lapsize:
        # A jump of a whole lap (or more) in one step is never real progress:
        # it's a state reload/glitch, or the lap counter settling by 1 at episode
        # start (global checkpoint shifts by exactly lapsize). Ignore it.
        delta = 0
    return cp, delta


class ProgressReward2P(gym.Wrapper):
    """Per-player forward progress: checkpoint delta + lap-completion bonus.

    Ports the single-player checkpoint/lap reward to act on each player's
    info["pN"] dict independently.
    """

    def __init__(self, env, checkpoint_reward=10.0, lap_reward=100.0):
        super().__init__(env)
        self.checkpoint_reward = checkpoint_reward
        self.lap_reward = lap_reward
        self._prev_cp = [0, 0]
        self._cur_lap = [0, 0]
        self._started = [False, False]

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        for p, key in enumerate(_PLAYER_KEYS):
            self._prev_cp[p] = get_checkpoint(info[key])
            self._cur_lap[p] = max(0, info[key].get("lap", 128) - 128)
            self._started[p] = info[key].get("speed", 0) != 0
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        rewards = _player_rewards(info)
        for p, key in enumerate(_PLAYER_KEYS):
            pinfo = info[key]
            if not self._started[p]:
                # Pre-race countdown: keep baselines synced and emit no reward
                # until this kart first moves. Avoids spurious deltas while the
                # lap/checkpoint counters settle at episode start.
                if pinfo.get("speed", 0):
                    self._started[p] = True
                self._prev_cp[p] = get_checkpoint(pinfo)
                self._cur_lap[p] = max(0, pinfo.get("lap", 128) - 128)
                continue
            cp, delta = _progress_delta(pinfo, self._prev_cp[p])
            self._prev_cp[p] = cp
            rewards[p] += delta * self.checkpoint_reward

            lap = max(0, pinfo.get("lap", 128) - 128)
            rewards[p] += self.lap_reward * (lap - self._cur_lap[p])
            self._cur_lap[p] = lap
        return obs, float(rewards.sum()), terminated, truncated, info


class RelativeProgressReward(gym.Wrapper):
    """Zero-sum competitive term: reward each player for out-progressing the other.

    r0 += coef * (d0 - d1); r1 -= coef * (d0 - d1), where dN is player N's
    global-checkpoint delta this step. This is the head-to-head incentive.
    """

    def __init__(self, env, coef=1.0):
        super().__init__(env)
        self.coef = coef
        self._prev_cp = [0, 0]
        self._started = [False, False]

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        for p, key in enumerate(_PLAYER_KEYS):
            self._prev_cp[p] = get_checkpoint(info[key])
            self._started[p] = info[key].get("speed", 0) != 0
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        rewards = _player_rewards(info)
        deltas = [0.0, 0.0]
        for p, key in enumerate(_PLAYER_KEYS):
            pinfo = info[key]
            if not self._started[p]:
                # No competitive credit before this kart starts moving.
                if pinfo.get("speed", 0):
                    self._started[p] = True
                self._prev_cp[p] = get_checkpoint(pinfo)
                continue
            cp, delta = _progress_delta(pinfo, self._prev_cp[p])
            self._prev_cp[p] = cp
            deltas[p] = delta
        rel = self.coef * (deltas[0] - deltas[1])
        rewards[0] += rel
        rewards[1] -= rel
        return obs, float(rewards.sum()), terminated, truncated, info


class LearnerStuckTermination(gym.Wrapper):
    """Terminate the episode only when the LEARNER stops making progress.

    Keyed solely on ``learner_idx``; an opponent stall never terminates (the
    learner keeps racing). On a learner stall a large one-time ``stuck_penalty``
    is added to the learner's reward, sized to dominate any loss it could
    accumulate by racing on, so bailing out is never the better option.
    The penalty is applied pre-scaling (RewardScaling2P scales it afterwards).
    """

    def __init__(self, env, learner_idx=0, max_no_progress_steps=600, stuck_penalty=-1000.0):
        super().__init__(env)
        self.learner_idx = learner_idx
        self.max_no_progress_steps = max_no_progress_steps
        self.stuck_penalty = stuck_penalty
        self._frames_no_progress = 0
        self._max_cp = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._max_cp = get_checkpoint(info[_PLAYER_KEYS[self.learner_idx]])
        self._frames_no_progress = 0
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        cp = get_checkpoint(info[_PLAYER_KEYS[self.learner_idx]])
        if cp > self._max_cp:
            self._max_cp = cp
            self._frames_no_progress = 0
        else:
            self._frames_no_progress += 1

        if self._frames_no_progress >= self.max_no_progress_steps:
            terminated = True
            rewards = _player_rewards(info)
            rewards[self.learner_idx] += self.stuck_penalty
            reward = float(rewards.sum())
        return obs, reward, terminated, truncated, info


class RewardScaling2P(gym.Wrapper):
    """Scale the per-player reward vector by a constant (PPO stability)."""

    def __init__(self, env, scale=0.01):
        super().__init__(env)
        self.scale = scale

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        rewards = _player_rewards(info)
        rewards *= self.scale
        return obs, float(rewards.sum()), terminated, truncated, info


class LearnerScalarReward(gym.Wrapper):
    """Expose only the learner's reward through the standard scalar channel.

    Phase 2 trains a single learner (``learner_idx``); the opponent is frozen, so
    its reward is never used for learning. This collapses the per-player reward
    vector to the scalar reward the training loop already consumes (so the loop
    needs no per-player reward plumbing and is unaffected by vector-env autoreset
    info quirks). The full ``info["rewards"]`` vector is left intact for tooling.

    Also annotates info with each side's global checkpoint (``learner_cp`` /
    ``opp_cp``) so the loop can log head-to-head win-rate cheaply.
    """

    def __init__(self, env, learner_idx=0):
        super().__init__(env)
        self.learner_idx = learner_idx
        self.opp_idx = 1 - learner_idx

    def _annotate(self, info):
        info["learner_cp"] = get_checkpoint(info[_PLAYER_KEYS[self.learner_idx]])
        info["opp_cp"] = get_checkpoint(info[_PLAYER_KEYS[self.opp_idx]])

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._annotate(info)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        rewards = info.get("rewards")
        learner_reward = float(rewards[self.learner_idx]) if rewards is not None else float(reward)
        self._annotate(info)
        return obs, learner_reward, terminated, truncated, info
