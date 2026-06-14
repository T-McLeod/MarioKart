import numpy as np
import gymnasium as gym

from src.wrapper import get_checkpoint
from src.wrapper_2p import (
    TwoPlayerMarioEnv,
    ProgressReward2P,
    RelativeProgressReward,
    LearnerStuckTermination,
    RewardScaling2P,
    LearnerScalarReward,
)
from src.agents.ppo_nature.agent import DISCOVERY_ACTIONS
from tests.helpers import Mock2PEnv

# Small, distinct action table so the joint-action de-mux is verifiable.
TEST_ACTIONS = [
    np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int8),  # idx 0
    np.array([0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], dtype=np.int8),  # idx 1
    np.array([1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], dtype=np.int8),  # idx 2
]


def _make(env=None, **kwargs):
    kwargs.setdefault("action_map", TEST_ACTIONS)
    return TwoPlayerMarioEnv(env or Mock2PEnv(), **kwargs)


# ---------------------------------------------------------------------------
# spaces
# ---------------------------------------------------------------------------

def test_2p_action_space_is_multidiscrete_per_player():
    env = _make()
    assert isinstance(env.action_space, gym.spaces.MultiDiscrete)
    np.testing.assert_array_equal(env.action_space.nvec, [3, 3])
    assert env.num_actions == 3


def test_2p_observation_space_shape():
    env = _make()
    assert env.observation_space.shape == (2, 4, 84, 84)
    assert env.observation_space.dtype == np.float32


def test_2p_custom_stack_depth_reflected_in_space():
    env = _make(num_stack=6)
    assert env.observation_space.shape == (2, 6, 84, 84)


# ---------------------------------------------------------------------------
# observation: shape / dtype / range
# ---------------------------------------------------------------------------

def test_2p_reset_obs_shape_and_range():
    env = _make()
    obs, info = env.reset()
    assert obs.shape == (2, 4, 84, 84)
    assert obs.dtype == np.float32
    assert obs.min() >= 0.0 and obs.max() <= 1.0


def test_2p_step_obs_shape_and_zero_reward():
    env = _make()
    env.reset()
    obs, reward, terminated, truncated, info = env.step([0, 0])
    assert obs.shape == (2, 4, 84, 84)
    assert reward == 0.0  # Python reward wrappers own shaping
    assert not terminated and not truncated


# ---------------------------------------------------------------------------
# split-screen crop: top half -> P1 (obs[0]), bottom half -> P2 (obs[1])
# ---------------------------------------------------------------------------

def test_2p_split_crop_assigns_top_to_p1_bottom_to_p2():
    # greyscale of a constant grey (v,v,v) is v; obs is normalised by 255.
    env = _make(Mock2PEnv(top_fill=200, bottom_fill=50))
    obs, _ = env.reset()
    np.testing.assert_allclose(obs[0], 200 / 255.0, atol=2 / 255.0)
    np.testing.assert_allclose(obs[1], 50 / 255.0, atol=2 / 255.0)


def test_2p_halves_differ_when_views_differ():
    env = _make(Mock2PEnv(top_fill=200, bottom_fill=50))
    obs, _ = env.reset()
    assert not np.array_equal(obs[0], obs[1])


# ---------------------------------------------------------------------------
# action de-multiplexing: [a0, a1] -> concat of the two 12-bit button arrays
# ---------------------------------------------------------------------------

def test_2p_action_demux_concatenates_button_arrays():
    mock = Mock2PEnv()
    env = TwoPlayerMarioEnv(mock, action_map=TEST_ACTIONS, skip=1)
    env.reset()
    env.step([0, 2])
    expected = np.concatenate([TEST_ACTIONS[0], TEST_ACTIONS[2]])
    assert mock.last_action.shape == (24,)
    np.testing.assert_array_equal(mock.last_action, expected)


def test_2p_action_demux_independent_players():
    mock = Mock2PEnv()
    env = TwoPlayerMarioEnv(mock, action_map=TEST_ACTIONS, skip=1)
    env.reset()
    env.step([1, 0])
    expected = np.concatenate([TEST_ACTIONS[1], TEST_ACTIONS[0]])
    np.testing.assert_array_equal(mock.last_action, expected)


# ---------------------------------------------------------------------------
# frame skip
# ---------------------------------------------------------------------------

def test_2p_frame_skip_advances_core_n_times():
    mock = Mock2PEnv()
    env = TwoPlayerMarioEnv(mock, action_map=TEST_ACTIONS, skip=4)
    env.reset()
    env.step([0, 0])
    assert mock.steps == 4


def test_2p_termination_breaks_skip_early():
    mock = Mock2PEnv(terminate_on=2)
    env = TwoPlayerMarioEnv(mock, action_map=TEST_ACTIONS, skip=4)
    env.reset()
    _, _, terminated, _, _ = env.step([0, 0])
    assert terminated
    assert mock.steps == 2  # stopped as soon as the core terminated


# ---------------------------------------------------------------------------
# per-player info split
# ---------------------------------------------------------------------------

BASE_INFO = {
    "current_checkpoint": 5, "lapsize": 10, "lap": 129,
    "kart1_speed": 300, "rank": 4, "surface": 64,
    "kart1_direction": 10, "kart1_X": 11, "kart1_Y": 12,
    "kart2_checkpoint": 7, "kart2_lap": 128,
    "kart2_speed": 250, "kart2_rank": 2, "kart2_surface": 40,
    "kart2_direction": 20, "kart2_X": 21, "kart2_Y": 22,
}


def test_2p_info_splits_into_per_player_dicts():
    env = _make(Mock2PEnv(base_info=BASE_INFO))
    _, info = env.reset()
    assert "p0" in info and "p1" in info

    p0, p1 = info["p0"], info["p1"]
    assert p0["current_checkpoint"] == 5 and p0["speed"] == 300 and p0["rank"] == 4
    assert p0["X"] == 11 and p0["Y"] == 12 and p0["direction"] == 10
    assert p1["current_checkpoint"] == 7 and p1["speed"] == 250 and p1["rank"] == 2
    assert p1["X"] == 21 and p1["Y"] == 22 and p1["direction"] == 20


def test_2p_get_checkpoint_works_on_each_player_dict():
    env = _make(Mock2PEnv(base_info=BASE_INFO))
    _, info = env.reset()
    # p0: lap 129 -> lap 1, global = 5 + 1*10 = 15
    assert get_checkpoint(info["p0"]) == 15
    # p1: lap 128 -> lap 0, global = 7 + 0*10 = 7
    assert get_checkpoint(info["p1"]) == 7


def test_2p_info_present_on_step_too():
    env = _make(Mock2PEnv(base_info=BASE_INFO))
    env.reset()
    _, _, _, _, info = env.step([0, 0])
    assert "p0" in info and "p1" in info


# ---------------------------------------------------------------------------
# frame stacking advances (newest frame appended, oldest dropped)
# ---------------------------------------------------------------------------

class _RampMock2P(Mock2PEnv):
    """Both halves filled with 40*steps so successive frames are distinct."""

    def _frame(self):
        v = min(self.steps * 40, 255)
        return np.full((224, 256, 3), v, dtype=np.uint8)


def test_2p_frame_stack_newest_last_oldest_first():
    env = TwoPlayerMarioEnv(_RampMock2P(), action_map=TEST_ACTIONS, skip=1)
    obs, _ = env.reset()
    # reset fills the stack with the initial (steps=0 -> fill 0) frame
    np.testing.assert_allclose(obs[0], 0.0, atol=2 / 255.0)
    obs, _, _, _, _ = env.step([0, 0])  # steps -> 1 -> fill 40
    np.testing.assert_allclose(obs[0, -1], 40 / 255.0, atol=2 / 255.0)  # newest
    np.testing.assert_allclose(obs[0, 0], 0.0, atol=2 / 255.0)          # oldest


# ---------------------------------------------------------------------------
# default action map is the agent's DISCOVERY_ACTIONS
# ---------------------------------------------------------------------------

def test_2p_default_action_map_matches_discovery_actions():
    env = TwoPlayerMarioEnv(Mock2PEnv())  # no action_map -> default
    assert env.num_actions == len(DISCOVERY_ACTIONS)
    np.testing.assert_array_equal(env.action_space.nvec,
                                  [len(DISCOVERY_ACTIONS), len(DISCOVERY_ACTIONS)])


# ===========================================================================
# Reward wrappers
# ===========================================================================

class _ScriptedPlayersEnv(gym.Env):
    """Emits info['p0']/['p1'] with scripted (checkpoint, lap_raw) per step.

    Each sequence is a list of (current_checkpoint, lap_raw) tuples; the last
    entry is held once the sequence is exhausted.
    """

    def __init__(self, p0_seq, p1_seq, lapsize=10, start_speed_at=0):
        super().__init__()
        self.observation_space = gym.spaces.Box(0.0, 1.0, shape=(2, 4, 84, 84), dtype=np.float32)
        self.action_space = gym.spaces.MultiDiscrete([3, 3])
        self.p0_seq = p0_seq
        self.p1_seq = p1_seq
        self.lapsize = lapsize
        self.start_speed_at = start_speed_at  # speed is 0 until this step index
        self.i = 0

    def _pinfo(self, seq, idx):
        cp, lap = seq[min(idx, len(seq) - 1)]
        speed = 0 if idx < self.start_speed_at else 100
        return {"current_checkpoint": cp, "lap": lap, "lapsize": self.lapsize, "speed": speed}

    def _obs(self):
        return np.zeros((2, 4, 84, 84), dtype=np.float32)

    def _info(self, idx):
        return {"p0": self._pinfo(self.p0_seq, idx), "p1": self._pinfo(self.p1_seq, idx)}

    def reset(self, **kwargs):
        self.i = 0
        return self._obs(), self._info(0)

    def step(self, action):
        self.i += 1
        return self._obs(), 0.0, False, False, self._info(self.i)


# ---- ProgressReward2P -----------------------------------------------------

def test_progress_reward_per_player_checkpoint_delta():
    p0 = [(0, 128), (1, 128), (2, 128)]
    p1 = [(0, 128)]  # opponent parked
    env = ProgressReward2P(_ScriptedPlayersEnv(p0, p1), checkpoint_reward=10.0, lap_reward=100.0)
    env.reset()
    _, _, _, _, info = env.step([0, 0])
    np.testing.assert_allclose(info["rewards"], [10.0, 0.0])
    _, _, _, _, info = env.step([0, 0])
    np.testing.assert_allclose(info["rewards"], [10.0, 0.0])


def test_progress_reward_lap_completion_bonus():
    # cp 9 (lap0) -> cp 0 (lap1): global 9 -> 10 (delta +1 => +10) and lap bonus +100
    p0 = [(9, 128), (0, 129)]
    p1 = [(0, 128)]
    env = ProgressReward2P(_ScriptedPlayersEnv(p0, p1, lapsize=10),
                           checkpoint_reward=10.0, lap_reward=100.0)
    env.reset()
    _, _, _, _, info = env.step([0, 0])
    np.testing.assert_allclose(info["rewards"], [110.0, 0.0])


def test_progress_reward_ignores_reload_glitch():
    # a jump of a whole lap or more in one step is treated as 0 (no reward)
    p0 = [(0, 128), (500, 128)]
    p1 = [(0, 128)]
    env = ProgressReward2P(_ScriptedPlayersEnv(p0, p1, lapsize=10), checkpoint_reward=10.0)
    env.reset()
    _, _, _, _, info = env.step([0, 0])
    np.testing.assert_allclose(info["rewards"], [0.0, 0.0])


def test_progress_reward_ignores_lap_counter_settle():
    # At episode start the lap byte settles by 1, so the global checkpoint jumps
    # by exactly lapsize. That is not real progress and must be ignored.
    p0 = [(27, 128), (27, 127)]  # lapsize 30: global 27 -> -3 (delta == -30)
    p1 = [(27, 128)]
    env = ProgressReward2P(_ScriptedPlayersEnv(p0, p1, lapsize=30), checkpoint_reward=10.0)
    env.reset()
    _, _, _, _, info = env.step([0, 0])
    np.testing.assert_allclose(info["rewards"], [0.0, 0.0])


def test_progress_reward_gated_until_kart_moves():
    # speed is 0 for the first 2 steps (countdown): no reward even though the
    # checkpoint moves; the baseline re-syncs so only post-start progress counts.
    p0 = [(0, 128), (5, 128), (6, 128), (7, 128)]
    p1 = [(0, 128)]
    env = ProgressReward2P(_ScriptedPlayersEnv(p0, p1, lapsize=30, start_speed_at=2),
                           checkpoint_reward=10.0)
    env.reset()
    _, _, _, _, info = env.step([0, 0])   # speed 0 -> gated
    np.testing.assert_allclose(info["rewards"], [0.0, 0.0])
    _, _, _, _, info = env.step([0, 0])   # first moving step -> baseline only
    np.testing.assert_allclose(info["rewards"], [0.0, 0.0])
    _, _, _, _, info = env.step([0, 0])   # cp 6 -> 7 -> +10
    np.testing.assert_allclose(info["rewards"], [10.0, 0.0])


def test_relative_reward_gated_until_kart_moves():
    # While speed is 0 no competitive credit accrues to either side.
    p0 = [(0, 128), (1, 128)]
    p1 = [(0, 128)]
    env = RelativeProgressReward(_ScriptedPlayersEnv(p0, p1, start_speed_at=5), coef=1.0)
    env.reset()
    _, _, _, _, info = env.step([0, 0])
    np.testing.assert_allclose(info["rewards"], [0.0, 0.0])


# ---- RelativeProgressReward ----------------------------------------------

def test_relative_reward_is_zero_sum_leader_vs_follower():
    p0 = [(0, 128), (1, 128)]   # advances
    p1 = [(0, 128)]             # parked
    env = RelativeProgressReward(_ScriptedPlayersEnv(p0, p1), coef=1.0)
    env.reset()
    _, _, _, _, info = env.step([0, 0])
    np.testing.assert_allclose(info["rewards"], [1.0, -1.0])


def test_relative_reward_equal_progress_cancels():
    p0 = [(0, 128), (1, 128)]
    p1 = [(0, 128), (1, 128)]
    env = RelativeProgressReward(_ScriptedPlayersEnv(p0, p1), coef=1.0)
    env.reset()
    _, _, _, _, info = env.step([0, 0])
    np.testing.assert_allclose(info["rewards"], [0.0, 0.0])


def test_relative_reward_respects_coef():
    p0 = [(0, 128), (1, 128)]
    p1 = [(0, 128)]
    env = RelativeProgressReward(_ScriptedPlayersEnv(p0, p1), coef=2.5)
    env.reset()
    _, _, _, _, info = env.step([0, 0])
    np.testing.assert_allclose(info["rewards"], [2.5, -2.5])


# ---- LearnerStuckTermination ---------------------------------------------

def test_learner_stuck_terminates_with_penalty():
    p0 = [(5, 128)]                       # learner parked
    p1 = [(0, 128), (1, 128), (2, 128)]   # opponent racing
    env = LearnerStuckTermination(_ScriptedPlayersEnv(p0, p1),
                                  learner_idx=0, max_no_progress_steps=3,
                                  stuck_penalty=-1000.0)
    env.reset()
    for _ in range(2):
        _, _, terminated, _, _ = env.step([0, 0])
        assert not terminated
    _, _, terminated, _, info = env.step([0, 0])
    assert terminated
    assert info["rewards"][0] == -1000.0


def test_opponent_stall_never_terminates():
    p0 = [(0, 128), (1, 128), (2, 128), (3, 128), (4, 128), (5, 128)]  # learner racing
    p1 = [(7, 128)]                                                    # opponent parked
    env = LearnerStuckTermination(_ScriptedPlayersEnv(p0, p1),
                                  learner_idx=0, max_no_progress_steps=3)
    env.reset()
    for _ in range(6):
        _, _, terminated, _, _ = env.step([0, 0])
        assert not terminated


def test_learner_idx_configurable():
    # learner is player 1; player 1 is parked, player 0 races -> must terminate
    p0 = [(0, 128), (1, 128), (2, 128)]
    p1 = [(5, 128)]
    env = LearnerStuckTermination(_ScriptedPlayersEnv(p0, p1),
                                  learner_idx=1, max_no_progress_steps=2,
                                  stuck_penalty=-500.0)
    env.reset()
    env.step([0, 0])
    _, _, terminated, _, info = env.step([0, 0])
    assert terminated
    assert info["rewards"][1] == -500.0


# ---- RewardScaling2P ------------------------------------------------------

def test_reward_scaling_2p_scales_vector():
    p0 = [(0, 128), (1, 128)]
    p1 = [(0, 128)]
    env = RewardScaling2P(
        ProgressReward2P(_ScriptedPlayersEnv(p0, p1), checkpoint_reward=10.0),
        scale=0.1,
    )
    env.reset()
    _, _, _, _, info = env.step([0, 0])
    np.testing.assert_allclose(info["rewards"], [1.0, 0.0])  # 10 * 0.1


# ---- full stack: ordering + scaled penalty -------------------------------

def test_full_reward_stack_penalty_applied_before_scaling():
    # learner parked, opponent advances 1 checkpoint.
    p0 = [(5, 128)]
    p1 = [(0, 128), (1, 128)]
    env = _ScriptedPlayersEnv(p0, p1, lapsize=10)
    env = ProgressReward2P(env, checkpoint_reward=10.0, lap_reward=100.0)
    env = RelativeProgressReward(env, coef=1.0)
    env = LearnerStuckTermination(env, learner_idx=0, max_no_progress_steps=1,
                                  stuck_penalty=-1000.0)
    env = RewardScaling2P(env, scale=0.01)
    env.reset()
    _, _, terminated, _, info = env.step([0, 0])
    assert terminated
    # progress: r=[0,10]; relative(d0=0,d1=1): r=[-1,11]; stuck penalty: r0-=1000
    # -> [-1001, 11]; scale 0.01 -> [-10.01, 0.11]
    np.testing.assert_allclose(info["rewards"], [-10.01, 0.11], rtol=1e-5)


# ---- LearnerScalarReward --------------------------------------------------

def test_learner_scalar_reward_exposes_learner_only():
    p0 = [(0, 128), (1, 128)]   # learner advances -> +10
    p1 = [(0, 128)]             # opponent parked
    env = LearnerScalarReward(
        ProgressReward2P(_ScriptedPlayersEnv(p0, p1), checkpoint_reward=10.0),
        learner_idx=0,
    )
    env.reset()
    _, scalar, _, _, info = env.step([0, 0])
    assert scalar == 10.0                 # scalar channel == learner's reward
    assert info["rewards"][0] == 10.0     # full vector still available
    assert info["learner_cp"] == 1 and info["opp_cp"] == 0


def test_learner_scalar_reward_respects_learner_idx():
    p0 = [(0, 128)]             # player 0 parked
    p1 = [(0, 128), (1, 128)]   # player 1 advances
    env = LearnerScalarReward(
        ProgressReward2P(_ScriptedPlayersEnv(p0, p1), checkpoint_reward=10.0),
        learner_idx=1,
    )
    env.reset()
    _, scalar, _, _, info = env.step([0, 0])
    assert scalar == 10.0                 # learner is player 1 here
    assert info["learner_cp"] == 1 and info["opp_cp"] == 0


# ===========================================================================
# evaluate_2p_and_record (checkpoint eval / visualization path)
# ===========================================================================

class _StubAgent:
    """Minimal agent stand-in: action_select returns a fixed discrete action."""

    def __init__(self, action=0):
        self._action = action

    def action_select(self, obs):
        return self._action


def test_evaluate_2p_runs_and_reports_metrics():
    from src.eval_2p import evaluate_2p_and_record

    base = Mock2PEnv()
    env = TwoPlayerMarioEnv(base, action_map=TEST_ACTIONS)
    env = ProgressReward2P(env)
    env = RelativeProgressReward(env)
    env = LearnerStuckTermination(env, max_no_progress_steps=10_000)
    env = RewardScaling2P(env)
    env = LearnerScalarReward(env)

    avg_return, avg_length, win_rate = evaluate_2p_and_record(
        _StubAgent(), _StubAgent(), env, video_path=None,
        num_episodes=2, max_timesteps=5,
    )
    assert avg_length == 5           # Mock2PEnv never terminates -> hits the cap
    assert 0.0 <= win_rate <= 1.0
    assert isinstance(avg_return, float)
