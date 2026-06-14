"""Visual 2-player validator with a live per-player reward HUD.

Loads a 2P state, applies the full Phase-2 reward stack (progress + zero-sum
relative + learner-stuck + scaling) and shows each player's reward live.

Two modes:

  # interactive: drive both karts (same keys as tools/play_2p.py); the
  # split-screen renders in a window and rewards stream to the console.
  python -m tools.watch_2p

  # headless: no window; a scripted policy (P1 accelerates, P2 idles so you
  # can see the relative reward go +/-) runs N steps, printing the HUD. Good
  # for a quick check without a display.
  python -m tools.watch_2p --headless 400

Run the interactive mode from WSL (Windows 11 WSLg provides the window).
"""
import argparse
import os

import numpy as np
import gymnasium as gym
import stable_retro as retro

from src.wrapper import get_checkpoint
from src.wrapper_2p import (
    split_player_info,
    ProgressReward2P,
    RelativeProgressReward,
    LearnerStuckTermination,
    RewardScaling2P,
)

GAME = "SuperMarioKart-Snes"
HERE = os.path.dirname(os.path.abspath(__file__))
CUSTOM = os.path.abspath(os.path.join(HERE, "..", "custom_integrations"))

# Same keyboard layout as tools/play_2p.py (button name -> pyglet key name).
P1_MAP = {
    "B": "Z", "Y": "X", "A": "A", "X": "S", "L": "Q", "R": "W",
    "UP": "UP", "DOWN": "DOWN", "LEFT": "LEFT", "RIGHT": "RIGHT",
    "START": "ENTER", "SELECT": "TAB",
}
P2_MAP = {
    "B": "SPACE", "Y": "COMMA", "A": "PERIOD", "X": "SLASH", "L": "U", "R": "O",
    "UP": "I", "DOWN": "K", "LEFT": "J", "RIGHT": "L",
    "START": "RSHIFT", "SELECT": "BACKSPACE",
}


class _RawPlayerInfo(gym.Wrapper):
    """Pass raw 24-bit MultiBinary actions through; attach per-player info."""

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        info["p0"], info["p1"] = split_player_info(info)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info["p0"], info["p1"] = split_player_info(info)
        return obs, reward, terminated, truncated, info


class RewardHUD(gym.Wrapper):
    """Print each player's per-step + cumulative reward and checkpoint."""

    def __init__(self, env, every=6):
        super().__init__(env)
        self.every = every
        self.cum = np.zeros(2, dtype=np.float64)
        self.t = 0

    def reset(self, **kwargs):
        self.cum[:] = 0.0
        self.t = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        r = np.asarray(info["rewards"], dtype=np.float64)
        self.cum += r
        self.t += 1
        if self.t % self.every == 0 or terminated or truncated:
            cp0 = get_checkpoint(info["p0"])
            cp1 = get_checkpoint(info["p1"])
            print(
                f"t={self.t:4d} | "
                f"P1 r={r[0]:+.3f} cum={self.cum[0]:+8.3f} cp={cp0:3d} | "
                f"P2 r={r[1]:+.3f} cum={self.cum[1]:+8.3f} cp={cp1:3d}"
                + ("  [TERMINATED]" if terminated or truncated else "")
            )
        return obs, reward, terminated, truncated, info


def make_wrapped(state, hud_every=6, stuck_steps=600):
    """Raw players=2 env + per-player info + full reward stack + HUD."""
    retro.data.Integrations.add_custom_path(CUSTOM)
    raw = retro.make(
        game=GAME, state=state, players=2,
        render_mode="rgb_array", inttype=retro.data.Integrations.ALL,
    )
    env = _RawPlayerInfo(raw)
    env = ProgressReward2P(env)
    env = RelativeProgressReward(env, coef=1.0)
    env = LearnerStuckTermination(env, learner_idx=0, max_no_progress_steps=stuck_steps)
    env = RewardScaling2P(env, scale=0.01)
    env = RewardHUD(env, every=hud_every)
    return raw, env


def run_headless(state, steps):
    # P1 accelerates (B), P2 idles -> P1 pulls ahead so the relative reward is
    # visibly asymmetric, and the opponent stall does NOT terminate (only the
    # learner's would).
    _, env = make_wrapped(state, hud_every=10, stuck_steps=10_000)
    env.reset()
    action = np.zeros(24, dtype=np.int8)
    action[0] = 1  # P1 accelerate; P2 block (indices 12-23) all zero
    print("Headless: P1 accelerates, P2 idles. Watch P1 cum climb, P2 go negative.")
    for _ in range(steps):
        _, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break
    env.close()


def run_interactive(state):
    from stable_retro.examples.interactive import Interactive

    class TwoPlayerWatch(Interactive):
        def __init__(self):
            raw, env = make_wrapped(state, hud_every=6)
            self._buttons = raw.buttons
            super().__init__(env=env, sync=False, tps=60, aspect_ratio=4 / 3)

        def get_image(self, _obs, env):
            return env.render()

        def keys_to_act(self, keys):
            p1 = {b: (P1_MAP[b] in keys) for b in self._buttons}
            p2 = {b: (P2_MAP[b] in keys) for b in self._buttons}
            return [p1[b] for b in self._buttons] + [p2[b] for b in self._buttons]

    print("Drive: P1 = arrows + Z (accel), P2 = I/J/K/L + SPACE (accel). ESC quits.")
    TwoPlayerWatch().run()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", default="MarioCircuit_2P")
    ap.add_argument("--headless", type=int, default=0,
                    help="run N steps with a scripted policy and no window")
    args = ap.parse_args()

    if args.headless:
        run_headless(args.state, args.headless)
    else:
        run_interactive(args.state)


if __name__ == "__main__":
    main()
