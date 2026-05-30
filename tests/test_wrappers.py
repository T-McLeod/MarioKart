import numpy as np
import gymnasium as gym
import pytest

from src.wrapper import (
    get_checkpoint,
    MarioResize,
    MarioToPyTorch,
    DiscreteActionWrapper,
    MaxAndSkipEnv,
    EarlyTermination,
    SpeedReward,
    CompleteLapReward,
    RewardScaling,
)
from tests.helpers import MockRetroEnv


# ---------------------------------------------------------------------------
# get_checkpoint
# ---------------------------------------------------------------------------

def test_get_checkpoint_lap_zero():
    info = {"current_checkpoint": 5, "lapsize": 10, "lap": 128}
    assert get_checkpoint(info) == 5  # lap=0, global = 5 + 0*10


def test_get_checkpoint_second_lap():
    info = {"current_checkpoint": 3, "lapsize": 10, "lap": 129}
    assert get_checkpoint(info) == 13  # lap=1, global = 3 + 1*10


def test_get_checkpoint_missing_keys():
    # Defaults: current_checkpoint=0, lapsize=0, lap=128 → global=0
    assert get_checkpoint({}) == 0


# ---------------------------------------------------------------------------
# MarioResize
# ---------------------------------------------------------------------------

def test_mario_resize_output_shape():
    env = MarioResize(MockRetroEnv(obs_shape=(224, 256, 3)))
    obs, _ = env.reset()
    assert obs.shape == (84, 84, 1)
    assert obs.dtype == np.uint8


def test_mario_resize_step_shape():
    env = MarioResize(MockRetroEnv(obs_shape=(224, 256, 3)))
    env.reset()
    obs, _, _, _, _ = env.step(np.zeros(12, dtype=np.int8))
    assert obs.shape == (84, 84, 1)


# ---------------------------------------------------------------------------
# MarioToPyTorch
# ---------------------------------------------------------------------------

class _StackedEnv(gym.Env):
    """Mock environment whose obs matches post-FrameStack shape (4, 84, 84, 1)."""

    def __init__(self, fill=128):
        super().__init__()
        self.observation_space = gym.spaces.Box(0, 255, shape=(4, 84, 84, 1), dtype=np.uint8)
        self.action_space = gym.spaces.Discrete(5)
        self._fill = fill

    def reset(self, **kwargs):
        return np.full((4, 84, 84, 1), self._fill, dtype=np.uint8), {}

    def step(self, action):
        return np.full((4, 84, 84, 1), self._fill, dtype=np.uint8), 0.0, False, False, {}


def test_mario_to_pytorch_shape_and_dtype():
    env = MarioToPyTorch(_StackedEnv())
    obs, _ = env.reset()
    assert obs.shape == (4, 84, 84)
    assert obs.dtype == np.float32


def test_mario_to_pytorch_normalizes_to_unit_range():
    env = MarioToPyTorch(_StackedEnv(fill=128))
    obs, _ = env.reset()
    np.testing.assert_allclose(obs, 128.0 / 255.0, rtol=1e-5)


def test_mario_to_pytorch_zeros_stay_zero():
    env = MarioToPyTorch(_StackedEnv(fill=0))
    obs, _ = env.reset()
    assert obs.max() == 0.0


def test_mario_to_pytorch_full_maps_to_one():
    env = MarioToPyTorch(_StackedEnv(fill=255))
    obs, _ = env.reset()
    np.testing.assert_allclose(obs, 1.0, rtol=1e-5)


# ---------------------------------------------------------------------------
# DiscreteActionWrapper
# ---------------------------------------------------------------------------

ACTIONS = [
    np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int8),
    np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int8),
    np.array([1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], dtype=np.int8),
]


def test_discrete_action_wrapper_space_size():
    env = DiscreteActionWrapper(MockRetroEnv(), action_map=ACTIONS)
    assert env.action_space.n == len(ACTIONS)


def test_discrete_action_wrapper_maps_correctly():
    env = DiscreteActionWrapper(MockRetroEnv(), action_map=ACTIONS)
    for i, expected in enumerate(ACTIONS):
        np.testing.assert_array_equal(env.action(i), expected)


# ---------------------------------------------------------------------------
# MaxAndSkipEnv
# ---------------------------------------------------------------------------

def test_max_skip_accumulates_reward():
    env = MaxAndSkipEnv(MockRetroEnv(base_reward=1.0), skip=4)
    env.reset()
    _, reward, _, _, _ = env.step(np.zeros(12))
    assert reward == 4.0


def test_max_skip_two_skips():
    env = MaxAndSkipEnv(MockRetroEnv(base_reward=2.0), skip=3)
    env.reset()
    _, reward, _, _, _ = env.step(np.zeros(12))
    assert reward == 6.0


def test_max_skip_stops_on_termination():
    class _TerminatesAt(MockRetroEnv):
        def __init__(self, terminate_on_step):
            super().__init__(base_reward=1.0)
            self._n = 0
            self._terminate_on = terminate_on_step

        def step(self, action):
            self._n += 1
            terminated = self._n == self._terminate_on
            return np.zeros((224, 256, 3), dtype=np.uint8), 1.0, terminated, False, {}

    base = _TerminatesAt(terminate_on_step=2)
    env = MaxAndSkipEnv(base, skip=4)
    env.reset()
    _, reward, terminated, _, _ = env.step(np.zeros(12))
    assert terminated
    assert reward == 2.0  # only 2 inner steps ran
    assert base._n == 2


def test_max_skip_stops_on_truncation():
    class _TruncatesAt(MockRetroEnv):
        def __init__(self):
            super().__init__(base_reward=1.0)
            self._n = 0

        def step(self, action):
            self._n += 1
            truncated = self._n == 3
            return np.zeros((224, 256, 3), dtype=np.uint8), 1.0, False, truncated, {}

    base = _TruncatesAt()
    env = MaxAndSkipEnv(base, skip=4)
    env.reset()
    _, reward, _, truncated, _ = env.step(np.zeros(12))
    assert truncated
    assert reward == 3.0


# ---------------------------------------------------------------------------
# EarlyTermination
# ---------------------------------------------------------------------------

class _StuckEnv(MockRetroEnv):
    """Always reports checkpoint=0 (agent is stuck)."""

    def step(self, action):
        obs, _, term, trunc, _ = super().step(action)
        return obs, 0.0, term, trunc, {"current_checkpoint": 0, "lapsize": 10, "lap": 128}


def test_early_termination_triggers_after_threshold():
    env = EarlyTermination(_StuckEnv(), max_no_progress_steps=5, stuck_penalty=-10)
    env.reset()
    for _ in range(4):
        _, _, terminated, _, _ = env.step(np.zeros(12))
        assert not terminated
    _, reward, terminated, _, _ = env.step(np.zeros(12))
    assert terminated
    assert reward == -10.0


def test_early_termination_resets_on_progress():
    class _ProgressEnv(MockRetroEnv):
        def __init__(self):
            super().__init__()
            self._step = 0

        def step(self, action):
            self._step += 1
            cp = self._step  # always advancing
            obs, _, term, trunc, _ = super().step(action)
            return obs, 0.0, term, trunc, {"current_checkpoint": cp, "lapsize": 10, "lap": 128}

    env = EarlyTermination(_ProgressEnv(), max_no_progress_steps=3, stuck_penalty=-10)
    env.reset()
    for _ in range(10):
        _, _, terminated, _, _ = env.step(np.zeros(12))
        assert not terminated  # never gets stuck


def test_early_termination_reset_clears_counter():
    env = EarlyTermination(_StuckEnv(), max_no_progress_steps=3, stuck_penalty=-10)
    env.reset()
    # Take 2 steps without progress
    env.step(np.zeros(12))
    env.step(np.zeros(12))
    # Reset mid-episode — counter must clear
    env.reset()
    for _ in range(2):
        _, _, terminated, _, _ = env.step(np.zeros(12))
        assert not terminated  # only 2 steps, threshold is 3


# ---------------------------------------------------------------------------
# SpeedReward
# ---------------------------------------------------------------------------

class _SpeedEnv(MockRetroEnv):
    def __init__(self, speed):
        super().__init__(base_reward=0.0)
        self._speed = speed

    def step(self, action):
        obs, r, term, trunc, _ = super().step(action)
        return obs, r, term, trunc, {"kart1_speed": self._speed}


def test_speed_reward_adds_scaled_bonus():
    env = SpeedReward(_SpeedEnv(speed=100.0), scale=0.01)
    env.reset()
    _, reward, _, _, _ = env.step(np.zeros(12))
    np.testing.assert_allclose(reward, 1.0, rtol=1e-5)  # 0 + 100 * 0.01


def test_speed_reward_zero_speed_no_bonus():
    env = SpeedReward(_SpeedEnv(speed=0.0), scale=0.01)
    env.reset()
    _, reward, _, _, _ = env.step(np.zeros(12))
    assert reward == 0.0


def test_speed_reward_missing_key_defaults_zero():
    env = SpeedReward(MockRetroEnv(base_reward=0.0), scale=0.01)
    env.reset()
    _, reward, _, _, _ = env.step(np.zeros(12))
    assert reward == 0.0


# ---------------------------------------------------------------------------
# CompleteLapReward
# ---------------------------------------------------------------------------

class _LapEnv(MockRetroEnv):
    def __init__(self):
        super().__init__(base_reward=0.0)
        self.lap_raw = 128  # encodes lap 0

    def step(self, action):
        obs, _, term, trunc, _ = super().step(action)
        return obs, 0.0, term, trunc, {"lap": self.lap_raw, "lapsize": 10, "current_checkpoint": 0}


def test_complete_lap_no_change_gives_no_bonus():
    env = CompleteLapReward(_LapEnv(), lap_reward=1000)
    env.reset()
    _, reward, _, _, _ = env.step(np.zeros(12))
    assert reward == 0.0


def test_complete_lap_first_lap_completion():
    inner = _LapEnv()
    env = CompleteLapReward(inner, lap_reward=1000)
    env.reset()
    inner.lap_raw = 129  # lap = 1
    _, reward, _, _, _ = env.step(np.zeros(12))
    assert reward == 1000.0


def test_complete_lap_reset_clears_lap_counter():
    inner = _LapEnv()
    env = CompleteLapReward(inner, lap_reward=1000)
    inner.lap_raw = 129
    env.reset()
    # After reset current_lap == 0; now lap_raw goes back to 128 (lap=0)
    inner.lap_raw = 128
    _, reward, _, _, _ = env.step(np.zeros(12))
    assert reward == 0.0


# ---------------------------------------------------------------------------
# RewardScaling
# ---------------------------------------------------------------------------

def test_reward_scaling_multiplies():
    env = RewardScaling(MockRetroEnv(base_reward=10.0), scale=0.1)
    env.reset()
    _, reward, _, _, _ = env.step(np.zeros(12))
    np.testing.assert_allclose(reward, 1.0, rtol=1e-6)


def test_reward_scaling_negative():
    env = RewardScaling(MockRetroEnv(base_reward=-5.0), scale=0.2)
    env.reset()
    _, reward, _, _, _ = env.step(np.zeros(12))
    np.testing.assert_allclose(reward, -1.0, rtol=1e-6)
