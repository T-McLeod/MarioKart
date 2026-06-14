"""Shared test utilities — import in test files, not via fixture system."""
import numpy as np
import gymnasium as gym


class MockRetroEnv(gym.Env):
    """Minimal stable_retro stand-in: RGB obs (224×256×3), MultiBinary(12) actions."""

    def __init__(self, obs_shape=(224, 256, 3), base_reward=1.0, base_info=None):
        super().__init__()
        self.observation_space = gym.spaces.Box(0, 255, shape=obs_shape, dtype=np.uint8)
        self.action_space = gym.spaces.MultiBinary(12)
        self._base_reward = base_reward
        self._base_info = base_info or {}

    def reset(self, **kwargs):
        return np.zeros(self.observation_space.shape, dtype=np.uint8), {}

    def step(self, action):
        obs = np.zeros(self.observation_space.shape, dtype=np.uint8)
        return obs, self._base_reward, False, False, dict(self._base_info)

    def render(self):
        return np.zeros(self.observation_space.shape, dtype=np.uint8)


class Mock2PEnv(gym.Env):
    """Stand-in for stable_retro with players=2.

    Split-screen RGB obs (224×256×3) with a constant top half (Player 1) and a
    distinct constant bottom half (Player 2) so crop assignment is verifiable.
    Action space is MultiBinary(24); the last action received is recorded so the
    de-multiplexing of the joint action can be asserted. ``terminate_on`` makes
    step() report terminated once that many inner steps have run (for frame-skip
    break tests).
    """

    def __init__(self, top_fill=200, bottom_fill=50, base_info=None, terminate_on=None):
        super().__init__()
        self.observation_space = gym.spaces.Box(0, 255, shape=(224, 256, 3), dtype=np.uint8)
        self.action_space = gym.spaces.MultiBinary(24)
        self._top = top_fill
        self._bottom = bottom_fill
        self._base_info = base_info or {}
        self._terminate_on = terminate_on
        self.steps = 0
        self.last_action = None

    def _frame(self):
        f = np.zeros((224, 256, 3), dtype=np.uint8)
        f[:112] = self._top
        f[112:] = self._bottom
        return f

    def reset(self, **kwargs):
        self.steps = 0
        return self._frame(), dict(self._base_info)

    def step(self, action):
        self.steps += 1
        self.last_action = np.asarray(action)
        terminated = self._terminate_on is not None and self.steps >= self._terminate_on
        return self._frame(), 1.0, terminated, False, dict(self._base_info)

    def render(self):
        return self._frame()
