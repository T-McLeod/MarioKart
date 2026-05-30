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
