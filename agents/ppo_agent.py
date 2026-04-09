"""
PPO (Proximal Policy Optimization) Agent for Super Mario Kart.

This agent is used as a comparison baseline against the custom DQN implementation.
It uses Stable Baselines3's PPO with a custom CNN feature extractor configured
for our preprocessed 4-frame stacked grayscale observations.
"""

# Block the broken system tensorboard BEFORE stable_baselines3 imports it.
# The cluster's system tensorboard was compiled against NumPy 1.x and crashes
# on NumPy 2.x. We replace it with the user-installed version.
import sys as _sys
_sys.modules.pop('tensorboard', None)
_sys.modules.pop('tensorboard.lazy', None)
# Remove miniconda paths so user-installed packages take priority
_sys.path = [p for p in _sys.path
             if '/usr/pkg/miniconda' not in p and '/usr/pkg/python' not in p]

import os
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
import gymnasium as gym

from wrapper import DiscreteActionWrapper, MarioResize, MarioToPyTorch, MaxAndSkipEnv
from gymnasium.wrappers import FrameStackObservation

# Mirror the DQN action set for fair comparison
SIMPLE_ACTIONS = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 0: No Action
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 1: Gas
    [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # 2: Gas + Left
    [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # 3: Gas + Right
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # 4: Brake
]


class MarioCNN(BaseFeaturesExtractor):
    """
    Custom CNN feature extractor matching the DQN NeuralNet architecture.
    Input: (4, 84, 84) stacked grayscale frames.
    Output: 256-dim feature vector fed into PPO's actor/critic heads.
    """
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        self.cnn = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            sample = torch.zeros(1, *observation_space.shape)
            n_flatten = self.cnn(sample).shape[1]
        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))


class PPORewardLogger(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self._ep_reward = 0.0
        self._ep_length = 0

    def _on_step(self) -> bool:
        self._ep_reward += self.locals["rewards"][0]
        self._ep_length += 1
        if self.locals.get("dones", [False])[0]:
            self.episode_rewards.append(self._ep_reward)
            self.episode_lengths.append(self._ep_length)
            self._ep_reward = 0.0
            self._ep_length = 0
        return True


class PPO_Agent:
    """Wrapper around Stable Baselines3 PPO configured for Mario Kart."""

    def __init__(self, env, learning_rate=2.5e-4, n_steps=2048, batch_size=64,
                 n_epochs=10, gamma=0.99, gae_lambda=0.95, clip_range=0.2,
                 ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5, verbose=1):
        self.env = env
        self.logger_callback = PPORewardLogger()
        policy_kwargs = dict(
            features_extractor_class=MarioCNN,
            features_extractor_kwargs=dict(features_dim=256),
            net_arch=dict(pi=[128], vf=[128]),
        )
        self.model = PPO(
            "CnnPolicy", env,
            learning_rate=learning_rate, n_steps=n_steps, batch_size=batch_size,
            n_epochs=n_epochs, gamma=gamma, gae_lambda=gae_lambda,
            clip_range=clip_range, ent_coef=ent_coef, vf_coef=vf_coef,
            max_grad_norm=max_grad_norm, policy_kwargs=policy_kwargs,
            verbose=verbose, tensorboard_log=None,  # disable TB to avoid conflicts
        )

    @staticmethod
    def wrap_env(env):
        env = MarioResize(env)
        env = MaxAndSkipEnv(env, skip=4)
        env = FrameStackObservation(env, 4)
        env = MarioToPyTorch(env)
        action_map = [np.array(a, dtype=np.int8) for a in SIMPLE_ACTIONS]
        env = DiscreteActionWrapper(env, action_map=action_map)
        env = Monitor(env)
        return env

    def train(self, total_timesteps=500_000):
        self.model.learn(total_timesteps=total_timesteps,
                         callback=self.logger_callback, progress_bar=True)

    def action_select(self, state):
        action, _ = self.model.predict(state, deterministic=True)
        return action

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save(path)
        print(f"PPO model saved to {path}")

    def load(self, path):
        self.model = PPO.load(path, env=self.env)
        print(f"PPO model loaded from {path}")

    @property
    def episode_rewards(self):
        return self.logger_callback.episode_rewards

    @property
    def episode_lengths(self):
        return self.logger_callback.episode_lengths
