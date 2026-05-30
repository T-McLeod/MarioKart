import torch
import os
import torch.nn as nn
import numpy as np
import uuid
from gymnasium.wrappers import FrameStackObservation
from ..base import BaseAgent
from .network import ImpalaCNN

from ...wrapper import (
    DebugObservation,
    DiscreteActionWrapper,
    MarioResize,
    MarioToPyTorch,
    MaxAndSkipEnv,
    EarlyTermination,
    CompleteLapReward,
    RewardScaling,
)


if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")


DISCOVERY_ACTIONS = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],

    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],

    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],

    [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    [1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0],

    [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
]


class PPOImpalaAgent(BaseAgent):
    def __init__(
        self,
        env,
        discount=0.99,
        learning_rate=5e-5,
        rollout_steps=2048,
        minibatch_size=256,
        n_epochs=4,
        clip_coef=0.1,
        vf_coef=0.5,
        ent_coef_start=0.03,
        ent_coef_end=0.01,
        gae_lambda=0.95,
        max_grad_norm=0.5,
        total_timesteps=3_000_000,
        no_improve_tolerance=999999,
        verbose=False,
        **kwargs
    ):
        super().__init__(env, total_timesteps=total_timesteps, **kwargs)
        self.discount = discount
        self.learning_rate = learning_rate
        self.rollout_steps = rollout_steps
        self.minibatch_size = minibatch_size
        self.n_epochs = n_epochs
        self.clip_coef = clip_coef
        self.vf_coef = vf_coef
        self.ent_coef_start = ent_coef_start
        self.ent_coef_end = ent_coef_end
        self.gae_lambda = gae_lambda
        self.max_grad_norm = max_grad_norm
        self.verbose = verbose

        self.no_improve_tolerance = no_improve_tolerance
        self.best_avg_return = float("-inf")
        self.intervals_without_improvement = 0
        self.should_stop = False

        self.custom_metrics = {}

        self.action_set = DISCOVERY_ACTIONS
        self.num_actions = len(self.action_set)

        self.ac_net = ImpalaCNN(self.num_actions).to(device)
        self.optimizer = torch.optim.Adam(
            self.ac_net.parameters(),
            lr=learning_rate,
            eps=1e-5,
        )

        self._init_rollout_buffer()
        self._cached_log_prob = None
        self._cached_value = None

    def _init_rollout_buffer(self):
        self._rb_states = []
        self._rb_actions = []
        self._rb_log_probs = []
        self._rb_rewards = []
        self._rb_values = []
        self._rb_dones = []

    def record_return(self, avg_return):
        if avg_return > self.best_avg_return + 1.0:
            self.best_avg_return = avg_return
            self.intervals_without_improvement = 0
        else:
            self.intervals_without_improvement += 1

        if self.intervals_without_improvement >= self.no_improve_tolerance:
            print(
                f"Early stopping: no improvement for "
                f"{self.no_improve_tolerance} intervals. "
                f"Best avg return: {self.best_avg_return:.2f}"
            )
            self.should_stop = True

    def action_select(self, state):
        is_single = len(state.shape) == 3
        if is_single:
            state = np.expand_dims(state, axis=0)

        state_t = torch.tensor(state, dtype=torch.float32).contiguous().to(device)

        with torch.no_grad():
            action_t, log_prob_t, _, value_t = self.ac_net.get_action_and_value(state_t)

        self._cached_log_prob = log_prob_t.cpu().numpy()
        self._cached_value = value_t.view(-1).cpu().numpy()

        actions = action_t.cpu().numpy()
        return actions[0] if is_single else actions

    def update(self, state, action, reward, next_state, done):
        self._rb_states.append(state)
        self._rb_actions.append(action)
        self._rb_log_probs.append(self._cached_log_prob)
        self._rb_rewards.append(np.array(reward, dtype=np.float32))
        self._rb_values.append(self._cached_value)
        self._rb_dones.append(np.array(done, dtype=np.float32))

        num_envs = len(state)
        self.steps += num_envs

        if len(self._rb_states) * num_envs >= self.rollout_steps:
            self._ppo_update(next_state, done)
            self._init_rollout_buffer()

    def _ppo_update(self, last_next_state, last_done):
        progress = min(self.steps / self.total_timesteps, 1.0)

        ent_coef = self.ent_coef_start + progress * (
            self.ent_coef_end - self.ent_coef_start
        )

        lr = max(self.learning_rate * (1.0 - progress), 1e-6)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

        with torch.no_grad():
            last_state_t = torch.tensor(
                last_next_state,
                dtype=torch.float32,
            ).contiguous().to(device)

            _, _, _, last_value_t = self.ac_net.get_action_and_value(last_state_t)
            last_value = last_value_t.view(-1).cpu().numpy()

        rewards = np.array(self._rb_rewards, dtype=np.float32)
        values = np.array(self._rb_values, dtype=np.float32)
        dones = np.array(self._rb_dones, dtype=np.float32)

        T = len(rewards)
        advantages = np.zeros_like(rewards, dtype=np.float32)
        last_gae = np.zeros_like(last_value, dtype=np.float32)

        for t in reversed(range(T)):
            if t == T - 1:
                non_terminal = 1.0 - np.array(last_done, dtype=np.float32)
                next_value = last_value
            else:
                non_terminal = 1.0 - dones[t]
                next_value = values[t + 1]

            delta = rewards[t] + self.discount * next_value * non_terminal - values[t]
            last_gae = (
                delta
                + self.discount
                * self.gae_lambda
                * non_terminal
                * last_gae
            )
            advantages[t] = last_gae

        returns = advantages + values

        b_states = torch.tensor(np.array(self._rb_states), dtype=torch.float32).contiguous().to(device).view(-1, 4, 84, 84)
        b_actions = torch.tensor(np.array(self._rb_actions), dtype=torch.long).to(device).view(-1)
        b_old_log_probs = torch.tensor(np.array(self._rb_log_probs), dtype=torch.float32).to(device).view(-1)
        b_advantages = torch.tensor(advantages, dtype=torch.float32).to(device).view(-1)
        b_returns = torch.tensor(returns, dtype=torch.float32).to(device).view(-1)
        b_old_values = torch.tensor(values, dtype=torch.float32).to(device).view(-1)

        b_advantages = (b_advantages - b_advantages.mean()) / (
            b_advantages.std() + 1e-8
        )

        batch_size = rewards.size
        indices = np.arange(batch_size)

        for _ in range(self.n_epochs):
            np.random.shuffle(indices)
            stop_epoch_early = False

            for start in range(0, batch_size, self.minibatch_size):
                mb_idx = indices[start:start + self.minibatch_size]

                _, new_log_probs, entropy, new_values = self.ac_net.get_action_and_value(
                    b_states[mb_idx],
                    b_actions[mb_idx],
                )

                new_values = new_values.view(-1)

                log_ratio = new_log_probs - b_old_log_probs[mb_idx]
                ratio = log_ratio.exp()

                mb_adv = b_advantages[mb_idx]

                pg_loss_1 = -mb_adv * ratio
                pg_loss_2 = -mb_adv * torch.clamp(
                    ratio,
                    1.0 - self.clip_coef,
                    1.0 + self.clip_coef,
                )
                pg_loss = torch.max(pg_loss_1, pg_loss_2).mean()

                value_clipped = b_old_values[mb_idx] + torch.clamp(
                    new_values - b_old_values[mb_idx],
                    -self.clip_coef,
                    self.clip_coef,
                )

                v_loss_unclipped = (new_values - b_returns[mb_idx]) ** 2
                v_loss_clipped = (value_clipped - b_returns[mb_idx]) ** 2
                v_loss = 0.5 * torch.max(
                    v_loss_unclipped,
                    v_loss_clipped,
                ).mean()

                entropy_loss = entropy.mean()

                loss = (
                    pg_loss
                    + self.vf_coef * v_loss
                    - ent_coef * entropy_loss
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.ac_net.parameters(),
                    self.max_grad_norm,
                )
                self.optimizer.step()

                approx_kl = ((ratio - 1.0) - log_ratio).mean().item()

                if approx_kl > 0.04:
                    stop_epoch_early = True
                    break

            if stop_epoch_early:
                break

        progress = min(self.steps / self.total_timesteps, 1.0)
        current_ent_coef = self.ent_coef_start + progress * (self.ent_coef_end - self.ent_coef_start)

        self.custom_metrics = {
            "pg_loss": pg_loss.item(),
            "v_loss": v_loss.item(),
            "entropy": entropy_loss.item(),
            "approx_kl": approx_kl,
            "entropy_coef": current_ent_coef
        }

    def get_custom_metrics(self):
        return self.custom_metrics

    @classmethod
    def get_scenario(cls):
        return "ppo_scenario"

    @classmethod
    def get_wrappers(cls, verbose=False):
        wrappers = []
        if verbose:
            wrappers.append(DebugObservation)

        wrappers.append(MarioResize)
        wrappers.append(lambda env: MaxAndSkipEnv(env, skip=4))
        wrappers.append(lambda env: FrameStackObservation(env, 4))
        wrappers.append(MarioToPyTorch)

        action_map = [np.array(a, dtype=np.int8) for a in DISCOVERY_ACTIONS]
        wrappers.append(lambda env: DiscreteActionWrapper(env, action_map=action_map))

        wrappers.append(lambda env: EarlyTermination(env, max_no_progress_steps=600, stuck_penalty=-5))
        wrappers.append(lambda env: CompleteLapReward(env, lap_reward=100))

        wrappers.append(lambda env: RewardScaling(env, scale=0.01))

        return wrappers

    def save_checkpoint(self, filepath, episode):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        ckpt_hash = uuid.uuid4().hex[:8]

        checkpoint = {
            "episode": episode,
            "steps": self.steps,
            "ac_net_state": self.ac_net.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "hash": ckpt_hash,
        }

        torch.save(checkpoint, f"{filepath}_model.pth")
        print(f"Checkpoint successfully saved at Episode {episode}.")
        return ckpt_hash

    def load_checkpoint(self, filepath):
        model_path = f"{filepath}_model.pth"

        print(f"Attempting to load checkpoint from {model_path}...")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"No checkpoint found at {model_path}")

        checkpoint = torch.load(model_path, map_location=device)

        self.ac_net.load_state_dict(checkpoint["ac_net_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.steps = checkpoint.get("steps", 0)

        resume_episode = checkpoint["episode"]
        self.ckpt_hash = checkpoint.get("hash", None)

        print(
            f"Successfully resumed from Episode {resume_episode} "
            f"with {self.steps} total steps."
        )

        return resume_episode
