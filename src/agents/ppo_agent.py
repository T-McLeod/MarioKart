import torch
import os
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from gymnasium.wrappers import FrameStackObservation

from ..wrapper import (
    DebugObservation,
    DiscreteActionWrapper,
    MarioResize,
    MarioToPyTorch,
    MaxAndSkipEnv,
    EarlyTermination,
    SpeedReward,
    CompleteLapReward,
)


# device selection — CleanRL ppo_atari.py
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")


# reduced from full SNES button space
SIMPLE_ACTIONS = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # idle
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # gas
    [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # gas + left
    [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # gas + right
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # brake
]



# architecture adapted from CleanRL ppo_atari.py Agent
class ActorCritic(nn.Module):
    def __init__(self, num_actions):
        super().__init__()

        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        self.flatten = nn.Flatten()
        self.fc_shared = nn.Linear(64 * 7 * 7, 256)

        self.actor = nn.Linear(256, num_actions)
        self.critic = nn.Linear(256, 1)

        self._init_weights()

    def _init_weights(self):
        # orthogonal init from CleanRL layer_init()
        # gains: sqrt(2) conv/fc, 0.01 actor, 1.0 critic
        nn.init.orthogonal_(self.conv1.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.conv2.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.conv3.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.fc_shared.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)

        nn.init.constant_(self.conv1.bias, 0)
        nn.init.constant_(self.conv2.bias, 0)
        nn.init.constant_(self.conv3.bias, 0)
        nn.init.constant_(self.fc_shared.bias, 0)
        nn.init.constant_(self.actor.bias, 0)
        nn.init.constant_(self.critic.bias, 0)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.flatten(x)
        x = F.relu(self.fc_shared(x))

        logits = self.actor(x)
        value = self.critic(x)

        return logits, value

    def get_action_and_value(self, x, action=None):
        # CleanRL Agent.get_action_and_value()
        # pixel norm handled in MarioToPyTorch
        logits, value = self.forward(x)
        dist = Categorical(logits=logits)

        if action is None:
            action = dist.sample()

        return action, dist.log_prob(action), dist.entropy(), value


class PPO_Agent:
    def __init__(
        self,
        env,
        discount=0.99,
        learning_rate=5e-5,
        rollout_steps=2048,
        minibatch_size=256,
        n_epochs=4,
        clip_coef=0.1,            # CleanRL atari default
        vf_coef=0.5,              # CleanRL default
        ent_coef_start=0.03,      
        ent_coef_end=0.01,        
        gae_lambda=0.95,          
        max_grad_norm=0.5,        # CleanRL default
        total_timesteps=3_000_000,
        no_improve_tolerance=999999,
        verbose=False,
    ):
        self.env = env
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
        self.total_timesteps = total_timesteps
        self.verbose = verbose

        self.steps = 0
        # early stopping
        self.no_improve_tolerance = no_improve_tolerance
        self.best_avg_return = float("-inf")
        self.intervals_without_improvement = 0
        self.should_stop = False

        self.action_set = SIMPLE_ACTIONS
        self.num_actions = len(self.action_set)

        self.ac_net = ActorCritic(self.num_actions).to(device)
        # eps=1e-5 -- CleanRL 
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
        # early stopping implemented
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
        # this is so it works with same train since episodic 
        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            action_t, log_prob_t, _, value_t = self.ac_net.get_action_and_value(state_t)

        self._cached_log_prob = log_prob_t.item()
        self._cached_value = value_t.item()

        return action_t.item()

    def _process_reward(self, reward):
        """
        PPO-only internal reward processing — original, not in CleanRL.
        CleanRL clips to {-1, 0, 1}; we use log-transform to prevent
        gradient shock from the +1000 lap reward while preserving scale.
        Scale factors tuned across two training iterations.
        """

        # Lap completion or other huge reward
        if abs(reward) >= 500:
            return 5.0 * np.sign(reward) * np.log1p(abs(reward))

        # Stuck / early termination penalty
        if reward <= -4:
            return 3.0 * reward

        # Normal checkpoint / speed / small shaping reward
        return 10.0 * np.sign(reward) * np.log1p(abs(reward))

    def update(self, state, action, reward, next_state, done):
        #buffer here and trigger update when rollout is full
        processed_reward = self._process_reward(reward)

        self._rb_states.append(state)
        self._rb_actions.append(action)
        self._rb_log_probs.append(self._cached_log_prob)
        self._rb_rewards.append(float(processed_reward))
        self._rb_values.append(float(self._cached_value))
        self._rb_dones.append(float(done))

        self.steps += 1

        if len(self._rb_states) >= self.rollout_steps:
            self._ppo_update(next_state, done)
            self._init_rollout_buffer()

    def _ppo_update(self, last_next_state, last_done):
        # core update adapted from CleanRL ppo_atari.py
        progress = min(self.steps / self.total_timesteps, 1.0)

        # entropy decay
        ent_coef = self.ent_coef_start + progress * (
            self.ent_coef_end - self.ent_coef_start
        )

        #lr annealing — CleanRL: frac = 1 - (iter-1)/num_iters
        lr = max(self.learning_rate * (1.0 - progress), 1e-6)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

        #CleanRL idea used
        with torch.no_grad():
            last_state_t = torch.tensor(
                last_next_state,
                dtype=torch.float32,
            ).unsqueeze(0).to(device)

            _, _, _, last_value_t = self.ac_net.get_action_and_value(last_state_t)
            last_value = last_value_t.item()

        rewards = np.array(self._rb_rewards, dtype=np.float32)
        values = np.array(self._rb_values, dtype=np.float32)
        dones = np.array(self._rb_dones, dtype=np.float32)

        T = len(rewards)
        advantages = np.zeros(T, dtype=np.float32)
        last_gae = 0.0

        #GAE — CleanRL
        for t in reversed(range(T)):
            if t == T - 1:
                non_terminal = 1.0 - float(last_done)
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

        returns = advantages + values  #CleanRL

        # flatten buffer to tensors — CleanRL "flatten the batch"
        b_states = torch.tensor(np.array(self._rb_states), dtype=torch.float32).to(device)
        b_actions = torch.tensor(np.array(self._rb_actions), dtype=torch.long).to(device)
        b_old_log_probs = torch.tensor(
            np.array(self._rb_log_probs),
            dtype=torch.float32,
        ).to(device)
        b_advantages = torch.tensor(advantages, dtype=torch.float32).to(device)
        b_returns = torch.tensor(returns, dtype=torch.float32).to(device)
        b_old_values = torch.tensor(values, dtype=torch.float32).to(device)

        # advantage normalization — CleanRL norm_adv=True
        b_advantages = (b_advantages - b_advantages.mean()) / (
            b_advantages.std() + 1e-8
        )

        indices = np.arange(T)

        for _ in range(self.n_epochs):
            np.random.shuffle(indices)
            stop_epoch_early = False

            for start in range(0, T, self.minibatch_size):
                mb_idx = indices[start:start + self.minibatch_size]

                _, new_log_probs, entropy, new_values = self.ac_net.get_action_and_value(
                    b_states[mb_idx],
                    b_actions[mb_idx],
                )

                new_values = new_values.view(-1)

                #KL
                log_ratio = new_log_probs - b_old_log_probs[mb_idx]
                ratio = log_ratio.exp()

                mb_adv = b_advantages[mb_idx]

                #CleanRL
                pg_loss_1 = -mb_adv * ratio
                pg_loss_2 = -mb_adv * torch.clamp(
                    ratio,
                    1.0 - self.clip_coef,
                    1.0 + self.clip_coef,
                )
                pg_loss = torch.max(pg_loss_1, pg_loss_2).mean()

                # clipped value loss from CleanRL clip_vloss=True
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

                # combined loss — CleanRL
                loss = (
                    pg_loss
                    + self.vf_coef * v_loss
                    - ent_coef * entropy_loss
                )

                self.optimizer.zero_grad()
                loss.backward()
                # grad clipping — CleanRL
                nn.utils.clip_grad_norm_(
                    self.ac_net.parameters(),
                    self.max_grad_norm,
                )
                self.optimizer.step()

                #aprrox kl
                approx_kl = ((ratio - 1.0) - log_ratio).mean().item()

                if approx_kl > 0.04:
                    stop_epoch_early = True
                    break

            if stop_epoch_early:
                break

    def wrap_env(self, env):
        # mirrors CleanRL's atari wrapper stack for Mario Kart
        # MaxAndSkipEnv, FrameStack -- CleanRL
        if self.verbose:
            env = DebugObservation(env)

        env = MarioResize(env)
        env = MaxAndSkipEnv(env, skip=4)
        env = FrameStackObservation(env, 4)
        env = MarioToPyTorch(env)

        action_map = [np.array(a, dtype=np.int8) for a in self.action_set]
        env = DiscreteActionWrapper(env, action_map=action_map)

        # Same wrapper rewards as DQN for fair environment comparison.
        env = EarlyTermination(env, max_no_progress_steps=600, stuck_penalty=-5)
        env = SpeedReward(env, scale=0.0001)
        env = CompleteLapReward(env)

        return env

    def save_checkpoint(self, filepath, episode):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        checkpoint = {
            "episode": episode,
            "steps": self.steps,
            "ac_net_state": self.ac_net.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
        }

        torch.save(checkpoint, f"{filepath}_model.pth")
        print(f"Checkpoint successfully saved at Episode {episode}.")

    def load_checkpoint(self, filepath):
        model_path = f"{filepath}_model.pth"

        print(f"Attempting to load checkpoint from {model_path}...")

        if not os.path.exists(model_path):
            print("No checkpoint found. Starting fresh.")
            return 0

        checkpoint = torch.load(model_path, map_location=device)

        self.ac_net.load_state_dict(checkpoint["ac_net_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.steps = checkpoint.get("steps", 0)

        resume_episode = checkpoint["episode"]

        print(
            f"Successfully resumed from Episode {resume_episode} "
            f"with {self.steps} total steps."
        )

        return resume_episode