import torch
import os
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from gymnasium.wrappers import FrameStackObservation
from wrapper import DebugObservation, DiscreteActionWrapper, MarioResize, MarioToPyTorch, MaxAndSkipEnv, EarlyTermination, SpeedReward, CompleteLapReward

# Check for GPU availability (CUDA first, then MPS, then CPU)
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")


SIMPLE_ACTIONS = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 0: No Action
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 1: Gas
    [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # 2: Gas + Left
    [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # 3: Gas + Right
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 4: Brake / Reverse
]

DISCOVERY_ACTIONS = [
    # --- The Basics (4) ---
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 0: Idle / Coasting
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 1: Gas
    [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # 2: Gas + Left
    [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # 3: Gas + Right

    # --- Braking & Correction (3) ---
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 4: Brake / Reverse
    [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # 5: Brake + Left
    [0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # 6: Brake + Right

    # --- Drifting (3) ---
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # 7: Gas + Hop
    [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],  # 8: Gas + Left + Drift
    [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],  # 9: Gas + Right + Drift

    # --- Items (3) ---
    [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],  # 10: Gas + Item
    [1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0],  # 11: Gas + Left + Item
    [1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0],  # 12: Gas + Right + Item

    # --- Weird combos (2) ---
    [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 13: Gas + Brake
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # 14: Brake + Hop
]


class ActorCritic(nn.Module):
    """
    Shared CNN backbone with separate actor (policy) and critic (value) heads.
    """
    def __init__(self, num_actions):
        super(ActorCritic, self).__init__()

        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.flatten = nn.Flatten()
        self.fc_shared = nn.Linear(64 * 7 * 7, 256)
        self.relu = nn.ReLU()

        self.actor = nn.Linear(256, num_actions)
        self.critic = nn.Linear(256, 1)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.fc_shared(self.flatten(x)))
        return self.actor(x), self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits, value = self.forward(x)
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value


class PPO_Agent:
    """
    PPO agent for Mario Kart.
    """
    def __init__(self, env, discount=0.99, learning_rate=2.5e-4,
                 rollout_steps=1024, minibatch_size=256, n_epochs=4,
                 clip_coef=0.2, vf_coef=0.5, ent_coef_start=0.01, ent_coef_end=0.001,
                 gae_lambda=0.95, max_grad_norm=0.5, total_timesteps=2_000_000,
                 no_improve_tolerance=999999,
                 use_discovery_actions=False,
                 verbose=False):

        self.env = env
        self.discount = discount
        self.rollout_steps = rollout_steps
        self.minibatch_size = minibatch_size
        self.n_epochs = n_epochs
        self.clip_coef = clip_coef
        self.vf_coef = vf_coef
        self.ent_coef_start = ent_coef_start
        self.ent_coef_end = ent_coef_end
        self.initial_lr = learning_rate
        self.gae_lambda = gae_lambda
        self.max_grad_norm = max_grad_norm
        self.total_timesteps = total_timesteps
        self.verbose = verbose
        self.steps = 0

        self.no_improve_tolerance = no_improve_tolerance
        self.best_avg_return = float("-inf")
        self.intervals_without_improvement = 0
        self.should_stop = False

        self.action_space = env.action_space
        self.action_set = DISCOVERY_ACTIONS if use_discovery_actions else SIMPLE_ACTIONS
        self.num_actions = len(self.action_set)

        self.ac_net = ActorCritic(self.num_actions).to(device)
        self.optimizer = torch.optim.Adam(self.ac_net.parameters(), lr=learning_rate, eps=1e-5)

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
        """
        Kept for compatibility, but effectively disables early stopping
        unless you set a much smaller tolerance.
        """
        if avg_return > self.best_avg_return + 1.0:
            self.best_avg_return = avg_return
            self.intervals_without_improvement = 0
        else:
            self.intervals_without_improvement += 1

        if self.intervals_without_improvement >= self.no_improve_tolerance:
            print(f"Early stopping: no improvement for {self.no_improve_tolerance} intervals. "
                  f"Best avg return: {self.best_avg_return:.2f}")
            self.should_stop = True

    def action_select(self, state):
        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)

        # Only divide by 255 here if your wrapper does NOT already normalize.
        # state_t = state_t / 255.0

        with torch.no_grad():
            action_t, log_prob_t, _, value_t = self.ac_net.get_action_and_value(state_t)

        self._cached_log_prob = log_prob_t.item()
        self._cached_value = value_t.item()

        return action_t.item()

    def update(self, state, action, reward, next_state, done):
        """
        Store one transition and trigger PPO update when rollout buffer is full.
        'done' should already be (terminated or truncated) from train.py.
        """
        self._rb_states.append(state)
        self._rb_actions.append(action)
        self._rb_log_probs.append(self._cached_log_prob)
        self._rb_rewards.append(float(reward))
        self._rb_values.append(self._cached_value)
        self._rb_dones.append(float(done))

        self.steps += 1

        if len(self._rb_states) >= self.rollout_steps:
            self._ppo_update(next_state, done)
            self._init_rollout_buffer()

    def _ppo_update(self, last_next_state, last_done):
        progress = min(self.steps / self.total_timesteps, 1.0)
        ent_coef = self.ent_coef_start + progress * (self.ent_coef_end - self.ent_coef_start)

        # Keep LR constant for now while debugging stability
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.initial_lr

        with torch.no_grad():
            last_state_t = torch.tensor(last_next_state, dtype=torch.float32).unsqueeze(0).to(device)

            # Only divide by 255 here if your wrapper does NOT already normalize.
            # last_state_t = last_state_t / 255.0

            _, _, _, last_value_t = self.ac_net.get_action_and_value(last_state_t)
            last_value = last_value_t.item()

        rewards = np.array(self._rb_rewards, dtype=np.float32)
        values = np.array(self._rb_values, dtype=np.float32)
        dones = np.array(self._rb_dones, dtype=np.float32)
        T = len(rewards)

        advantages = np.zeros(T, dtype=np.float32)
        last_gae = 0.0

        for t in reversed(range(T)):
            if t == T - 1:
                non_terminal = 1.0 - float(last_done)
                next_value = last_value
            else:
                non_terminal = 1.0 - dones[t]
                next_value = values[t + 1]

            delta = rewards[t] + self.discount * next_value * non_terminal - values[t]
            last_gae = delta + self.discount * self.gae_lambda * non_terminal * last_gae
            advantages[t] = last_gae

        returns = advantages + values

        b_states = torch.tensor(np.array(self._rb_states), dtype=torch.float32).to(device)
        b_actions = torch.tensor(np.array(self._rb_actions), dtype=torch.long).to(device)
        b_log_probs = torch.tensor(np.array(self._rb_log_probs), dtype=torch.float32).to(device)
        b_advantages = torch.tensor(advantages, dtype=torch.float32).to(device)
        b_returns = torch.tensor(returns, dtype=torch.float32).to(device)

        # Only divide by 255 here if your wrapper does NOT already normalize.
        # b_states = b_states / 255.0

        b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

        indices = np.arange(T)

        for _ in range(self.n_epochs):
            np.random.shuffle(indices)

            stop_epoch_early = False

            for start in range(0, T, self.minibatch_size):
                mb_idx = indices[start:start + self.minibatch_size]

                _, new_log_probs, entropy, new_values = self.ac_net.get_action_and_value(
                    b_states[mb_idx], b_actions[mb_idx]
                )
                new_values = new_values.view(-1)

                log_ratio = new_log_probs - b_log_probs[mb_idx]
                ratio = log_ratio.exp()
                mb_adv = b_advantages[mb_idx]

                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * ratio.clamp(1 - self.clip_coef, 1 + self.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                v_loss = F.mse_loss(new_values, b_returns[mb_idx])

                entropy_loss = -entropy.mean()

                loss = pg_loss + self.vf_coef * v_loss + ent_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.ac_net.parameters(), self.max_grad_norm)
                self.optimizer.step()

                approx_kl = ((ratio - 1) - log_ratio).mean().item()
                if approx_kl > 0.03:
                    stop_epoch_early = True
                    break

            if stop_epoch_early:
                break

    def wrap_env(self, env):
        if self.verbose:
            print("Debug Observation Wrapper Enabled: Original observations will be printed to console.")
            env = DebugObservation(env)

        env = MarioResize(env)
        env = MaxAndSkipEnv(env, skip=4)
        env = FrameStackObservation(env, 4)
        env = MarioToPyTorch(env)

        action_map = [np.array(a, dtype=np.int8) for a in self.action_set]
        env = DiscreteActionWrapper(env, action_map=action_map)

        env = EarlyTermination(env, max_no_progress_steps=600, stuck_penalty=-5)

        # Reduced speed shaping for stability
        env = SpeedReward(env, scale=0.0001)

        env = CompleteLapReward(env)

        return env

    def save_checkpoint(self, filepath, episode):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        checkpoint = {
            'episode': episode,
            'steps': self.steps,
            'ac_net_state': self.ac_net.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
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
        self.ac_net.load_state_dict(checkpoint['ac_net_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.steps = checkpoint.get('steps', 0)
        resume_episode = checkpoint['episode']

        print(f"Successfully resumed from Episode {resume_episode} with {self.steps} total steps.")
        return resume_episode