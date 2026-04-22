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
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # 0: No Action
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # 1: Gas
    [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], # 2: Gas + Left
    [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], # 3: Gas + Right
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], # 4: Brake
]


DISCOVERY_ACTIONS = [
    # --- The Basics (4) ---
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 0: Idle / Coasting
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 1: Gas
    [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # 2: Gas + Left
    [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # 3: Gas + Right

    # --- Braking & Correction (3) ---
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 4: Brake / Reverse
    [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # 5: Brake + Left (Sharp correction)
    [0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # 6: Brake + Right

    # --- Advanced Physics: Drifting (3) ---
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # 7: Gas + Hop (Initiate Drift)
    [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],  # 8: Gas + Left + Drift (Tight Corner)
    [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],  # 9: Gas + Right + Drift

    # --- Combat & Item Usage (3) ---
    [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],  # 10: Gas + Item
    [1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0],  # 11: Gas + Left + Item
    [1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0],  # 12: Gas + Right + Item

    # --- The "Weird" Combos (2) ---
    [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 13: Gas + Brake (Burnout/Traction loss)
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # 14: Brake + Hop (Emergency stop turn)
]


class ActorCritic(nn.Module):
    """
    Shared CNN backbone with separate actor (policy) and critic (value) heads.
    The CNN architecture mirrors NeuralNet from deep_rl_agent for a fair comparison.
    """
    def __init__(self, num_actions):
        super(ActorCritic, self).__init__()

        # Shared CNN backbone — same architecture as NeuralNet in deep_rl_agent
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.flatten = nn.Flatten()
        self.fc_shared = nn.Linear(64 * 7 * 7, 256)
        self.relu = nn.ReLU()

        # Actor head: outputs logits over the action space
        self.actor = nn.Linear(256, num_actions)

        # Critic head: outputs a scalar state-value estimate
        self.critic = nn.Linear(256, 1)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.fc_shared(self.flatten(x)))
        return self.actor(x), self.critic(x)

    def get_action_and_value(self, x, action=None):
        """Sample (or evaluate) an action, returning log_prob, entropy, and value."""
        logits, value = self.forward(x)
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value


class PPO_Agent:
    """
    Implements a Proximal Policy Optimization (PPO) agent for Mario Kart.
    Collects on-policy rollouts, computes GAE advantages, and updates with
    the clipped surrogate objective.
    """
    def __init__(self, env, discount=0.99, learning_rate=2.5e-4,
                 rollout_steps=128, minibatch_size=64, n_epochs=10,
                 clip_coef=0.2, vf_coef=0.5, ent_coef_start=0.05, ent_coef_end=0.001,
                 gae_lambda=0.95, max_grad_norm=0.5, total_timesteps=2_000_000,
                 no_improve_tolerance=50,
                 verbose=False):
        """
        Initialize the PPO agent.

        Args:
            discount: Discount factor (gamma)
            learning_rate: Learning rate for Adam optimizer
            rollout_steps: Number of env steps to collect before each update
            minibatch_size: Mini-batch size for PPO update epochs
            n_epochs: Number of gradient update passes per rollout
            clip_coef: PPO clipping epsilon for the surrogate objective
            vf_coef: Value function loss coefficient
            ent_coef_start: Entropy bonus at the start of training (high = more exploration)
            ent_coef_end: Entropy bonus at the end of training (low = more exploitation)
            gae_lambda: Lambda for Generalized Advantage Estimation
            max_grad_norm: Max gradient norm for clipping
            total_timesteps: Expected total env steps — used to schedule entropy and LR decay
            no_improve_tolerance: Stop training if avg return hasn't improved for this
                                  many print_every intervals (checked externally via should_stop)
        """
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

        # Early stopping state
        self.no_improve_tolerance = no_improve_tolerance
        self.best_avg_return = float("-inf")
        self.intervals_without_improvement = 0
        self.should_stop = False

        self.action_space = env.action_space
        self.num_actions = len(SIMPLE_ACTIONS)

        # Actor-Critic network and optimizer
        self.ac_net = ActorCritic(self.num_actions).to(device)
        self.optimizer = torch.optim.Adam(self.ac_net.parameters(), lr=learning_rate, eps=1e-5)

        # On-policy rollout buffer — cleared after each PPO update
        self._init_rollout_buffer()

        # Cached values from the most recent action_select call, used in update()
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
        Call this at every print_every interval with the latest avg return.
        Sets self.should_stop = True if no improvement for no_improve_tolerance intervals.
        """
        if avg_return > self.best_avg_return + 1.0:  # 1.0 = minimum meaningful improvement
            self.best_avg_return = avg_return
            self.intervals_without_improvement = 0
        else:
            self.intervals_without_improvement += 1

        if self.intervals_without_improvement >= self.no_improve_tolerance:
            print(f"Early stopping: no improvement for {self.no_improve_tolerance} intervals. "
                  f"Best avg return: {self.best_avg_return:.2f}")
            self.should_stop = True

    def action_select(self, state):
        """
        Sample an action from the current policy distribution.

        Args:
            state: NumPy array of shape (4, 84, 84)

        Returns:
            action: Integer action index
        """
        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            action_t, log_prob_t, _, value_t = self.ac_net.get_action_and_value(state_t)

        # Cache for retrieval in update()
        self._cached_log_prob = log_prob_t.item()
        self._cached_value = value_t.item()

        return action_t.item()

    def update(self, state, action, reward, next_state, terminated):
        """
        Store one transition and trigger a PPO update when the rollout buffer is full.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            terminated: Whether episode terminated
        """
        self._rb_states.append(state)
        self._rb_actions.append(action)
        self._rb_log_probs.append(self._cached_log_prob)
        self._rb_rewards.append(float(reward))
        self._rb_values.append(self._cached_value)
        self._rb_dones.append(float(terminated))

        self.steps += 1

        if len(self._rb_states) >= self.rollout_steps:
            self._ppo_update(next_state, terminated)
            self._init_rollout_buffer()

    def _ppo_update(self, last_next_state, last_done):
        """Compute GAE advantages over the rollout and run PPO update epochs."""

        # --- Scheduled hyperparameters ---
        # Linear decay from ent_coef_start → ent_coef_end over total_timesteps
        progress = min(self.steps / self.total_timesteps, 1.0)
        ent_coef = self.ent_coef_start + progress * (self.ent_coef_end - self.ent_coef_start)

        # Linear LR decay (standard PPO practice)
        lr = self.initial_lr * (1.0 - progress)
        lr = max(lr, 1e-6)  # floor to avoid going to zero
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

        # Bootstrap the value of the state after the final rollout step
        with torch.no_grad():
            last_state_t = torch.tensor(last_next_state, dtype=torch.float32).unsqueeze(0).to(device)
            _, _, _, last_value_t = self.ac_net.get_action_and_value(last_state_t)
            last_value = last_value_t.item()

        rewards = np.array(self._rb_rewards, dtype=np.float32)
        values  = np.array(self._rb_values,  dtype=np.float32)
        dones   = np.array(self._rb_dones,   dtype=np.float32)
        T = len(rewards)

        # Generalized Advantage Estimation (GAE)
        # non_terminal masks bootstrap when the current step ended the episode
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

        # Move full rollout to tensors on device
        b_states     = torch.tensor(np.array(self._rb_states),  dtype=torch.float32).to(device)
        b_actions    = torch.tensor(np.array(self._rb_actions), dtype=torch.long).to(device)
        b_log_probs  = torch.tensor(np.array(self._rb_log_probs), dtype=torch.float32).to(device)
        b_advantages = torch.tensor(advantages, dtype=torch.float32).to(device)
        b_returns    = torch.tensor(returns,    dtype=torch.float32).to(device)

        # Normalize advantages across the full rollout for training stability
        b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

        indices = np.arange(T)

        for _ in range(self.n_epochs):
            np.random.shuffle(indices)
            for start in range(0, T, self.minibatch_size):
                mb_idx = indices[start : start + self.minibatch_size]

                _, new_log_probs, entropy, new_values = self.ac_net.get_action_and_value(
                    b_states[mb_idx], b_actions[mb_idx]
                )
                new_values = new_values.squeeze()

                # Clipped surrogate policy loss
                log_ratio = new_log_probs - b_log_probs[mb_idx]
                ratio = log_ratio.exp()
                mb_adv = b_advantages[mb_idx]
                pg_loss = torch.max(
                    -mb_adv * ratio,
                    -mb_adv * ratio.clamp(1 - self.clip_coef, 1 + self.clip_coef)
                ).mean()

                # Value function loss
                v_loss = F.mse_loss(new_values, b_returns[mb_idx])

                # Entropy bonus (negative because we maximise entropy)
                entropy_loss = -entropy.mean()

                loss = pg_loss + self.vf_coef * v_loss + ent_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.ac_net.parameters(), self.max_grad_norm)
                self.optimizer.step()

    def wrap_env(self, env):
        """
        The Agent-Environment Contract:
        Applies all necessary transformations for this specific agent.

        PPO-specific reward shaping vs DQN:
        - EarlyTermination uses longer patience (600 vs 300) and a smaller
          stuck penalty (-5 vs -50) to prevent policy collapse during early exploration.
        - SpeedReward adds a small per-step bonus (scale=0.0005) calibrated so
          that one checkpoint crossing (+10) is always worth more than any single
          step of max speed (~1.0), keeping track navigation as the primary signal
          while still rewarding the agent for going fast.
        """
        if self.verbose:
            print("Debug Observation Wrapper Enabled: Original observations will be printed to console.")
            env = DebugObservation(env)

        # 1. Custom Preprocessing (Grayscale + Resize to 84x84)
        env = MarioResize(env)

        # 2. Frame Skipping
        env = MaxAndSkipEnv(env, skip=4)

        # 3. Temporal Stacking (4 frames for velocity)
        env = FrameStackObservation(env, 4)

        # 4. PyTorch Formatting (NCHW + Normalization)
        env = MarioToPyTorch(env)

        # 5. Discrete Action Wrapper
        action_map = [np.array(a, dtype=np.int8) for a in SIMPLE_ACTIONS]
        env = DiscreteActionWrapper(env, action_map=action_map)

        # 6. Reward Shaping — PPO-tuned parameters
        # SpeedReward scale is calibrated against the checkpoint reward:
        # each checkpoint crossing = +10 (from Lua script).
        # kart1_speed is a raw int typically 0-2000, so scale=0.0005 gives
        # max ~1.0/step — meaningful encouragement to stay fast without
        # drowning out the checkpoint signal that teaches track navigation.
        env = EarlyTermination(env, max_no_progress_steps=600, stuck_penalty=-5)
        env = SpeedReward(env, scale=0.0005)
        env = CompleteLapReward(env)

        return env

    def save_checkpoint(self, filepath, episode):
        """Saves the entire training state for a seamless resume."""
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
        """Loads the training state. Returns the episode to resume from."""
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