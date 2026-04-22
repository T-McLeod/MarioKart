import torch
import os
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from gymnasium.wrappers import FrameStackObservation
from wrapper import DebugObservation, DiscreteActionWrapper, MarioResize, MarioToPyTorch, MaxAndSkipEnv, EarlyTermination, CompleteLapReward

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
                 rollout_steps=512, minibatch_size=64, n_epochs=10,
                 clip_coef=0.2, vf_coef=0.5, ent_coef=0.01,
                 gae_lambda=0.95, max_grad_norm=0.5,
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
            ent_coef: Entropy bonus coefficient (encourages exploration)
            gae_lambda: Lambda for Generalized Advantage Estimation
            max_grad_norm: Max gradient norm for clipping
        """
        self.env = env
        self.discount = discount
        self.rollout_steps = rollout_steps
        self.minibatch_size = minibatch_size
        self.n_epochs = n_epochs
        self.clip_coef = clip_coef
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.gae_lambda = gae_lambda
        self.max_grad_norm = max_grad_norm
        self.verbose = verbose
        self.steps = 0

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

                loss = pg_loss + self.vf_coef * v_loss + self.ent_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.ac_net.parameters(), self.max_grad_norm)
                self.optimizer.step()

    def wrap_env(self, env):
        """
        The Agent-Environment Contract:
        Applies all necessary transformations for this specific agent.
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

        # 6. Reward Shaping
        env = EarlyTermination(env)
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
