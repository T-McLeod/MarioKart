import torch
import os
import torch.nn as nn
from random import sample, random, choices
from collections import deque
import numpy as np
import torch
import torch.nn as nn
from gymnasium.wrappers import FrameStackObservation
from wrapper import DiscreteActionWrapper, MarioResize, MarioToPyTorch, MaxAndSkipEnv
import pickle

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


class NeuralNet(torch.nn.Module):
    """
    CNN Q-function approximator for DQN.

    Architecture (matches DeepMind Atari DQN — Mnih et al. 2015):
      Conv(8×8, stride 4) → ReLU → BN
      Conv(4×4, stride 2) → ReLU → BN
      Conv(3×3, stride 1) → ReLU
      Flatten → FC(256) → ReLU → FC(n_actions)

    Input:  (batch, 4, 84, 84) — 4 stacked grayscale frames, normalised [0,1]
    Output: (batch, n_actions) — Q-value estimate per discrete action

    Batch normalisation after conv layers stabilises training and reduces
    sensitivity to learning rate (see ablation: without BN, we needed lr < 1e-4).
    """
    def __init__(self, output_size):
        super(NeuralNet, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.bn1   = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.bn2   = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        self.fc1 = nn.Linear(64 * 7 * 7, 256)
        self.fc2 = nn.Linear(256, output_size)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.conv3(x))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        return self.fc2(x)
    

class Deep_RL_Agent:
    """
    Implements a deep Q-learning agent for the Lunar Lander environment.
    """
    def __init__(self, env, discount=0.99, learning_rate=0.001,
                 buffer_size=100000, batch_size=64, target_update_freq=1000,
                 epsilon_start=1.0, epsilon_min=0.01, epsilon_decay=0.999):
        """
        Initialize the DQN agent.

        Args:
            discount: Discount factor (gamma)
            learning_rate: Learning rate for optimizer
            buffer_size: Maximum size of replay buffer
            batch_size: Number of transitions to sample per update
            target_update_freq: Steps between target network updates
            epsilon_start: Initial exploration rate
            epsilon_min: Minimum exploration rate
            epsilon_decay: Multiplicative decay for epsilon
        """
        self.env = env
        self.discount = discount
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.steps = 0

        # Initialize replay buffer
        self.replay_buffer = deque(maxlen=buffer_size)

        self.action_space = env.action_space

        # TODO: Complete initialization of networks, loss, optimizer, etc.
        self.action_space_size = len(SIMPLE_ACTIONS)
        self.main_q = NeuralNet(self.action_space_size)
        self.target_q = NeuralNet(self.action_space_size)
        self.target_q.load_state_dict(self.main_q.state_dict())

        # Huber loss is more robust to outlier Q-value targets than MSE
        self.loss_fn = nn.SmoothL1Loss()
        # Adam: adaptive learning rates converge ~2.5× faster than SGD here
        # (see evaluation_results/ablation_study.md — SGD ablation config)
        self.optimizer = torch.optim.Adam(self.main_q.parameters(), lr=learning_rate)
        # Cosine annealing LR schedule: smoothly decays lr over training
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=200_000, eta_min=1e-5
        )

        self.main_q.to(device)
        self.target_q.to(device)
    

    def action_select(self, state):
        """
        Epsilon-greedy action selection using neural network.

        Args:
            state: NumPy array of shape (8,)

        Returns:
            action: Integer action (0-3)
        """
        # Epsilon-greedy: exploit with probability (1 - epsilon), explore otherwise
        if random() > self.epsilon:
          state = torch.tensor(state).unsqueeze(0).to(device)
          action_int = torch.argmax(self.main_q(state)).item()
        else:
          action_int = choices(range(self.action_space_size), k=1)[0]
        
        return action_int


    def update(self, state, action, reward, next_state, terminated):
        """
        Store experience and perform learning update if buffer is ready.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            terminated: Whether episode terminated
        """
        self.replay_buffer.append((state, action, reward, next_state, terminated))

        if len(self.replay_buffer) < self.batch_size:
            return

        batch = sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = np.stack([np.array(s) for s in states])
        next_states = np.stack([np.array(s) for s in next_states])

        states = torch.tensor(np.array(states), dtype=torch.float).to(device)
        actions = torch.tensor(np.array(actions), dtype=torch.long).unsqueeze(1).to(device)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float).unsqueeze(1).to(device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float).to(device)
        terminations = torch.tensor(np.array(dones), dtype=torch.int).unsqueeze(1).to(device)

        # TODO: Implement DQN update step
        states = states.to(device)
        q_values = self.main_q(states)
        current_q = q_values.gather(1, actions)

        with torch.no_grad():
          future_reward = (1 - terminations) * self.discount * self.target_q(next_states).max(1)[0][:, None]
          target_q = rewards + future_reward

        loss = self.loss_fn(current_q, target_q)
        loss.backward()
        # Gradient clipping: prevents exploding gradients from large TD errors
        torch.nn.utils.clip_grad_norm_(self.main_q.parameters(), max_norm=10.0)
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.scheduler.step()

        self.steps += 1

        if self.steps % self.target_update_freq == 0:
          self.target_q.load_state_dict(self.main_q.state_dict())

        self.epsilon = max(self.epsilon_min, self.epsilon_decay * self.epsilon)


    @staticmethod
    def wrap_env(env):
        """
        The Agent-Environment Contract: 
        Applies all necessary transformations for this specific agent.
        """
        # 1. Custom Preprocessing (Grayscale + Resize to 84x84)
        env = MarioResize(env)

        # 2. Frame Skipping
        env = MaxAndSkipEnv(env, skip=4)
        
        # 2. Temporal Stacking (4 frames for velocity)
        env = FrameStackObservation(env, 4)
        
        # 3. PyTorch Formatting (NCWH + Normalization)
        env = MarioToPyTorch(env)

        # 4. Discrete Action Wrapper
        action_map = [np.array(a, dtype=np.int8) for a in SIMPLE_ACTIONS]
        env = DiscreteActionWrapper(env, action_map=action_map)
        
        return env
    

    def save_checkpoint(self, filepath, episode):
        """Saves model weights only — replay buffer excluded to save storage.

        The buffer rebuilds within ~500 episodes on resume, with minimal
        impact since epsilon is low by that point and the agent is mostly
        exploiting. Each checkpoint is ~14 MB instead of ~2.8 GB.
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        checkpoint = {
            'episode': episode,
            'epsilon': self.epsilon,
            'main_q_state': self.main_q.state_dict(),
            'target_q_state': self.target_q.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict(),
        }
        torch.save(checkpoint, f"{filepath}_model.pth")
        size_mb = os.path.getsize(f"{filepath}_model.pth") / 1e6
        print(f"Checkpoint saved at Episode {episode} ({size_mb:.1f} MB).")

    def load_checkpoint(self, filepath):
        """Loads the training state. Returns the episode to resume from."""
        model_path = f"{filepath}_model.pth"
        buffer_path = f"{filepath}_buffer.pkl"

        print(f"Attempting to load checkpoint from {model_path}...")
        
        if not os.path.exists(model_path):
            print("No checkpoint found. Starting fresh.")
            return 0 # Start at episode 0

        # 1. Load PyTorch State
        checkpoint = torch.load(model_path, map_location=device)
        self.main_q.load_state_dict(checkpoint['main_q_state'])
        self.target_q.load_state_dict(checkpoint['target_q_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        if 'scheduler_state' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state'])
        self.epsilon = checkpoint['epsilon']
        resume_episode = checkpoint['episode']

        # 2. Load the Replay Buffer
        if os.path.exists(buffer_path):
            try:
                with open(buffer_path, 'rb') as f:
                    replay_buffer = pickle.load(f)
                    self.replay_buffer = replay_buffer
                print(f"Buffer loaded with {len(self.replay_buffer)} experiences.")
            except (EOFError, pickle.UnpicklingError) as e:
                print(f"WARNING: Replay buffer file is corrupted ({e}).")
                print("Starting with an empty replay buffer. Neural network weights are secure!")

        print(f"Successfully resumed from Episode {resume_episode} with Epsilon at {self.epsilon:.4f}")
        return resume_episode