import cv2
import numpy as np
import gymnasium as gym

def get_checkpoint(info):
    local_checkpoint = info.get("current_checkpoint", 0)
    lapsize = info.get("lapsize", 0)
    lap = info.get("lap", 128) - 128

    global_checkpoint = local_checkpoint + (lap) * lapsize
    return global_checkpoint


class MarioResize(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)

    def observation(self, obs):
        # 1. Grayscale
        obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        # 2. Resize to 84x84
        obs = cv2.resize(obs, (84, 84), interpolation=cv2.INTER_AREA)
        return obs[:, :, None] # Add channel dim back for FrameStack


class DebugObservation(gym.Wrapper):
    def __init__(self, env, print_every=60):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)
        self.print_every = print_every
        self.step_count = 0

    
    def step(self, action):
        self.step_count += 1
        obs, reward, terminated, truncated, info = self.env.step(action)
        if self.step_count % self.print_every == 0:
            global_checkpoint = get_checkpoint(info)
            speed = info.get("kart1_speed", 0)

            print(f"Step: {self.step_count}, Checkpoint: {global_checkpoint}, Speed: {speed:.2f}")
        return obs, reward, terminated, truncated, info


class MarioToPyTorch(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        # Final shape: (4, 84, 84) -> 4 frames, 84 high, 84 wide
        old_shape = self.observation_space.shape # (4, 84, 84, 1)
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(old_shape[0], 84, 84), dtype=np.float32
        )

    def observation(self, obs):
        # 1. 'obs' is currently a LazyFrame of shape (4, 84, 84, 1)
        # 2. Convert to numpy and squeeze the extra '1' dimension
        obs = np.array(obs).squeeze(-1) # Results in (4, 84, 84)
        
        # 3. Normalize to [0, 1] for better CNN convergence
        return obs.astype(np.float32) / 255.0
    

class DiscreteActionWrapper(gym.ActionWrapper):

    def __init__(self, env, action_map):
        super().__init__(env)
        self.action_map = action_map
        self.action_space = gym.spaces.Discrete(len(self.action_map))


    def action(self, action):
        return self.action_map[action]
    

class MaxAndSkipEnv(gym.Wrapper):
    """
    Returns only every `skip`-th frame. 
    Repeats the action for the skipped frames and accumulates the reward.
    """
    def __init__(self, env, skip=4):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        terminated = False
        truncated = False
        
        for _ in range(self._skip):
            obs, reward, term, trunc, info = self.env.step(action)
            total_reward += reward
            terminated = term or terminated
            truncated = trunc or truncated
            
            if terminated or truncated:
                break
                
        # Return the final observation of the skipped sequence
        return obs, total_reward, terminated, truncated, info
    
class EarlyTermination(gym.Wrapper):
    """
    Terminates the episode early if the agent is stuck or not making progress.
    """
    def __init__(self, env, max_no_progress_steps=150):
        super().__init__(env)
        self.frames_without_progress = 0
        self.max_frames_without_progress = max_no_progress_steps
        self.max_checkpoint = 0

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        checkpoint = get_checkpoint(info)
        
        # Check for early termination conditions
        if checkpoint <= self.max_checkpoint:
            self.frames_without_progress += 1
        else:
            self.max_checkpoint = checkpoint
            self.frames_without_progress = 0

        if self.frames_without_progress >= self.max_frames_without_progress:
            terminated = True  # End the episode early
            reward -= 50  # Optional: Penalize for being stuck

            self.frames_without_progress = 0  # Reset counter for next episode
            self.max_checkpoint = 0  # Reset checkpoint for next episode

        return obs, reward, terminated, truncated, info
    

class CompleteLapReward(gym.Wrapper):
    """
     Provides a large reward for completing a lap (when checkpoint resets to 0 after reaching the end).
     This encourages the agent to complete laps rather than just progressing through checkpoints.
    """
    def __init__(self, env, lap_reward=1000):
        super().__init__(env)
        self.current_lap = 0
        self.lap_reward = lap_reward

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        checkpoint = get_checkpoint(info)
        
        lap = info.get("lap", 128) - 128
        if lap > self.current_lap:
            reward += self.lap_reward
        
        self.current_lap = max(0, lap)
        

        return obs, reward, terminated, truncated, info