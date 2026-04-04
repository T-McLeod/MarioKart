import cv2
import numpy as np
import gymnasium as gym

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
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)
    
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        # print("Kart Speed: ", info.get("kart1_speed", "N/A"))
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