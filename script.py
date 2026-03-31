import stable_retro
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

def main():
    # 1. Create the environment
    # Note: This will fail until 'Imported 1 games' works
    env = stable_retro.make(game='SuperMarioKart-Snes')
    
    # 2. Wrap it for Stable Baselines3
    env = DummyVecEnv([lambda: env])

    # 3. Define the Model (PPO is great for Mario Kart)
    model = PPO("CnnPolicy", env, verbose=1, tensorboard_log="./logs/")

    print("Model initialized. Ready to train!")
    model.learn(total_timesteps=10000)

if __name__ == "__main__":
    main()