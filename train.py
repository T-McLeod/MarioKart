import stable_retro
from agents.random_agent import MarioKartRandomAgent
import config as cfg
import numpy as np

def main():
    env = stable_retro.make(game='SuperMarioKart-Snes', render_mode=None)
    agent = MarioKartRandomAgent(env)
    total_episodes_length = 0
    episode_returns = []
    
    # Training Loop
    for episode in range(cfg.n_episodes):
        state, info = env.reset()
        episode_over = False
        t = 0
        episode_return = 0
        
        while not episode_over and (cfg.max_timesteps <= 0 or t < cfg.max_timesteps):
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            episode_return += reward
            agent.update(state, action, reward, next_state, terminated)

            episode_over = terminated or truncated
            state = next_state
            t += 1

        episode_returns.append(episode_return)
        if cfg.print_every and episode % cfg.print_every == 0:
            avg_return = np.mean(episode_returns[-cfg.print_every:])
            print(f"Average Return (last {cfg.print_every} episodes): {avg_return}")
            print(f"Average Episode Length (last {cfg.print_every} episodes): {total_episodes_length / cfg.print_every}")

if __name__ == "__main__":
    main()