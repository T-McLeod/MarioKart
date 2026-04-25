import stable_retro
from .agents.random_agent import MarioKartRandomAgent
from .agents.deep_rl_agent import Deep_RL_Agent
from . import config as cfg
import numpy as np
import os
from .wrapper import MarioResize, MarioToPyTorch


GAME_NAME = "SuperMarioKart-Snes"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    checkpoint_prefix = "models/dqn_circuit1_ckpt"
    episode_suffix = "3000"

    env = stable_retro.make(
        game=GAME_NAME,
        state=cfg.state,
        scenario=cfg.scenario if hasattr(cfg, "scenario") else 'scenario',
        render_mode=cfg.render_mode,
        inttype=stable_retro.data.Integrations.ALL
    )
    agent = Deep_RL_Agent(
        env,
        discount=0.99,
        learning_rate=0,
        epsilon_start=0.0,
        epsilon_min=0.1,
    )
    start_episode = agent.load_checkpoint(checkpoint_prefix + f"_{episode_suffix}")
    env = agent.wrap_env(env)

    episode_returns = []
    episode_lengths = []
    # Training Loop
    for episode in range(start_episode, cfg.n_episodes):
        state, info = env.reset()
        episode_over = False
        t = 0
        episode_return = 0
        
        while not episode_over and (cfg.max_timesteps <= 0 or t < cfg.max_timesteps):
            action = agent.action_select(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            episode_return += reward
            # agent.update(state, action, reward, next_state, terminated)

            episode_over = terminated or truncated
            state = next_state
            t += 1

        episode_returns.append(episode_return)
        episode_lengths.append(t)

        if cfg.print_every and (episode + 1) % cfg.print_every == 0:
            avg_return = np.mean(episode_returns[-cfg.print_every:])
            avg_length = np.mean(episode_lengths[-cfg.print_every:])

            print(f"Episode {episode + 1}/{cfg.n_episodes} completed.")
            print(f"    Average Return (last {cfg.print_every} episodes): {avg_return}")
            print(f"    Average Episode Length (last {cfg.print_every} episodes): {avg_length}")
            print(f"    Epsilon: {agent.epsilon:.4f}")

        if episode % 500 == 0 and episode > int(episode_suffix):
            print(f"Saving checkpoint at episode {episode}...")
            agent.save_checkpoint(checkpoint_prefix + f"_{episode}", episode)


if __name__ == "__main__":
    custom_path = os.path.join(SCRIPT_DIR, "custom_integrations")
    stable_retro.data.Integrations.add_custom_path(custom_path)

    main()