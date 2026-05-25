import stable_retro
from .agents.ppo_agent import PPO_Agent
from . import config as cfg
import numpy as np
import os

GAME_NAME = "SuperMarioKart-Snes"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    # Adjust this checkpoint name and suffix to point to the model you want to test
    checkpoint_prefix = "models/ppo_mario_ckpt" + f"_{cfg.state}"
    update_suffix = "50"  # e.g., "final" or an update number like "1500"

    env = stable_retro.make(
        game=GAME_NAME,
        state=cfg.state,
        scenario=cfg.scenario if hasattr(cfg, "scenario") else 'scenario',
        render_mode="human",  # Hardcoded to 'human' so you can actually watch the visualization
        inttype=stable_retro.data.Integrations.ALL
    )

    # Initialize the agent (hyperparameters like learning rate don't matter for testing)
    agent = PPO_Agent(
        env,
        learning_rate=0.0, # Not training
        rollout_steps=2048,
        minibatch_size=256,
        n_epochs=4,
        ent_coef_start=0.0,
        ent_coef_end=0.0,
        clip_coef=0.1,
        max_grad_norm=0.5,
        total_timesteps=3_000_000,
    )
    
    # Load the trained weights
    _, _ = agent.load_checkpoint(checkpoint_prefix + f"_{update_suffix}")
    env = agent.wrap_env(env)

    episode_returns = []
    episode_lengths = []
    
    print(f"Starting testing on {cfg.state}...")

    # Testing Loop
    for episode in range(cfg.n_episodes):
        state, info = env.reset()
        episode_over = False
        t = 0
        episode_return = 0
        
        while not episode_over and (cfg.max_timesteps <= 0 or t < cfg.max_timesteps):
            # Select action deterministically or stochastically (PPO_Agent uses Categorical sample by default)
            action = agent.action_select(state)
            
            next_state, reward, terminated, truncated, info = env.step(action)
            episode_return += reward

            # No agent.update() because we are only testing!

            episode_over = terminated or truncated
            state = next_state
            t += 1

        episode_returns.append(episode_return)
        episode_lengths.append(t)

        print(f"Test Episode {episode + 1} completed:")
        print(f"    Return: {episode_return}")
        print(f"    Length: {t} steps")

        if (episode + 1) % 5 == 0:
            avg_return = np.mean(episode_returns[-5:])
            avg_length = np.mean(episode_lengths[-5:])
            print(f"--- Average over last 5 test episodes ---")
            print(f"    Avg Return: {avg_return:.2f}")
            print(f"    Avg Length: {avg_length:.2f}")


if __name__ == "__main__":
    custom_path = os.path.join(SCRIPT_DIR, "custom_integrations")
    stable_retro.data.Integrations.add_custom_path(custom_path)

    main()
