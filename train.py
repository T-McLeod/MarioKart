import stable_retro
from agents.random_agent import MarioKartRandomAgent
from agents.deep_rl_agent import Deep_RL_Agent
from agents.ppo_agent import PPO_Agent
import config as cfg
import numpy as np
import os
import matplotlib
matplotlib.use("Agg")  # headless backend — no display needed on the cluster
import matplotlib.pyplot as plt
from wrapper import MarioResize, MarioToPyTorch


GAME_NAME = "SuperMarioKart-Snes"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def plot_and_save(plot_episodes, avg_returns, avg_lengths, out_dir="plots"):
    """Save a two-panel training curve PNG every time it's called."""
    os.makedirs(out_dir, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    ax1.plot(plot_episodes, avg_returns, color="steelblue", linewidth=1.5)
    ax1.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax1.set_ylabel("Avg Return")
    ax1.set_title("PPO Training Curve — Super Mario Kart")
    ax1.grid(True, alpha=0.3)

    ax2.plot(plot_episodes, avg_lengths, color="darkorange", linewidth=1.5)
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Avg Episode Length")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "training_curve.png"), dpi=120)
    plt.close(fig)


def main():
    checkpoint_prefix = "models/mario_cluster_ckpt"
    episode_suffix = "1500"

    env = stable_retro.make(
        game=GAME_NAME,
        state=cfg.state,
        scenario=cfg.scenario if hasattr(cfg, "scenario") else 'scenario',
        render_mode=cfg.render_mode,
        inttype=stable_retro.data.Integrations.ALL
    )
    """agent = Deep_RL_Agent(
        env,
        discount=0.99,
        learning_rate=0.00025,
        buffer_size=25000,
        batch_size=64,
        target_update_freq=5000,
        epsilon_start=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.99999
    )"""

    agent = PPO_Agent(
        env,
        learning_rate=2.5e-4,
        rollout_steps=1024,
        minibatch_size=256,
        n_epochs=10,
        clip_coef=0.2,
        vf_coef=0.5,
        ent_coef_start=0.05,   # high entropy early → forces exploration
        ent_coef_end=0.001,    # decays toward exploitation as training matures
        gae_lambda=0.95,
        max_grad_norm=0.5,
        total_timesteps=3_000_000,  # rough upper bound for scheduling
        no_improve_tolerance=999999,  # stop if no improvement for 50 print intervals
    )
    start_episode = agent.load_checkpoint(checkpoint_prefix + f"_{episode_suffix}")
    env = agent.wrap_env(env)

    episode_returns = []
    episode_lengths = []
    # Tracking lists for the live training curve (one point per print_every block)
    plot_episodes = []
    plot_avg_returns = []
    plot_avg_lengths = []
    # Training Loop
    print(f"Starting training from Episode {start_episode} to Episode {cfg.n_episodes}...")
    for episode in range(start_episode, cfg.n_episodes):
        state, info = env.reset()
        episode_over = False
        t = 0
        episode_return = 0
        
        while not episode_over and (cfg.max_timesteps <= 0 or t < cfg.max_timesteps):
            action = agent.action_select(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            episode_return += reward
            agent.update(state, action, reward, next_state, terminated)

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
            print(f"    Total steps: {agent.steps}")

            # Record and redraw the training curve
            plot_episodes.append(episode + 1)
            plot_avg_returns.append(avg_return)
            plot_avg_lengths.append(avg_length)
            plot_and_save(plot_episodes, plot_avg_returns, plot_avg_lengths)

            # Early stopping — PPO only
            if isinstance(agent, PPO_Agent):
                agent.record_return(avg_return)
                if agent.should_stop:
                    print("Stopping training early.")
                    agent.save_checkpoint(checkpoint_prefix + f"_early_stop", episode)
                    break

        if episode % 500 == 0 and episode > 0:
            print(f"Saving checkpoint at episode {episode}...")
            agent.save_checkpoint(checkpoint_prefix + f"_{episode}", episode)
    # Save final checkpoint after training completes
    print("Training complete. Saving final checkpoint...")
    agent.save_checkpoint(checkpoint_prefix + "_final", cfg.n_episodes)

if __name__ == "__main__":
    custom_path = os.path.join(SCRIPT_DIR, "custom_integrations")
    stable_retro.data.Integrations.add_custom_path(custom_path)

    main()