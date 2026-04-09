"""
train.py — DQN Training Entry Point for Super Mario Kart

Key design decisions vs. naive DQN:
  - Adam optimizer (not SGD): adaptive learning rates handle sparse rewards better.
    Ablation study (evaluation_results/ablation_study.md) shows Adam converges
    ~2.5× faster than SGD on this task.
  - Experience replay (25k buffer): decorrelates sequential transitions.
  - Target network (sync every 5000 steps): stabilises Q-value targets.
  - Frame-skip=4: reduces temporal correlation, speeds training.
  - 4-frame stack: gives agent velocity/direction information.
  - Checkpoint every 500 episodes for seamless cluster resumption.
  - TensorBoard logging: tracks return and epsilon in real time.
  - Training curves saved to CSV for offline analysis via evaluate.py.

Usage:
    python train.py                          # fresh training run
    python train.py --resume 500             # resume from episode-500 checkpoint
    python train.py --episodes 2000          # override episode count
"""

import argparse
import os
import csv

import stable_retro
import numpy as np

from agents.deep_rl_agent import Deep_RL_Agent
import config as cfg

GAME_NAME = "SuperMarioKart-Snes"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CUSTOM_INTEGRATIONS = os.path.join(SCRIPT_DIR, "custom_integrations")

stable_retro.data.Integrations.add_custom_path(CUSTOM_INTEGRATIONS)


def make_env(render_mode=None):
    return stable_retro.make(
        game=GAME_NAME,
        state=cfg.state,
        scenario=cfg.scenario if hasattr(cfg, "scenario") else "scenario",
        render_mode=render_mode,
        inttype=stable_retro.data.Integrations.ALL,
    )


def main(args):
    checkpoint_prefix = args.checkpoint_prefix
    n_episodes = args.episodes or cfg.n_episodes

    env = make_env(render_mode=cfg.render_mode)

    agent = Deep_RL_Agent(
        env,
        discount=0.99,
        learning_rate=0.00025,   # Adam lr — tuned via validation set
        buffer_size=25000,
        batch_size=64,
        target_update_freq=5000,
        epsilon_start=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.99999,
    )

    start_episode = 0
    if args.resume:
        start_episode = agent.load_checkpoint(f"{checkpoint_prefix}_{args.resume}")

    env = agent.wrap_env(env)

    # TensorBoard (optional — only imported if available)
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=f"runs/dqn/{os.path.basename(checkpoint_prefix)}")
        use_tb = True
        print("TensorBoard logging enabled. Run: tensorboard --logdir runs/dqn/")
    except ImportError:
        writer = None
        use_tb = False

    # CSV log for offline plotting
    os.makedirs(os.path.dirname(checkpoint_prefix), exist_ok=True)
    log_csv = f"{checkpoint_prefix}_training_log.csv"
    csv_file = open(log_csv, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["episode", "return", "length", "epsilon"])

    episode_returns = []
    episode_lengths = []

    print(f"Training DQN from episode {start_episode} to {n_episodes}.")

    for episode in range(start_episode, n_episodes):
        state, info = env.reset()
        episode_over = False
        t = 0
        episode_return = 0.0

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

        if use_tb:
            writer.add_scalar("Train/EpisodeReturn", episode_return, episode)
            writer.add_scalar("Train/EpisodeLength", t, episode)
            writer.add_scalar("Train/Epsilon", agent.epsilon, episode)

        csv_writer.writerow([episode, episode_return, t, agent.epsilon])
        csv_file.flush()

        if cfg.print_every and (episode + 1) % cfg.print_every == 0:
            avg_return = np.mean(episode_returns[-cfg.print_every:])
            avg_length = np.mean(episode_lengths[-cfg.print_every:])
            if use_tb:
                writer.add_scalar("Train/RollingAvgReturn", avg_return, episode)
            print(
                f"Episode {episode+1}/{n_episodes} | "
                f"AvgReturn: {avg_return:.1f} | "
                f"AvgLen: {avg_length:.0f} | "
                f"Epsilon: {agent.epsilon:.4f}"
            )

        if episode > start_episode and episode % 500 == 0:
            print(f"Saving checkpoint at episode {episode}...")
            agent.save_checkpoint(f"{checkpoint_prefix}_{episode}", episode)

    agent.save_checkpoint(f"{checkpoint_prefix}_{n_episodes}", n_episodes)

    # Save training curves
    os.makedirs("evaluation_results", exist_ok=True)
    from evaluate import plot_training_curves_from_list
    plot_training_curves_from_list(
        episode_returns, episode_lengths,
        label="DQN",
        out_path="evaluation_results/training_curves_dqn.png",
    )

    csv_file.close()
    if use_tb:
        writer.close()
    print("Training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None,
                        help="Episode suffix to resume from (e.g. '500')")
    parser.add_argument("--episodes", type=int, default=None,
                        help="Override total episode count from config")
    parser.add_argument("--checkpoint_prefix", type=str,
                        default="models/mario_dqn_ckpt")
    args = parser.parse_args()

    custom_path = os.path.join(SCRIPT_DIR, "custom_integrations")
    stable_retro.data.Integrations.add_custom_path(custom_path)

    main(args)