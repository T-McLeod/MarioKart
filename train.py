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
  - CSV logging: training curves saved for offline analysis via evaluate.py.
    TensorBoard is used when available but gracefully skipped if broken.

Usage:
    python train.py                          # fresh training run
    python train.py --resume 500             # resume from episode-500 checkpoint
    python train.py --episodes 2000          # override episode count
"""

import argparse
import os
import csv
import sys

import stable_retro
import numpy as np

from agents.deep_rl_agent import Deep_RL_Agent
import config as cfg

GAME_NAME = "SuperMarioKart-Snes"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def make_env(render_mode=None):
    return stable_retro.make(
        game=GAME_NAME,
        state=cfg.state,
        scenario=cfg.scenario if hasattr(cfg, "scenario") else "scenario",
        render_mode=render_mode,
        inttype=stable_retro.data.Integrations.ALL,
    )


def get_tensorboard_writer(log_dir):
    """
    Safely import TensorBoard writer.
    
    The Duke cluster has a system TensorBoard compiled against NumPy 1.x which
    crashes on NumPy 2.x. We isolate the import so a crash here never kills
    training — CSV logging is always available as a fallback.
    """
    try:
        # Block the broken system tensorboard from being found first
        # by temporarily hiding the system site-packages path
        original_path = sys.path[:]
        sys.path = [p for p in sys.path if 'miniconda' not in p and 'site-packages' not in p
                    or '.local' in p]
        from torch.utils.tensorboard import SummaryWriter
        sys.path = original_path
        writer = SummaryWriter(log_dir=log_dir)
        print(f"TensorBoard logging enabled → {log_dir}")
        print("Run: tensorboard --logdir runs/dqn/")
        return writer
    except Exception as e:
        sys.path = original_path if 'original_path' in dir() else sys.path
        print(f"TensorBoard unavailable ({type(e).__name__}: {e})")
        print("Falling back to CSV logging only — training unaffected.")
        return None


def main(args):
    checkpoint_prefix = args.checkpoint_prefix
    n_episodes = args.episodes or cfg.n_episodes

    # Resolve to absolute path so resume works regardless of cwd
    if not os.path.isabs(checkpoint_prefix):
        checkpoint_prefix = os.path.join(SCRIPT_DIR, checkpoint_prefix)

    env = make_env(render_mode=cfg.render_mode)

    agent = Deep_RL_Agent(
        env,
        discount=0.99,
        learning_rate=0.00025,
        buffer_size=25000,
        batch_size=64,
        target_update_freq=5000,
        epsilon_start=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.99999,
    )

    start_episode = 0
    if args.resume:
        # Try the given prefix, then fall back to script-relative path
        ckpt_path = f"{checkpoint_prefix}_{args.resume}"
        model_file = f"{ckpt_path}_model.pth"
        if not os.path.exists(model_file):
            # Try relative to script dir
            alt = os.path.join(SCRIPT_DIR, "models",
                               f"mario_dqn_ckpt_{args.resume}")
            if os.path.exists(f"{alt}_model.pth"):
                ckpt_path = alt
                print(f"Found checkpoint at {alt}_model.pth")
            else:
                print(f"WARNING: No checkpoint found at {model_file}")
                print("Searched also:", f"{alt}_model.pth")
                print("Starting from scratch.")
        start_episode = agent.load_checkpoint(ckpt_path)

    env = agent.wrap_env(env)

    # TensorBoard — safe import that won't crash training if broken
    log_dir = os.path.join(SCRIPT_DIR, "runs", "dqn",
                           os.path.basename(checkpoint_prefix))
    writer = get_tensorboard_writer(log_dir)
    use_tb = writer is not None

    # CSV log — always written, regardless of TensorBoard status
    os.makedirs(os.path.dirname(checkpoint_prefix), exist_ok=True)
    log_csv = f"{checkpoint_prefix}_training_log.csv"
    # Append mode so resume doesn't overwrite existing log
    csv_mode = "a" if args.resume and os.path.exists(log_csv) else "w"
    csv_file = open(log_csv, csv_mode, newline="")
    csv_writer = csv.writer(csv_file)
    if csv_mode == "w":
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
    os.makedirs(os.path.join(SCRIPT_DIR, "evaluation_results"), exist_ok=True)
    try:
        from evaluate import plot_training_curves_from_list
        plot_training_curves_from_list(
            episode_returns, episode_lengths,
            label="DQN",
            out_path=os.path.join(SCRIPT_DIR,
                                  "evaluation_results",
                                  "training_curves_dqn.png"),
        )
    except Exception as e:
        print(f"Could not save training curves: {e}")

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
