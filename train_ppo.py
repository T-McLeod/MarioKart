"""
train_ppo.py — PPO Training Entry Point for Super Mario Kart

PPO (Proximal Policy Optimization) is used as our comparison baseline against
the custom DQN implementation. This script trains the PPO agent using the same
environment preprocessing pipeline (frame-skip=4, 4-frame stack, 84x84 grayscale)
and action space as the DQN to ensure a fair comparison.

Key differences from DQN:
  - On-policy: PPO collects a rollout buffer of n_steps transitions, then
    performs multiple gradient updates, then discards the data.
  - Clipped surrogate objective: prevents destructive policy updates.
  - Actor-critic: separate value function head stabilises advantage estimation.
  - No replay buffer or target network needed.

Design tradeoff: PPO is generally more sample-efficient on continuous control
but DQN with experience replay can be more efficient on discrete action spaces
with sparse rewards. We compare both empirically (see evaluation_results/).

Usage:
    python train_ppo.py
    python train_ppo.py --timesteps 1000000 --checkpoint models/ppo_mario
"""

import argparse
import os

import stable_retro
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import config as cfg
from agents.ppo_agent import PPO_Agent

GAME_NAME = "SuperMarioKart-Snes"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def make_env(render_mode=None):
    custom_path = os.path.join(SCRIPT_DIR, "custom_integrations")
    stable_retro.data.Integrations.add_custom_path(custom_path)
    return stable_retro.make(
        game=GAME_NAME,
        state=cfg.state,
        scenario=cfg.scenario if hasattr(cfg, "scenario") else "scenario",
        render_mode=render_mode,
        inttype=stable_retro.data.Integrations.ALL,
    )


def main(args):
    os.makedirs(os.path.dirname(args.checkpoint), exist_ok=True)
    os.makedirs("evaluation_results", exist_ok=True)

    # Build and wrap the environment
    env_base = make_env(render_mode=cfg.render_mode)
    env = PPO_Agent.wrap_env(env_base)

    agent = PPO_Agent(
        env,
        learning_rate=2.5e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,   # entropy bonus encourages exploration
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
    )

    if args.load:
        agent.load(args.load)
        print(f"Resuming PPO from {args.load}")

    print(f"Training PPO for {args.timesteps:,} timesteps...")
    print("TensorBoard: tensorboard --logdir runs/ppo/")

    agent.train(total_timesteps=args.timesteps)

    # Save final model
    agent.save(args.checkpoint)

    # Save training curves
    returns = agent.episode_rewards
    lengths = agent.episode_lengths

    if returns:
        from evaluate import plot_training_curves_from_list
        plot_training_curves_from_list(
            returns, lengths,
            label="PPO",
            out_path="evaluation_results/training_curves_ppo.png",
        )

        # Also save CSV for offline analysis
        log_csv = f"{args.checkpoint}_training_log.csv"
        df = pd.DataFrame({
            "episode": range(len(returns)),
            "return": returns,
            "length": lengths,
        })
        df.to_csv(log_csv, index=False)
        print(f"PPO training log saved → {log_csv}")

    env.close()
    print("PPO training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=500_000,
                        help="Total environment timesteps to train for")
    parser.add_argument("--checkpoint", type=str, default="models/ppo_mario",
                        help="Path to save final model (no extension)")
    parser.add_argument("--load", type=str, default=None,
                        help="Path to an existing PPO checkpoint to resume from")
    args = parser.parse_args()

    custom_path = os.path.join(SCRIPT_DIR, "custom_integrations")
    stable_retro.data.Integrations.add_custom_path(custom_path)

    main(args)
