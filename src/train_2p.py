"""2-Player Grand Prix PPO training loop (Phase 2).

Trains a single learner (kart 0 / Player 1) against a frozen opponent (kart 1 /
Player 2). Both karts run in one `players=2` race alongside the 6 CPUs; only the
learner is updated. The env stack (see wrapper_2p.py) exposes the learner's
shaped reward through the standard scalar channel, so the rollout is structurally
the same as the single-player loop in train.py -- the only differences are:

  * each env yields a (2, 4, 84, 84) observation; we slice per player,
  * we query the learner and the frozen opponent for their own actions and feed
    the joint MultiDiscrete([N, N]) action to the env,
  * only the learner's transitions are stored / learned from.

The opponent is loaded from a checkpoint (--opponent-checkpoint) or, by default,
initialised as a frozen copy of the learner so they start evenly matched. This
is the structure Phase 3 (PFSP) will reuse by swapping the opponent for a sampled
league agent.
"""
import importlib
import inspect
import os
from pathlib import Path

import cv2
import numpy as np
import wandb
import gymnasium as gym
import stable_retro

from .agents.base import BaseAgent
from . import config as cfg
from .utils import seed_everything
from .wrapper_2p import (
    TwoPlayerMarioEnv,
    ProgressReward2P,
    RelativeProgressReward,
    LearnerStuckTermination,
    RewardScaling2P,
    LearnerScalarReward,
)

GAME_NAME = "SuperMarioKart-Snes"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def load_agent_class(name):
    module_name = f"src.agents.{name}.agent"
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as e:
        raise ValueError(f"Could not find agent module {module_name}: {e}") from e
    for _, obj in inspect.getmembers(module):
        if inspect.isclass(obj) and issubclass(obj, BaseAgent) and obj is not BaseAgent:
            return obj
    raise ValueError(f"No BaseAgent subclass found in {module_name}")


def make_env(args, seed):
    def _init():
        # Register the custom integration inside the worker (spawn context).
        custom_path = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "custom_integrations"))
        stable_retro.data.Integrations.add_custom_path(custom_path)

        base = stable_retro.make(
            game=GAME_NAME,
            state=args.state,
            scenario="scenario",      # lua reward ignored; lua done = race-end/mode-exit
            players=2,
            render_mode="rgb_array",
            inttype=stable_retro.data.Integrations.ALL,
        )
        env = TwoPlayerMarioEnv(base)
        env = ProgressReward2P(env)
        env = RelativeProgressReward(env, coef=args.relative_coef)
        env = LearnerStuckTermination(env, learner_idx=0, max_no_progress_steps=args.stuck_steps)
        env = RewardScaling2P(env, scale=0.01)
        env = LearnerScalarReward(env, learner_idx=0)
        env.action_space.seed(seed)
        return env
    return _init


def evaluate_2p_and_record(learner, opponent, env, video_path, num_episodes=1, max_timesteps=-1):
    """Run learner vs frozen opponent and record the split-screen video."""
    video_path = Path(video_path)
    os.makedirs(video_path.parent, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    probe = env.render()
    h, w = probe.shape[:2]
    writer = cv2.VideoWriter(str(video_path), fourcc, 15, (w, h))

    for _ in range(num_episodes):
        state, _ = env.reset()
        writer.write(cv2.cvtColor(env.render(), cv2.COLOR_RGB2BGR))
        done = False
        t = 0
        while not done and (max_timesteps <= 0 or t < max_timesteps):
            la = learner.action_select(state[0])
            oa = opponent.action_select(state[1])
            state, _, terminated, truncated, _ = env.step([la, oa])
            writer.write(cv2.cvtColor(env.render(), cv2.COLOR_RGB2BGR))
            done = terminated or truncated
            t += 1
    writer.release()


def main():
    args, _ = cfg.parse_args_2p()
    hyperparams = cfg.PPO_HYPERPARAMS

    run_name = args.name if args.name else wandb.util.generate_id()
    wandb.init(
        project="mariokart-2p",
        id=run_name,
        name=run_name,
        resume="allow",
        config={"state": args.state, "opponent_agent": args.opponent_agent or args.agent,
                "relative_coef": args.relative_coef, **hyperparams},
    )
    wandb.define_metric("global_step")
    wandb.define_metric("*", step_metric="global_step")

    seed_everything(hyperparams["seed"])

    total_timesteps = hyperparams["total_timesteps"]
    rollout_steps = hyperparams["rollout_steps"]
    num_envs = hyperparams["num_envs"]
    assert rollout_steps % num_envs == 0, "rollout_steps must be divisible by num_envs"
    steps_per_env = rollout_steps // num_envs

    learner_class = load_agent_class(args.agent)
    opponent_class = load_agent_class(args.opponent_agent or args.agent)
    learner = learner_class(None, **hyperparams)
    opponent = opponent_class(None, **hyperparams)

    # Resume the learner from its own checkpoints (auto or explicit).
    checkpoint_prefix, load_base_path, start_update = cfg.resolve_run_config(run_name, args.checkpoint)
    start_update = learner.load_checkpoint(load_base_path) if load_base_path else 0

    # Freeze the opponent: explicit checkpoint, else a copy of the learner so they
    # start evenly matched. Never updated during training.
    if args.opponent_checkpoint:
        opponent.load_checkpoint(args.opponent_checkpoint)
    else:
        opponent.ac_net.load_state_dict(learner.ac_net.state_dict())
    opponent.ac_net.eval()

    envs = gym.vector.AsyncVectorEnv(
        [make_env(args, hyperparams["seed"] + i) for i in range(num_envs)],
        context="spawn",
    )
    eval_env = make_env(args, hyperparams["seed"] + 9999)()

    global_step = learner.steps
    num_updates = total_timesteps // rollout_steps

    state, info = envs.reset(seed=[hyperparams["seed"] + i for i in range(num_envs)])
    episode_returns = np.zeros(num_envs)
    episode_lengths = np.zeros(num_envs)
    all_returns, all_lengths, all_wins = [], [], []

    print(f"Starting 2P PPO from update {start_update} to {num_updates} "
          f"({total_timesteps} steps, {num_envs} envs) on {args.state}...")

    for update in range(start_update + 1, num_updates + 1):
        for _ in range(steps_per_env):
            learner_obs = state[:, 0]
            opponent_obs = state[:, 1]
            learner_action = learner.action_select(learner_obs)
            opponent_action = opponent.action_select(opponent_obs)
            joint_action = np.stack([learner_action, opponent_action], axis=1)

            next_state, reward, terminated, truncated, info = envs.step(joint_action)
            episode_returns += reward
            episode_lengths += 1
            global_step += num_envs
            done = np.logical_or(terminated, truncated)

            # Only the learner learns.
            learner.update(learner_obs, learner_action, reward, next_state[:, 0], done)

            learner_cp = info.get("learner_cp")
            opp_cp = info.get("opp_cp")
            for i in range(num_envs):
                if done[i]:
                    all_returns.append(episode_returns[i])
                    all_lengths.append(episode_lengths[i])
                    if learner_cp is not None and opp_cp is not None:
                        all_wins.append(1.0 if learner_cp[i] >= opp_cp[i] else 0.0)
                    episode_returns[i] = 0
                    episode_lengths[i] = 0

            state = next_state

        print(f"Update {update}/{num_updates} done. Steps: {global_step}/{total_timesteps}")

        metrics = {"global_step": global_step, "update": update}
        if all_returns:
            metrics["learner/avg_return"] = np.mean(all_returns[-20:])
            metrics["avg_length"] = np.mean(all_lengths[-20:])
        if all_wins:
            metrics["learner/win_rate"] = np.mean(all_wins[-20:])
        metrics.update({f"learner/{k}": v for k, v in learner.get_custom_metrics().items()})

        if update % hyperparams.get("checkpoint_freq", 100) == 0:
            print(f"Saving learner checkpoint at update {update}...")
            learner.save_checkpoint(f"{checkpoint_prefix}{update}", update)

            video_path = os.path.join(
                SCRIPT_DIR, "..", "videos",
                f"{run_name}_{args.state}_update_{update}_2p.mp4",
            )
            evaluate_2p_and_record(learner, opponent, eval_env, video_path,
                                    num_episodes=1, max_timesteps=cfg.max_timesteps)
            if os.path.exists(video_path):
                metrics["gameplay_video"] = wandb.Video(video_path, format="mp4")

        wandb.log(metrics, step=global_step)

    print("Training complete. Saving final learner checkpoint...")
    learner.save_checkpoint(f"{checkpoint_prefix}final", num_updates)
    envs.close()
    wandb.finish()


if __name__ == "__main__":
    custom_path = os.path.abspath("custom_integrations")
    stable_retro.data.Integrations.add_custom_path(custom_path)
    main()
