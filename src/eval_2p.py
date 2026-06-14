"""Evaluate / visualize 2-Player Grand Prix checkpoints.

Loads a trained learner (and a frozen opponent) and runs them head-to-head in the
2P env, reporting return / episode length / win-rate. By default it renders the
split-screen live in a window (run from WSL); with --record it writes an mp4
instead.

    # watch a checkpoint live
    python -m src.eval_2p --agent ppo_nature --name my-2p-run --checkpoint 500

    # record a video, learner vs a specific frozen opponent
    python -m src.eval_2p --agent ppo_nature --name my-2p-run --record \
        --opponent-checkpoint models/phase1-run_7100

`build_2p_env` and `evaluate_2p_and_record` are also imported by src/train_2p.py
(mirroring how train.py reuses src/eval.py).
"""
import argparse
import importlib
import inspect
import os
from pathlib import Path

import cv2
import numpy as np
import stable_retro

from .agents.base import BaseAgent
from . import config as cfg
from .wrapper_2p import (
    TwoPlayerMarioEnv,
    ProgressReward2P,
    RelativeProgressReward,
    LearnerStuckTermination,
    RewardScaling2P,
    LearnerScalarReward,
)

GAME_NAME = "SuperMarioKart-Snes"
PROJECT_ROOT = Path(__file__).resolve().parent.parent


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


def build_2p_env(state, relative_coef=1.0, stuck_steps=600, learner_idx=0, render_mode="rgb_array"):
    """Full 2P env + reward stack used by both training and evaluation."""
    custom_path = os.path.abspath(os.path.join(PROJECT_ROOT, "custom_integrations"))
    stable_retro.data.Integrations.add_custom_path(custom_path)

    base = stable_retro.make(
        game=GAME_NAME,
        state=state,
        scenario="scenario",      # lua reward ignored; lua done = race-end/mode-exit
        players=2,
        render_mode=render_mode,
        inttype=stable_retro.data.Integrations.ALL,
    )
    env = TwoPlayerMarioEnv(base)
    env = ProgressReward2P(env)
    env = RelativeProgressReward(env, coef=relative_coef)
    env = LearnerStuckTermination(env, learner_idx=learner_idx, max_no_progress_steps=stuck_steps)
    env = RewardScaling2P(env, scale=0.01)
    env = LearnerScalarReward(env, learner_idx=learner_idx)
    return env


def evaluate_2p_and_record(learner, opponent, env, video_path=None, num_episodes=1, max_timesteps=-1):
    """Run learner vs frozen opponent. Optionally record the split-screen video.

    Returns (avg_return, avg_length, win_rate). A "win" is the learner reaching a
    global checkpoint >= the opponent's by episode end.
    """
    writer = None
    if video_path:
        video_path = Path(video_path)
        os.makedirs(video_path.parent, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        probe = env.render()
        if probe is None:
            probe = np.zeros((224, 256, 3), dtype=np.uint8)
        h, w = probe.shape[:2]
        writer = cv2.VideoWriter(str(video_path), fourcc, 15, (w, h))
        if not writer.isOpened():
            raise RuntimeError(f"Failed to open video writer for {video_path}")

    returns, lengths, wins = [], [], []

    for ep in range(num_episodes):
        state, info = env.reset()
        if writer is not None:
            frame = env.render()
            if frame is not None:
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        done = False
        t = 0
        ep_return = 0.0
        while not done and (max_timesteps <= 0 or t < max_timesteps):
            learner_action = learner.action_select(state[0])
            opponent_action = opponent.action_select(state[1])
            state, reward, terminated, truncated, info = env.step([learner_action, opponent_action])
            ep_return += reward
            if writer is not None:
                frame = env.render()
                if frame is not None:
                    writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            done = terminated or truncated
            t += 1

        learner_cp = info.get("learner_cp", 0)
        opp_cp = info.get("opp_cp", 0)
        win = 1.0 if learner_cp >= opp_cp else 0.0
        returns.append(ep_return)
        lengths.append(t)
        wins.append(win)
        print(f"Episode {ep + 1}: return={ep_return:.2f} length={t} "
              f"learner_cp={learner_cp} opp_cp={opp_cp} -> {'WIN' if win else 'LOSS'}")

    if writer is not None:
        writer.release()
        print(f"Video saved to: {video_path}")

    return float(np.mean(returns)), float(np.mean(lengths)), float(np.mean(wins))


def main(agent_name, run_name, checkpoint_arg, opponent_agent, opponent_checkpoint,
         state, record, num_episodes):
    learner_class = load_agent_class(agent_name)
    opponent_class = load_agent_class(opponent_agent or agent_name)

    if run_name is None:
        run_name = agent_name

    _, load_base_path, _ = cfg.resolve_run_config(run_name, checkpoint_arg)
    if load_base_path is None:
        raise FileNotFoundError("Could not find a learner checkpoint to evaluate.")

    render_mode = "rgb_array" if record else "human"
    env = build_2p_env(state, render_mode=render_mode)

    hyperparams = cfg.PPO_HYPERPARAMS
    learner = learner_class(None, **hyperparams)
    start_update = learner.load_checkpoint(load_base_path)

    opponent = opponent_class(None, **hyperparams)
    if opponent_checkpoint:
        opponent.load_checkpoint(opponent_checkpoint)
    else:
        opponent.ac_net.load_state_dict(learner.ac_net.state_dict())
    opponent.ac_net.eval()

    video_path = None
    if record:
        out_dir = PROJECT_ROOT / "videos"
        out_dir.mkdir(parents=True, exist_ok=True)
        video_path = out_dir / f"{run_name}_{state}_update_{start_update}_2p_eval.mp4"

    print(f"Evaluating {run_name} (update {start_update}) on {state}...")
    avg_return, avg_length, win_rate = evaluate_2p_and_record(
        learner, opponent, env, video_path=video_path,
        num_episodes=num_episodes, max_timesteps=cfg.max_timesteps,
    )
    print(f"--- Average over {num_episodes} episode(s) ---")
    print(f"    Avg return : {avg_return:.2f}")
    print(f"    Avg length : {avg_length:.1f}")
    print(f"    Win rate   : {win_rate:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a 2-player Mario Kart checkpoint.")
    parser.add_argument("--agent", type=str, required=True, help="Learner agent module (e.g. ppo_nature)")
    parser.add_argument("--name", type=str, default=None, help="Run name used to locate checkpoints")
    parser.add_argument("--checkpoint", type=int, default=None, help="Specific update number to evaluate")
    parser.add_argument("--opponent-agent", type=str, default=None, help="Opponent module (default: same as --agent)")
    parser.add_argument("--opponent-checkpoint", type=str, default=None,
                        help="Path base (without _model.pth) for frozen opponent weights")
    parser.add_argument("--state", type=str, default="MarioCircuit_2P", help="2P savestate name")
    parser.add_argument("--record", action="store_true", help="Record an mp4 instead of rendering live")
    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes to run")
    args = parser.parse_args()

    main(agent_name=args.agent, run_name=args.name, checkpoint_arg=args.checkpoint,
         opponent_agent=args.opponent_agent, opponent_checkpoint=args.opponent_checkpoint,
         state=args.state, record=args.record, num_episodes=args.episodes)
