import stable_retro
from .agents.base import BaseAgent
import importlib
import inspect
from . import config as cfg
import numpy as np
import os
import cv2
from pathlib import Path
import argparse

GAME_NAME = "SuperMarioKart-Snes"
PROJECT_ROOT = Path(__file__).resolve().parent.parent

def main(agent_name, run_name, checkpoint_arg, record=False, num_episodes=1):
    agent_module_name = f"src.agents.{agent_name}.agent"
    try:
        agent_module = importlib.import_module(agent_module_name)
    except ModuleNotFoundError:
        raise ValueError(f"Could not find agent module: {agent_module_name}")

    agent_class = None
    for name, obj in inspect.getmembers(agent_module):
        if inspect.isclass(obj) and issubclass(obj, BaseAgent) and obj is not BaseAgent:
            agent_class = obj
            break
            
    if agent_class is None:
        raise ValueError(f"Could not find a valid BaseAgent subclass in {agent_module_name}")

    if run_name is None:
        run_name = agent_name

    checkpoint_prefix, load_base_path, _ = cfg.resolve_run_config(run_name, checkpoint_arg)

    if load_base_path is None:
        raise FileNotFoundError("Could not find a checkpoint to evaluate.")

    render_mode = "rgb_array" if record else "human"
    
    env = stable_retro.make(
        game=GAME_NAME,
        state=cfg.state,
        scenario=cfg.scenario if hasattr(cfg, "scenario") else 'scenario',
        render_mode=render_mode,
        inttype=stable_retro.data.Integrations.ALL
    )
    
    for wrapper in agent_class.get_wrappers(verbose=not record):
        env = wrapper(env)
        
    # Dummy hyperparams for init
    hyperparams = cfg.PPO_HYPERPARAMS
    agent = agent_class(None, **hyperparams)
    start_update = agent.load_checkpoint(load_base_path)

    episode_returns = []
    episode_lengths = []

    if record:
        output_dir = PROJECT_ROOT / "videos"
        output_dir.mkdir(parents=True, exist_ok=True)
        fps = 60
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        probe_frame = env.render()
        if probe_frame is None:
            probe_frame = np.zeros((240, 256, 3), dtype=np.uint8)
        height, width = probe_frame.shape[:2]
        combined_video_path = output_dir / f"{agent_name}_{cfg.state}_update_{start_update}_eval.mp4"
        video_writer = cv2.VideoWriter(str(combined_video_path), fourcc, fps, (width, height))

    print(f"Starting evaluation on {cfg.state}...")

    for episode_idx in range(num_episodes):
        state, info = env.reset()
        if record:
            frame = env.render()
            if frame is not None:
                video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        episode_over = False
        t = 0
        episode_return = 0

        while not episode_over and (cfg.max_timesteps <= 0 or t < cfg.max_timesteps):
            action = agent.action_select(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            episode_return += reward

            if record:
                frame = env.render()
                if frame is not None:
                    video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

            episode_over = terminated or truncated
            state = next_state
            t += 1

        episode_returns.append(episode_return)
        episode_lengths.append(t)

        print(f"Episode {episode_idx + 1} completed:")
        print(f"    Return: {episode_return}")
        print(f"    Length: {t} steps")

    if record:
        video_writer.release()
        print(f"Video saved to: {combined_video_path}")

    avg_return = np.mean(episode_returns)
    avg_length = np.mean(episode_lengths)
    print(f"--- Average over {num_episodes} test episodes ---")
    print(f"    Avg Return: {avg_return:.2f}")
    print(f"    Avg Length: {avg_length:.2f}")


if __name__ == "__main__":
    custom_path = PROJECT_ROOT / "custom_integrations"
    stable_retro.data.Integrations.add_custom_path(str(custom_path))

    parser = argparse.ArgumentParser(description="Evaluate a Mario Kart agent.")
    parser.add_argument("--agent", type=str, required=True, help="Agent name (e.g. ppo_nature)")
    parser.add_argument("--name", type=str, default=None, help="The name of the run (e.g. aggressive-learner-v3). Used to find the saved models.")
    parser.add_argument("--checkpoint", type=int, default=None, help="Specific update number to resume from")
    parser.add_argument("--record", action="store_true", help="Record episodes to a video file")
    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes to evaluate")
    args = parser.parse_args()

    main(agent_name=args.agent, run_name=args.name, checkpoint_arg=args.checkpoint, record=args.record, num_episodes=args.episodes)
