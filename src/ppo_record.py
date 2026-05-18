import stable_retro
from .agents.ppo_agent import PPO_Agent
from . import config as cfg
import numpy as np
import os
import cv2
from pathlib import Path
import argparse


GAME_NAME = "SuperMarioKart-Snes"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_DIR = PROJECT_ROOT / "extra_files" / "episodes_recorded"
NAME = "PPO_Agent"


def main(num_episodes=1):
    checkpoint_prefix = "models/ppo_mario_ckpt" + f"_{cfg.state}"
    update_suffix = "50"  # e.g., "final" or an update number like "1500"
    num_episodes = num_episodes

    env = stable_retro.make(
        game=GAME_NAME,
        state=cfg.state,
        scenario=cfg.scenario if hasattr(cfg, "scenario") else 'scenario',
        render_mode="rgb_array",  # Use rgb_array for frame capture
        inttype=stable_retro.data.Integrations.ALL
    )
    
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
    
    start_update = agent.load_checkpoint(checkpoint_prefix + f"_{update_suffix}")
    env = agent.wrap_env(env)

    # Create output directory for recording
    output_dir = SCRIPT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fps = 60
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # Initialize one video writer so all episodes are stored in a single file.
    probe_state, _ = env.reset()
    probe_frame = env.render()
    if probe_frame is None:
        print("Warning: render() returned None, using a fallback frame shape")
        probe_frame = np.zeros((240, 256, 3), dtype=np.uint8)
    
    height, width = probe_frame.shape[:2]
    combined_video_path = output_dir / f"{NAME}_{cfg.state}_update_{start_update}_eps_{num_episodes}.mp4"
    video_writer = cv2.VideoWriter(str(combined_video_path), fourcc, fps, (width, height))

    for episode_idx in range(num_episodes):
        episode_number = episode_idx + 1
        state, info = env.reset()

        frame = env.render()
        if frame is None:
            print("Warning: render() returned None, using a fallback frame shape")
            frame = np.zeros((240, 256, 3), dtype=np.uint8)

        # Write the initial frame so playback starts at reset.
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        video_writer.write(frame_bgr)

        print(f"Recording episode {episode_number} ({episode_idx + 1}/{num_episodes})...")
        episode_over = False
        t = 0
        episode_return = 0

        while not episode_over and (cfg.max_timesteps <= 0 or t < cfg.max_timesteps):
            action = agent.action_select(state)
            next_state, reward, terminated, truncated, info = env.step(action)

            # Capture frame
            frame = env.render()
            if frame is not None:
                # Convert RGB to BGR for OpenCV
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                video_writer.write(frame_bgr)

            episode_return += reward
            episode_over = terminated or truncated
            state = next_state
            t += 1

        print(f"Episode {episode_number} completed!")
        print(f"    Episode Return: {episode_return}")
        print(f"    Episode Length: {t} timesteps")

    video_writer.release()
    print(f"Combined video saved to: {combined_video_path}")


if __name__ == "__main__":
    custom_path = PROJECT_ROOT / "custom_integrations"
    stable_retro.data.Integrations.add_custom_path(str(custom_path))

    parser = argparse.ArgumentParser(description="Record PPO Mario Kart agent episodes to video files.")
    parser.add_argument(
        "--episodes",
        type=int,
        default=300,
        help="Number of episodes to record (default: 1).",
    )
    args = parser.parse_args()

    main(num_episodes=args.episodes)
