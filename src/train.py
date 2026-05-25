import stable_retro
from .agents.base import BaseAgent
import importlib
import inspect
from . import config as cfg
import numpy as np
import os
import wandb
import gymnasium as gym

GAME_NAME = "SuperMarioKart-Snes"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    args, provided_hyperparams = cfg.parse_args()

    hyperparams = cfg.PPO_HYPERPARAMS

    wandb_config = {
        "state": cfg.state,
        "n_episodes": cfg.n_episodes,
        "max_timesteps": cfg.max_timesteps,
        **hyperparams
    }

    run_name = args.name if args.name else wandb.util.generate_id()
    wandb.init(
        project="mariokart-rl", 
        id=run_name, 
        name=run_name, 
        resume="allow", 
        config=wandb_config
    )
    print("WANDB INIT DONE - PROCEEDING TO ENV SETUP", flush=True)

    if wandb.run.resumed:
        if provided_hyperparams:
            raise ValueError("Cannot override hyperparameters when resuming! The original hyperparameters must be used.")
        print("Run resumed from W&B! Loading original hyperparameters from cloud config.")
        for key in hyperparams.keys():
            if key in wandb.config:
                hyperparams[key] = wandb.config[key]

    checkpoint_prefix, load_base_path, start_update = cfg.resolve_run_config(run_name, args.checkpoint)

    total_timesteps = hyperparams["total_timesteps"]
    rollout_steps = hyperparams["rollout_steps"]
    num_envs = hyperparams["num_envs"]
    video_freq = hyperparams["video_freq"]

    agent_module_name = f"src.agents.{args.agent}.agent"
    try:
        agent_module = importlib.import_module(agent_module_name)
    except ModuleNotFoundError:
        raise ValueError(f"Could not find agent module: {agent_module_name}. Make sure --agent is correct (e.g., ppo_nature)")

    agent_class = None
    for name, obj in inspect.getmembers(agent_module):
        if inspect.isclass(obj) and issubclass(obj, BaseAgent) and obj is not BaseAgent:
            agent_class = obj
            break
            
    if agent_class is None:
        raise ValueError(f"Could not find a valid BaseAgent subclass in {agent_module_name}")

    # Dummy env for agent init
    agent = agent_class(None, **hyperparams)

    def make_env(idx, record_video=False):
        def _init():
            # Must register custom path inside the worker process because we use multiprocessing 'spawn'
            custom_path = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "custom_integrations"))
            stable_retro.data.Integrations.add_custom_path(custom_path)

            env = stable_retro.make(
                game=GAME_NAME,
                state=cfg.state,
                scenario=cfg.scenario if hasattr(cfg, "scenario") else 'scenario',
                render_mode="rgb_array",
                inttype=stable_retro.data.Integrations.ALL
            )
            for wrapper in agent_class.get_wrappers():
                env = wrapper(env)
            if record_video and idx == 0:
                env = gym.wrappers.RecordVideo(
                    env, 
                    video_folder="videos/", 
                    episode_trigger=lambda ep: ep % video_freq == 0,
                    name_prefix=f"{run_name}"
                )
            return env
        return _init

    envs = gym.vector.AsyncVectorEnv(
        [make_env(i, record_video=True) for i in range(num_envs)],
        context="spawn"
    )

    # Checkpoint loading logic
    if load_base_path is not None:
        start_update = agent.load_checkpoint(load_base_path)
    else:
        start_update = 0

    global_step = agent.steps
    num_updates = (total_timesteps // rollout_steps)
    last_logged_video = None

    state, info = envs.reset()
    episode_returns = np.zeros(num_envs)
    episode_lengths = np.zeros(num_envs)
    
    # Tracking for wandb
    all_episode_returns = []
    all_episode_lengths = []

    print(f"Starting PPO training from Update {start_update} to {num_updates} ({total_timesteps} total steps) with {num_envs} envs...")
    
    assert rollout_steps % num_envs == 0, f"rollout_steps ({rollout_steps}) must be perfectly divisible by num_envs ({num_envs}) to avoid truncation!"
    steps_per_env = rollout_steps // num_envs

    for update in range(start_update + 1, num_updates + 1):
        # Collect exactly `rollout_steps` across all envs for this update
        for step in range(steps_per_env):
            action = agent.action_select(state)
            next_state, reward, terminated, truncated, info = envs.step(action)
            
            episode_returns += reward
            episode_lengths += 1
            global_step += num_envs

            # Combine terminated and truncated for the update step
            done = np.logical_or(terminated, truncated)

            agent.update(state, action, reward, next_state, done)

            for i in range(num_envs):
                if done[i]:
                    all_episode_returns.append(episode_returns[i])
                    all_episode_lengths.append(episode_lengths[i])
                    episode_returns[i] = 0
                    episode_lengths[i] = 0

            state = next_state

        print(f"Update {update}/{num_updates} completed. Total steps: {global_step}/{total_timesteps}")
        
        metrics = {
            "global_step": global_step,
            "update": update
        }

        if len(all_episode_returns) > 0:
            avg_return = np.mean(all_episode_returns[-20:])
            avg_length = np.mean(all_episode_lengths[-20:])
            
            metrics.update({
                "avg_return": avg_return,
                "avg_length": avg_length,
            })
            
        metrics.update(agent.get_custom_metrics())
        
        # We can grab W&B video automatically if monitor_gym was used, or we log manually if file exists
        # RecordVideo creates videos in videos/ folder. We upload the most recent one if it's new.
        videos = [f for f in os.listdir("videos/") if f.endswith(".mp4") and f.startswith(run_name)] if os.path.exists("videos/") else []
        if videos:
            latest_video = sorted(videos, key=lambda x: os.path.getmtime(os.path.join("videos/", x)))[-1]
            if latest_video != last_logged_video:
                video_path = os.path.join("videos/", latest_video)
                # Add to metrics
                metrics["gameplay_video"] = wandb.Video(video_path, format="mp4")
                last_logged_video = latest_video

        wandb.log(metrics)

        if update % hyperparams.get("checkpoint_freq", 50) == 0:
            print(f"Saving checkpoint at update {update}...")
            ckpt_hash = agent.save_checkpoint(f"{checkpoint_prefix}{update}", update)


    print("Training complete. Saving final checkpoint...")
    agent.save_checkpoint(f"{checkpoint_prefix}final", num_updates)
    envs.close()
    wandb.finish()

if __name__ == "__main__":
    custom_path = os.path.abspath("custom_integrations")
    stable_retro.data.Integrations.add_custom_path(custom_path)
    main()
