import os
from pathlib import Path


def _read_dotenv(path: Path) -> dict:
	values = {}
	if not path.exists():
		return values

	for line in path.read_text(encoding="utf-8").splitlines():
		text = line.strip()
		if not text or text.startswith("#") or "=" not in text:
			continue

		key, value = text.split("=", 1)
		values[key.strip()] = value.strip().strip('"').strip("'")

	return values


_dotenv = _read_dotenv(Path(__file__).resolve().parent / ".env")


def _get(key: str, default):
	return os.getenv(key, _dotenv.get(key, default))


def _get_int(key: str, default: int) -> int:
	raw = _get(key, default)
	try:
		return int(raw)
	except (TypeError, ValueError):
		return default


# Retro state name from the imported Super Mario Kart integration.
state = _get("MK_STATE", "Level1")

# None runs headless. Use "human" to render gameplay.
_render_mode = str(_get("MK_RENDER_MODE", "none")).strip().lower()
render_mode = None if _render_mode in {"", "none", "null"} else _render_mode

# Number of episodes to run.
n_episodes = _get_int("MK_N_EPISODES", 100)

# Max steps per episode. Set to 0 to disable this limit.
max_timesteps = _get_int("MK_MAX_TIMESTEPS", 5000)

# Print rolling metrics every N episodes. Set to 0 to disable logs.
print_every = _get_int("MK_PRINT_EVERY", 5)

# Optional scenario name for the integration. Only needed if your integration defines multiple scenarios.
scenario = _get("MK_SCENARIO", "scenario")

import argparse
import glob
import re

PPO_HYPERPARAMS = {
    "learning_rate": 5e-5,
    "rollout_steps": 2048,
    "minibatch_size": 256,
    "n_epochs": 2,
    "ent_coef_start": 0.03,
    "ent_coef_end": 0.01,
    "gae_lambda": 0.95,
    "clip_coef": 0.1,
    "max_grad_norm": 0.5,
    "total_timesteps": 10_000_000,
    "no_improve_tolerance": 500,
    "num_envs": 4,
    "video_freq": 100,
    "checkpoint_freq": 100,
}

def parse_args():
    parser = argparse.ArgumentParser(description="PPO Mario Kart Training")
    parser.add_argument("--name", type=str, default=None, help="Name of the run")
    parser.add_argument("--checkpoint", type=int, default=None, help="Specific update number to resume from")
    parser.add_argument("--agent", type=str, required=True, help="Agent name (e.g. ppo_nature, ppo_impala)")
    
    # Hyperparameters (default None to detect if they were explicitly provided)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--rollout-steps", type=int, default=None)
    parser.add_argument("--minibatch-size", type=int, default=None)
    parser.add_argument("--n-epochs", type=int, default=None)
    parser.add_argument("--ent-coef-start", type=float, default=None)
    parser.add_argument("--ent-coef-end", type=float, default=None)
    parser.add_argument("--gae-lambda", type=float, default=None)
    parser.add_argument("--clip-coef", type=float, default=None)
    parser.add_argument("--max-grad-norm", type=float, default=None)
    parser.add_argument("--total-timesteps", type=int, default=None)
    parser.add_argument("--no-improve-tolerance", type=int, default=None)
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--video-freq", type=int, default=None)
    parser.add_argument("--checkpoint-freq", type=int, default=None)
    
    args = parser.parse_args()
    
    provided_hyperparams = False
    for key, value in vars(args).items():
        if key not in ["name", "checkpoint", "agent"] and value is not None:
            provided_hyperparams = True
            PPO_HYPERPARAMS[key] = value
            
    return args, provided_hyperparams

def resolve_run_config(run_name, checkpoint_arg):
    """
    Scans for checkpoints and returns (checkpoint_prefix, load_base_path, start_update)
    load_base_path is the path WITHOUT the _model.pth suffix, to be compatible with agent.load_checkpoint()
    """
    checkpoint_prefix = f"models/{run_name}_"
    
    if checkpoint_arg is not None:
        load_base_path = f"{checkpoint_prefix}{checkpoint_arg}"
        full_path = f"{load_base_path}_model.pth"
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Requested checkpoint {full_path} does not exist.")
            
        return checkpoint_prefix, load_base_path, checkpoint_arg
        
    # Auto-resume logic
    search_pattern = f"{checkpoint_prefix}*_model.pth"
    existing_ckpts = glob.glob(search_pattern)
    
    if not existing_ckpts:
        return checkpoint_prefix, None, 0
        
    max_update = -1
    best_base_path = None
    
    for ckpt in existing_ckpts:
        match = re.search(r'_(\d+)_model\.pth$', ckpt)
        if match:
            update_num = int(match.group(1))
            if update_num > max_update:
                max_update = update_num
                best_base_path = ckpt.replace("_model.pth", "")
                
    if best_base_path is None:
        return checkpoint_prefix, None, 0
        
    print(f"Auto-resuming from highest checkpoint found: {best_base_path}_model.pth (update {max_update})")
    
    return checkpoint_prefix, best_base_path, max_update
