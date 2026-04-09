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


# Retro state name — must match one of the .state files in custom_integrations.
# Available states: MarioCircuit_M, MarioCircuit2_M, MarioCircuit3_M, MarioCircuit4_M,
#   MarioCircuit_M_50cc, MarioCircuit_B, DonutPlains_M, GhostValley_M, etc.
state = _get("MK_STATE", "MarioCircuit_M")

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
