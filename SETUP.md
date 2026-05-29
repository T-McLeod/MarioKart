# Setup Guide

## Prerequisites
* **Docker** (Recommended for consistency), OR **Python 3.10+**
* **NVIDIA GPU** with CUDA 12.1+ (Highly recommended for RL training)
* **Super Mario Kart (USA) ROM**: You must provide your own `Super Mario Kart (USA).sfc` file.

---

## ROM Setup & Integration

The project uses `stable-retro` to interface with the emulator. Simply placing the file in the folder isn't enough; the library needs to "import" it into its internal database.

### 1. ROM Placement
Place your `Super Mario Kart (USA).sfc` directly into the custom integrations directory:
`custom_integrations/SuperMarioKart-Snes/rom.sfc`

> **Note:** The ROM file must be named `rom.sfc`. This pairs it perfectly with the custom states and scenarios already defined in that folder.

That's it! You no longer need to run any manual `retro.import` scripts. The training script uses `add_custom_path` to instantly and automatically detect your ROM and configurations on boot.

> **Note:** If successful, you should see a message stating: `Importing SuperMarioKart-Snes`.

---

## Configuration

Parameters are controlled via a `.env` file in the `src` directory. Create this `.env` file and change variables where necessary.

**Example `.env`:**
```env
# Mario Kart training config
# Use MK_RENDER_MODE=human to watch gameplay, or none to run headless

MK_STATE=MarioCircuit_M
MK_RENDER_MODE=Human
MK_N_EPISODES=5000
MK_MAX_TIMESTEPS=5000
MK_PRINT_EVERY=1
MK_DEBUG_OBSERVATION=1
```

---

## Installation & Execution

### Option 1: Docker (Recommended)

1. **Build the Docker Image:**
   ```powershell
   docker build -t mariokart-rl .
   ```

2. **Run Training in Docker:**
   Execute the container and mount your local directory so changes and checkpoints are synced. You must set your `WANDB_API_KEY`.
   
   To start a brand new run (auto-generates a human-readable name in W&B):
   ```powershell
   docker run --rm -e PYTHONUNBUFFERED=1 -e WANDB_API_KEY=YOUR_API_KEY -v "${PWD}:/workspace/MarioKart" mariokart-rl python -u -m src.train --agent ppo_nature
   ```

   To name your run (or auto-resume from the highest checkpoint if the name exists):
   ```powershell
   docker run -it --rm \
    -e PYTHONUNBUFFERED=1 \
    -e WANDB_API_KEY=YOUR_WANDB_API_KEY \
    -v "${PWD}:/workspace/MarioKart" \
    mariokart-rl python -u -m src.train --agent ppo_nature --name "my-cool-run"
   ```

   To force-load a specific checkpoint (e.g., update 50):
   ```powershell
   docker run --rm -e PYTHONUNBUFFERED=1 -e WANDB_API_KEY=YOUR_API_KEY -v "${PWD}:/workspace/MarioKart" mariokart-rl python -u -m src.train --agent ppo_nature --name "my-cool-run" --checkpoint 50
   ```

   **Overriding Hyperparameters:**
   You can override the default PPO hyperparameters by appending them as flags (e.g., `--learning-rate 3e-4`, `--rollout-steps 2048`). Use `python -m src.ppo_train --help` to see all available flags.
   > **Important:** Hyperparameters are securely tethered to your W&B cloud config! You **cannot** pass new hyperparameter flags if you are resuming an existing run (the script will throw an error to protect your run consistency).


### Option 2: Local Python Environment

It is highly recommended to use a virtual environment or Conda (especially on WSL):
```bash
pip install -r requirements.txt
```

To start a new training run:
```bash
python -m src.train --agent ppo_nature
```
*(You can append `--name` and `--checkpoint` flags exactly as shown in the Docker instructions).*

### Evaluation & Visualization

To evaluate an agent, calculate its average returns, and optionally record a video of its gameplay, use the `src.eval` script. 

Example for visualizing a specific checkpoint and recording a video:
```bash
python -m src.eval --agent ppo_nature --name "aggressive-learner-v3" --checkpoint 7100 --record
```

**Options:**
- `--agent`: (Required) The agent architecture (e.g., `ppo_nature`).
- `--name`: The W&B run name used during training to locate the correct models folder.
- `--checkpoint`: A specific update number to evaluate.
- `--record`: Include this flag to output an MP4 video of the gameplay to the `videos/` folder.
- `--episodes`: Number of episodes to run (default: 1).
