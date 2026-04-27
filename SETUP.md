# Setup Guide

## Prerequisites
* **Docker** (Recommended for consistency), OR **Python 3.10+**
* **NVIDIA GPU** with CUDA 12.1+ (Highly recommended for RL training)
* **Super Mario Kart (USA) ROM**: You must provide your own `Super Mario Kart (USA).sfc` file.

---

## ROM Setup & Integration

The project uses `stable-retro` to interface with the emulator. Simply placing the file in the folder isn't enough; the library needs to "import" it into its internal database.

### 1. ROM Placement
Place your `rom.sfc` (or `Super Mario Kart (USA).sfc`) in the root of the project or a dedicated `/roms` folder.

Note: ROM is not provided in the repo, must find online.

### 2. Import to stable-retro
Run the following command to let the library detect and hash the ROM. This ensures the game is recognized by the Gymnasium environment:

```bash
# This scans the current directory and imports any valid ROMs found
python -m retro.import .
```

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
MK_SCENARIO=speed
MK_DEBUG_OBSERVATION=1
```

---

## Installation & Execution

### 1. Install Dependencies
It is highly recommended to use a virtual environment or Conda (especially on WSL):
```bash
pip install -r requirements.txt
```

### 2. Training the Agent
To start a new training run or resume from the last checkpoint:
```bash
python -m src.train
```

### 3. Evaluation & Visualization
To watch your trained agent drive or run quantitative tests:
```bash
python -m src.test
```
