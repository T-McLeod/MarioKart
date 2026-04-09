# SETUP.md — Installation & Environment Guide

## System Requirements

- Python 3.9–3.11 (3.12 has known issues with stable-retro)
- 8 GB RAM minimum; 16 GB recommended for training
- NVIDIA GPU with CUDA 12.1 recommended for full training runs
  (CPU/Apple MPS also supported for testing/small runs)

---

## Step-by-Step Installation

### 1. Clone the repository

```bash
git clone https://github.com/T-McLeod/MarioKart.git
cd MarioKart
```

### 2. Obtain the ROM

The Super Mario Kart (USA) ROM is **not included** in this repository for copyright reasons. You need to supply it yourself.

**Source used in development:**  
`Super_Mario_Kart_(USA).sfc` — available at [archive.org/details/super-mario-kart-usa_202303](https://archive.org/details/super-mario-kart-usa_202303)

Once you have the file, place it at:
```
custom_integrations/SuperMarioKart-Snes/rom.sfc
```

**Verify the SHA-1 hash matches exactly** (stable-retro will reject a wrong ROM):
```bash
sha1sum custom_integrations/SuperMarioKart-Snes/rom.sfc
# Expected: 47e103d8398cf5b7cbb42b95df3a3c270691163b
cat custom_integrations/SuperMarioKart-Snes/rom.sha   # same hash stored here
```

If the hashes don't match, the environment will raise a `RomNotFoundError` or silently produce garbage observations. Only the exact USA (NTSC) version of the ROM is compatible with the memory addresses in `data.json` and `script.lua`.

### 3. Create a virtual environment (recommended)


```bash
python3 -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate          # Windows
```

### 4. Install PyTorch

**CPU / Apple MPS (local development):**
```bash
pip install torch torchvision
```

**NVIDIA GPU — CUDA 12.1 (cluster / workstation):**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Verify your installation:
```bash
python3 -c "import torch; print('CUDA:', torch.cuda.is_available(), '| MPS:', torch.backends.mps.is_available())"
```

### 5. Install all other dependencies

```bash
pip install -r requirements.txt
```

### 6. Verify stable-retro and the custom integration

The Super Mario Kart ROM and integration files are already bundled in
`custom_integrations/SuperMarioKart-Snes/`. Both `train.py` and `evaluate.py`
register this path automatically. To verify manually:

```bash
python3 -c "
import stable_retro, os
stable_retro.data.Integrations.add_custom_path(
    os.path.join(os.getcwd(), 'custom_integrations')
)
env = stable_retro.make('SuperMarioKart-Snes', state='MarioCircuit_M',
                         inttype=stable_retro.data.Integrations.ALL)
obs, _ = env.reset()
print('Environment OK — obs shape:', obs.shape)
env.close()
"
```

Expected output: `Environment OK — obs shape: (224, 256, 3)`

### 7. (Optional) Configure via `.env` file

Create a `.env` file in the project root to override defaults:

```
MK_STATE=MarioCircuit_M         # track to train on
MK_RENDER_MODE=none             # 'human' to watch training
MK_N_EPISODES=2000
MK_MAX_TIMESTEPS=5000
MK_PRINT_EVERY=10
```

Available states (from `custom_integrations/SuperMarioKart-Snes/`):

| State name | Track |
|------------|-------|
| `MarioCircuit_M` | Mario Circuit 1 — Mushroom Cup (default) |
| `MarioCircuit_M_50cc` | Mario Circuit 1 — 50cc |
| `MarioCircuit_B` | Mario Circuit 1 — Battle mode |
| `MarioCircuit2_M` | Mario Circuit 2 |
| `MarioCircuit3_M` | Mario Circuit 3 |
| `MarioCircuit4_M` | Mario Circuit 4 |
| `DonutPlains_M` | Donut Plains 1 |
| `DonutPlains2_M` | Donut Plains 2 |
| `DonutPlains3_M` | Donut Plains 3 |
| `GhostValley_M` | Ghost Valley 1 |
| `GhostValley2_M` | Ghost Valley 2 |
| `GhostValley3_M` | Ghost Valley 3 |
| `KoopaBeach_M` | Koopa Beach 1 |
| `KoopaBeach2_M` | Koopa Beach 2 |
| `ChocoIsland_M` | Choco Island 1 |
| `ChocoIsland2_M` | Choco Island 2 |
| `ChocoIsland` | Choco Island (alt save) |
| `VanillaLake_M` | Vanilla Lake 1 |
| `VanillaLake2_M` | Vanilla Lake 2 |
| `BowserCastle_M` | Bowser Castle 1 |
| `BowserCastle2_M` | Bowser Castle 2 |
| `BowserCastle3_M` | Bowser Castle 3 |
| `RainbowRoad_M` | Rainbow Road — Mario |
| `RainbowRoad_DK` | Rainbow Road — Donkey Kong Jr. |

---

## Running the Project

### Train the DQN agent
```bash
python train.py
# With options:
python train.py --episodes 2000 --checkpoint_prefix models/mario_dqn_ckpt
# Resume from checkpoint:
python train.py --resume 500
```

### Train the PPO comparison agent
```bash
python train_ppo.py
python train_ppo.py --timesteps 1000000
```

### Evaluate and generate all plots
```bash
python evaluate.py --agent dqn --checkpoint models/mario_dqn_ckpt_2000
python evaluate.py --ablation    # ablation study only (no checkpoint needed)
```

### Monitor training with TensorBoard
```bash
tensorboard --logdir runs/
# Open http://localhost:6006 in your browser
```

### Watch the agent play
```bash
# In .env set MK_RENDER_MODE=human, then:
python test.py
```

---

## Common Issues

**`ImportError: No module named 'cv2'`**
```bash
pip install opencv-python
```

**`stable_retro.data.GameNotFoundError`**
Make sure you're running from the project root directory so the
`custom_integrations/` path resolves correctly.

**`RuntimeError: CUDA out of memory`**
Reduce `batch_size` in `train.py` (try 32) or set `MK_MAX_TIMESTEPS=2000`.

**`torch.load` deprecation warning on PyTorch ≥ 2.6**
Harmless — the checkpoint loader in `deep_rl_agent.py` will be updated
to use `weights_only=True` when stable-retro supports it.
