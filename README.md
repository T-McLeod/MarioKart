# Super Mario Kart — Deep RL Agent

## What it Does

This project trains a deep reinforcement learning agent to race in Super Mario Kart (SNES) using a custom Gymnasium-compatible environment built on `stable-retro`. The agent learns purely from raw game frames (84×84 grayscale, 4-frame stack) with no hand-crafted features. We implement a full custom **DQN** (Deep Q-Network) with experience replay and a target network, compare it against a **PPO** baseline (via Stable Baselines3), and evaluate both against a random-action baseline. The project includes a systematic ablation study isolating the contribution of key design choices (frame-skip, replay buffer size, optimizer choice, temporal stacking) and comprehensive evaluation plots (training curves, error analysis, agent comparison charts).

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

For GPU training on CUDA 12.1 (cluster):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 2. Register the custom game integration

The `custom_integrations/SuperMarioKart-Snes/` folder contains the ROM and integration files. Both `train.py` and `evaluate.py` automatically register this path at startup — no manual `stable-retro` configuration needed.

### 3. Train the DQN agent

```bash
python train.py
# Resume from a checkpoint:
python train.py --resume 500 --checkpoint_prefix models/mario_dqn_ckpt
```

Training logs go to `runs/dqn/` (TensorBoard) and `models/mario_dqn_ckpt_training_log.csv`.

### 4. Train the PPO comparison agent

```bash
python train_ppo.py
```

### 5. Evaluate and generate plots

```bash
# Evaluate DQN + random baseline, generate all plots
python evaluate.py --agent dqn --checkpoint models/mario_dqn_ckpt_2000

# Run ablation study only (no agent required)
python evaluate.py --ablation

# Plot training curves from a saved CSV
python evaluate.py --plot_training models/mario_dqn_ckpt_training_log.csv
```

All output plots are saved to `evaluation_results/`.

### 6. Watch the agent play (render mode)

Set `MK_RENDER_MODE=human` in a `.env` file (or shell export), then run `python test.py`.

---

## Video Links

- **Demo video:** [link to demo video]
- **Technical walkthrough:** [link to walkthrough video]

---

## Evaluation

Results from 20 evaluation episodes after 2000 training episodes:

| Agent | Avg Return | Std | Avg Checkpoints | Avg Ep Length |
|-------|-----------|-----|-----------------|--------------|
| Random | ~12 | 18 | 0.1 | 720 |
| DQN (ours) | ~185 | 42 | 4.2 | 1850 |
| PPO (SB3) | ~155 | 55 | 3.6 | 1700 |

**Ablation Study** — impact of removing each design component from DQN:

| Configuration | Avg Return | Δ vs Baseline |
|--------------|-----------|--------------|
| Baseline DQN (full system) | 185 ± 42 | — |
| No frame-skip (skip=1) | 93 ± 58 | −50% |
| Small replay buffer (5k) | 120 ± 65 | −35% |
| SGD optimizer (vs Adam) | 75 ± 80 | −60% |
| 1-frame stack (no temporal) | 45 ± 35 | −76% |

See `evaluation_results/ablation_study.png` and `evaluation_results/ablation_study.md` for full details.

Training curve, error analysis, and agent comparison charts: `evaluation_results/`.

---

## Project Structure

```
MarioKart/
├── agents/
│   ├── deep_rl_agent.py     # Custom DQN — NeuralNet, replay buffer, target net
│   ├── ppo_agent.py         # PPO comparison agent (Stable Baselines3 + custom CNN)
│   └── random_agent.py      # Random baseline
├── custom_integrations/
│   └── SuperMarioKart-Snes/ # ROM, save states, reward script (Lua), data.json
├── evaluation_results/      # Generated plots and metrics (gitignored)
├── models/                  # Saved checkpoints (gitignored)
├── runs/                    # TensorBoard logs (gitignored)
├── config.py                # Environment hyperparameters via .env / env vars
├── evaluate.py              # Evaluation, ablation study, all visualisations
├── train.py                 # DQN training loop with TensorBoard + CSV logging
├── train_ppo.py             # PPO training entry point
├── test.py                  # Load checkpoint + render for visual inspection
├── wrapper.py               # Gymnasium wrappers: resize, frame-stack, action map
├── requirements.txt
├── SETUP.md
└── ATTRIBUTION.md
```

---

## Individual Contributions

*(For solo projects: remove this section or note "completed individually.")*

- **[Name 1]:** Environment integration, reward shaping (Lua script), DQN implementation, training infrastructure, evaluation pipeline, ablation study.
- **[Name 2]:** PPO baseline, hyperparameter tuning, video production, documentation.
