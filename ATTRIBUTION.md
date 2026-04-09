# ATTRIBUTION.md — Sources and AI Tool Attribution

## External Code and Resources

### stable-retro Game Integration
- **Source:** Adapted from [esteveste/gym-SuperMarioKart-Snes](https://github.com/esteveste/gym-SuperMarioKart-Snes)
- **Used:** `custom_integrations/SuperMarioKart-Snes/` — ROM integration files,
  `.state` save files, and the Lua reward/done scripts (`script.lua`, `data.json`,
  `scenario.json`, `metadata.json`)
- **Modifications:** Extended `script.lua` with `getExperimentalReward()` and the
  `isHittingWall()` early-termination logic. Added `speed.json` / `speed_less.json`
  scenario variants for reward-shaping experiments.

### MarIQ Lua Memory Map Reference
- **Source:** SethBling's MarIQ project (SNES memory address annotations in `Notes.txt`)
- **Used:** As a reference for identifying SNES memory addresses (kart position,
  speed, lap counter, checkpoint counter, surface type). No code directly copied.

### DQN Architecture Reference
- **Source:** Mnih et al., "Human-level control through deep reinforcement learning,"
  *Nature* 518, 529–533 (2015).
- **Used:** The 3-layer CNN architecture (8×8/4, 4×4/2, 3×3/1 convolutions with
  ReLU activations) and hyperparameter choices (experience replay buffer, target
  network update frequency, epsilon-greedy schedule).

### PPO Reference
- **Source:** Schulman et al., "Proximal Policy Optimization Algorithms," arXiv:1707.06347
- **Used:** Algorithm reference for the PPO agent in `agents/ppo_agent.py`. The
  implementation itself uses Stable Baselines3.

---

## AI Development Tool Usage

### Claude (Anthropic)
AI assistance was used at the following points in this project. In all cases,
generated code was reviewed, debugged, and adapted before use.

**`agents/ppo_agent.py`** — *Substantially AI-generated, reviewed and modified.*
The `MarioCNN` feature extractor class and `PPO_Agent` wrapper were initially
generated with Claude assistance. I verified the CNN architecture matched the DQN
network dimensions, debugged a shape mismatch in the `features_dim` calculation
(the `n_flatten` computation via a dummy forward pass), and added the
`PPORewardLogger` callback to match the metric-tracking format of the DQN loop.

**`evaluate.py`** — *Scaffold AI-generated, substantially modified.*
The overall structure (argparse CLI, evaluation loop, matplotlib figure layouts)
was generated with Claude. I rewrote the checkpoint-crossing metric (the original
used `info["current_checkpoint"]` naively without tracking deltas), fixed the
error analysis scatter plot colormap to use checkpoints as the color axis (more
informative than raw episode index), and replaced placeholder ablation numbers
with values derived from actual training runs.

**`README.md`, `SETUP.md`** — *AI-generated templates, filled with project-specific content.*
Document structure and section headings were suggested by Claude. All quantitative
results, state file names, hardware-specific pip commands, and troubleshooting
entries were written or verified manually.

**`train.py`** (rewrite) — *Largely written by hand, AI used for TensorBoard boilerplate.*
The TensorBoard `SummaryWriter` integration and CSV logging were adapted from a
Claude-suggested snippet. The core training loop, checkpoint logic, and argparse
interface were written manually.

### What required the most manual work / debugging

1. **Environment compatibility**: `stable-retro` + `gymnasium` version pinning.
   The `FrameStackObservation` API changed between gymnasium 0.26 and 0.29 (the
   `stack_dim` argument was removed). This required reading the gymnasium changelog,
   not AI assistance.

2. **Epsilon-greedy bug**: The original `action_select` condition
   `if random() <= 1 - self.epsilon` is mathematically equivalent to
   `if random() > self.epsilon` but the original code had an off-by-one style
   inversion that caused the agent to exploit too early during warmup.
   Caught by manually tracing the exploration curve in TensorBoard.

3. **Reward scale**: The Lua `getCheckpointReward()` returns multiples of 10
   (not 100 as in the Notes.txt reference). This caused initial confusion about
   why returns were an order of magnitude lower than expected.
