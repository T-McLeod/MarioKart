# Super Mario Kart — Deep Reinforcement Learning Agent

## What it Does
We trained autonomous agents to play Super Mario Kart on the SNES
using deep reinforcement learning, learning entirely from raw pixel
observations with no hand-crafted features. We implemented two agents
from scratch — Deep Q-Network (DQN) and Proximal Policy Optimization
(PPO) — and compared their performance across two tracks: MarioCircuit_M
and MarioCircuit4_M. Both agents use a custom CNN architecture to process
stacked grayscale frames and output discrete driving actions, trained with
custom reward shaping that incentivizes track progress and lap completion.
Our central research question: **can deep RL agents learn to drive from
raw pixels alone, and how do DQN and PPO compare on this task?**

## Quick Start
See [SETUP.md](SETUP.md) for installation. Once set up:
```bash
# Train the PPO agent
python train.py

# Evaluate all checkpoints + generate all plots and error analysis
python test.py
```

## Video Links
- 🎮 [Demo Video](YOUR_LINK_HERE)
- 🔧 [Technical Walkthrough](YOUR_LINK_HERE)

## Evaluation

### Model Comparison
All agents evaluated over 50 episodes per checkpoint on their
respective tracks. Full raw data in `eval_results/eval_log.csv`.

#### Best Checkpoint Per Agent
| Agent | Avg Return | Std Dev | Avg Checkpoint | Avg Ep Length |
|---|---|---|---|---|
| Random Baseline | -133.6 | 74.5 | 3.7 | 1000.4 |
| DQN Circuit_M best (ep 2750) | 6632.4 | 1647.4 | 142.1 | 1270.5 |
| DQN Circuit4_M best (ep 2750) | 7789.3 | 1082.1 | 199.3 | 2410.1 |
| PPO Circuit_M best (ep 500) | 338.3 | 171.6 | 4.2 | 1320.4 |
| PPO Circuit4_M best (ep 2000) | 236.5 | 138.0 | 2.4 | 1103.4 |

#### DQN MarioCircuit_M — Full Progression
| Checkpoint | Avg Return | Std Dev | Avg Checkpoint | Avg Ep Length |
|---|---|---|---|---|
| ep 250 | -97.2 | 6.6 | 0.1 | 311.2 |
| ep 500 | 118.1 | 353.6 | 10.7 | 744.4 |
| ep 750 | -138.6 | 66.0 | -9.7 | 336.8 |
| ep 1000 | 245.3 | 672.7 | 9.0 | 562.3 |
| ep 1250 | 70.4 | 288.7 | 3.7 | 525.5 |
| ep 1500 | 5552.8 | 2399.9 | 121.0 | 1559.7 |
| ep 1750 | 5861.4 | 2397.8 | 125.6 | 1427.6 |
| ep 2000 | 6538.2 | 1633.1 | 139.1 | 1442.0 |
| ep 2250 | 6516.0 | 1765.6 | 138.6 | 1280.5 |
| ep 2500 | -110.0 | 0.0 | -1.0 | 300.0 |
| ep 2750 | 6632.4 | 1647.4 | 142.1 | 1270.5 |
| ep 3000 | 6255.6 | 1987.2 | 136.0 | 1267.0 |

#### DQN MarioCircuit4_M — Full Progression
| Checkpoint | Avg Return | Std Dev | Avg Checkpoint | Avg Ep Length |
|---|---|---|---|---|
| ep 500 | -109.4 | 2.4 | -0.9 | 300.0 |
| ep 1500 | 187.6 | 89.4 | 11.5 | 825.9 |
| ep 2000 | 5104.0 | 3012.9 | 134.6 | 2126.3 |
| ep 2250 | 5254.9 | 3095.1 | 139.2 | 2144.8 |
| ep 2500 | 6143.8 | 2444.3 | 161.9 | 2204.2 |
| ep 2750 | 7789.3 | 1082.1 | 199.3 | 2410.1 |
| ep 3000 | 1862.7 | 1898.8 | 59.8 | 1110.0 |

#### PPO MarioCircuit_M — Full Progression
| Checkpoint | Avg Return | Std Dev | Avg Checkpoint | Avg Ep Length |
|---|---|---|---|---|
| ep 500 | 338.3 | 171.6 | 4.2 | 1320.4 |
| ep 1000 | 335.7 | 122.2 | 4.4 | 1334.4 |
| ep 1500/early | -116.1 | 30.6 | 0.4 | 734.8 |

#### PPO MarioCircuit4_M — Full Progression
| Checkpoint | Avg Return | Std Dev | Avg Checkpoint | Avg Ep Length |
|---|---|---|---|---|
| ep 500 | 179.0 | 116.2 | 1.4 | 799.7 |
| ep 1000 | 183.6 | 129.0 | 2.2 | 803.2 |
| ep 1500 | 152.8 | 86.4 | -0.3 | 803.3 |
| ep 2000 | 236.5 | 138.0 | 2.4 | 1103.4 |
| ep 2500 | 105.8 | 118.3 | -2.7 | 828.0 |
| final | 96.9 | 184.9 | -7.1 | 824.9 |

![Agent Comparison](eval_results/agent_comparison.png)
![DQN Circuit_M Progression](eval_results/dqn_circuit1_progression.png)
![DQN Circuit4_M Progression](eval_results/dqn_circuit4_progression.png)
![PPO Circuit_M Progression](eval_results/ppo_circuitM_progression.png)
![PPO Circuit4_M Progression](eval_results/ppo_circuit4M_progression.png)
![PPO Training Curve](plots/training_curve.png)

### Key Findings
- **DQN significantly outperformed PPO** on both tracks, achieving a
  best avg return of 6632 (Circuit_M) and 7789 (Circuit4_M) vs PPO's
  best of 338 (Circuit_M) and 236 (Circuit4_M). We believe this might be due in large part to us tailoring our reward function specifically to developing the DQN agent, like the high-reward for lap completion that works well for DQN but seemingly not for PPO. In order to keep our agents comparable we didn't modify it at all for the PPO. DQN's off-policy
  experience replay allowed it to learn from rare high-reward lap
  completion events more efficiently than PPO's on-policy rollouts.
- **Both agents vastly outperformed the random baseline** (-133.6 avg
  return) once training matured past episode 1500 for DQN and episode
  500 for PPO. So despite difficulties with the PPO agent, both architectures were demonstrably able to learn beyond simply random input.
- **DQN showed a sharp performance jump at episode 1500** on
  Circuit_M, going from near-zero returns to 5552 avg return — the
  point at which the agent learned to complete full laps consistently
  (avg checkpoint reached 121 vs 3.7 prior).
- **PPO was more stable but plateaued earlier** — returns stayed in
  the 150–340 range throughout training on both tracks with no
  breakthrough to high-return lap-completion behavior.
- **PPO early-stopped at episode 1500 on Circuit_M** with degraded
  performance (-116 avg return), indicating the policy collapsed
  rather than converged — consistent with the catastrophic forgetting
  pattern we documented in the Model Development section.

### Error Analysis
We categorised evaluation episodes into four failure types:

| Failure Type | Definition | Root Cause |
|---|---|---|
| `never_moved` | Final checkpoint = 0, did not time out | Agent not yet learned gas action produces reward; concentrated in DQN ep 250 (44/50 episodes) and early PPO Circuit4M checkpoints |
| `stuck_early` | Stuck termination fired, checkpoint < 5 | Discrete action space (5 actions) lacks fine-grained steering; agent trapped on tight corners. Dominant failure for PPO (43/50 at early-stop checkpoint) and early DQN |
| `timeout` | Hit `max_timesteps` without terminating | Agent drives forward but corners inefficiently — absent in most trained agents, replaced by success |
| `success` | Meaningful checkpoint progress made | N/A — DQN-Circ1 at ep 1500+ achieves 50/50 success consistently |

Key observations from the failure data:
- DQN Circuit_M from episode 1500 and beyond achieved **50/50
  success** with zero failures, the most consistent performance
  across all agents
- PPO never achieved full success on either track. stuck_early
  remained the dominant failure mode throughout, indicating the agent was never really able to get going and moving
- DQN's OOD performance on Circuit4_M collapsed completely at ep
  3000 (48/50 never_moved) despite strong in-distribution performance. With some investigation we found this agent was still very strong until it went off course, then it was unable to recover. We attribute this to replay buffer collapse, where the replay buffer was filled with almost exclusivley on-road states that in the rare scenario where it went off-road it no longer could recover.

![Error Analysis](eval_results/error_dqn_circuitM.png)
![Error Analysis](eval_results/error_dqn_circuit4M.png)
![Error Analysis](eval_results/error_ppo_circuitM.png)
![Error Analysis](eval_results/error_ppo_circuit4M.png)

### Out-of-Distribution Generalisation
We evaluated Circuit_M-trained models on Circuit4_M (a track neither
agent trained on) to test transfer of learned driving skills.

**DQN generalisation:** Strong in-distribution performance (6632 avg
return at ep 2750) collapsed almost completely on Circuit4_M (-110 avg
return at ep 2750, 50/50 stuck_early). The exception was ep 2500 which
achieved 223.9 OOD avg return despite 0.0 in-distribution return —
an anomalous checkpoint where the policy happened to generalise better.
This indicates DQN learned track-specific Q-values rather than
transferable driving behavior.

**PPO generalisation:** More consistent but weaker. PPO Circuit_M at
ep 500 retained 55% of its in-distribution return on Circuit4_M
(187.7 OOD vs 338.3 in-distribution), suggesting the CNN learned some
generalizable low-level features (edge detection, forward motion) but
not track-agnostic navigation.

![PPO OOD Generalisation](eval_results/ood_generalisation.png)
![DQN OOD Generalisation](eval_results/dqn_ood_generalisation.png)

### Model Development Iterations

Our final PPO agent was the result of two improvement cycles, each
driven by training curve analysis and failure diagnosis.


### **Iteration 1: Transitioning from Collision-Based to Progress-Based Termination**

**The Problem: Training Stagnation & "The Lazy Agent"**
During initial training, we observed a massive spike in episode lengths (averaging 5,000+ frames) that severely throttled training throughput. Upon investigation, we discovered our early stopping mechanism, which terminated the episode after 3 wall collisions, was being exploited. The agent discovered a **local optimum**: by remaining stationary or "coasting" in front of a wall, it could avoid collisions and indefinitely delay the end of an episode. This resulted in a Replay Buffer flooded with thousands of identical, low-value frames of the agent staring at a wall, providing zero informational gain for the model.

**Changes Implemented:**
We shifted from collision-based termination to a **Progress-Based Timeout**. We implemented a "frames-without-progress" counter that resets only when the agent triggers a new track checkpoint (retrieved via RAM). If the agent fails to reach a new checkpoint within a specific window, the episode is terminated with a "stuck penalty."

```python
# Progress-based termination logic in wrapper.py
if checkpoint <= self.max_checkpoint:
    self.frames_without_progress += 1
else:
    self.max_checkpoint = checkpoint
    self.frames_without_progress = 0

# Trigger termination if progress stagnates
if self.frames_without_progress >= self.max_frames_without_progress:
    terminated = True
    reward += self.stuck_penalty
```

**The Result**: 10x Training Throughput
The impact was immediate. Average episode lengths during the early training phase (Episodes 0–1,500) plummeted from over 5,000 frames to approximately 500 frames.

- **Training Speed**: We achieved a 10x increase in iteration speed, allowing us to run more experiments per hour.
- **Sample Efficiency**: By clearing out the "static" data, the Replay Buffer was filled with diverse, high-quality transitions involving actual movement and track navigation.
- **Behavioral Shift**: The agent was forced to "explore or die," leading to much faster convergence on the basic driving policy.

**Iteration 2 — Stabilizing Reward Scale (MarioCircuit_M, Episodes 0–1500)**

During initial training we observed severe sawtooth oscillations —
the agent would occasionally reach returns above 500 on lap completion,
then immediately crash back to negative returns. We diagnosed this as
catastrophic forgetting caused by the raw +1000 lap reward. A single
successful lap produced a gradient shock large enough to overwrite the
policy's basic driving behavior in the next PPO update.

*What we measured:* high variance in avg return, erratic episode length
spikes, no stable performance floor.

*Change implemented:* replaced raw rewards with a symmetric log-transform
applied inside the PPO update:

```python
processed_reward = 5.0 * np.sign(reward) * np.log1p(np.abs(reward))
```

*Result:* the performance floor rose from -100 to a consistent +150.
The agent retained driving knowledge across updates rather than resetting
after each large reward event.

---

### Design Decision: DQN vs PPO
We developed both DQN and PPO to directly compare off-policy vs
on-policy learning on the same task. Contrary to our initial hypothesis
that PPO's stable on-policy updates would handle sparse rewards better,
DQN significantly outperformed PPO — achieving 6632 vs 338 avg return
at best checkpoints on Circuit_M. We attribute this to DQN's experience
replay allowing it to repeatedly train on the rare high-reward lap
completion events, while PPO's on-policy rollouts discarded this
experience after each update. This finding highlights a key tradeoff:
PPO offers training stability and smoother convergence, but DQN's
sample reuse is a decisive advantage when high-value transitions are
rare.

## Individual Contributions
- **Avi Wagner:** Implemented the PPO agent from scratch, including the
  Actor-Critic architecture, GAE advantage estimation, clipped surrogate
  objective, entropy scheduling, and rollout buffer management
  (`agents/ppo_agent.py`). Designed and ran all PPO training runs across
  both MarioCircuit_M and MarioCircuit4_M, including the two
  hyperparameter tuning iterations documented above. Built the full
  evaluation pipeline (`test.py`) including multi-checkpoint evaluation,
  cross-track OOD testing, error analysis categorisation, and all plot
  generation.

- **Tanner McLeod:** Implemented the DQN agent from scratch, including
  experience replay buffer, target network architecture, and
  epsilon-greedy exploration (`agents/deep_rl_agent.py`). Built the
  core project infrastructure, including the custom Gymnasium wrappers
  (`wrapper.py`), The LUA-based reward function integration
  (`custom_integrations/`), the training loop (`train.py`), the
  environment configuration system (`config.py`), and the Dockerfile.
  Designed the reward shaping philosophy — incentivizing speed and
  checkpoint progress without harshly penalizing wall contact or
  off-road driving.

## Project Structure
```
├── agents/
│   ├── deep_rl_agent.py   # DQN with experience replay + target network
│   ├── ppo_agent.py       # PPO with GAE and actor-critic architecture
│   └── random_agent.py    # Random baseline
├── wrapper.py             # Gymnasium wrappers (preprocessing, reward shaping)
├── train.py               # Training loop with live curve plotting
├── test.py                # Checkpoint evaluation, error analysis, OOD testing
├── config.py              # Environment configuration
├── Dockerfile             # Container for reproducible training
└── custom_integrations/   # SNES Mario Kart retro integration
```
