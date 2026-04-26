# Attribution
## External Integrations & Libraries
- - **gym-SuperMarioKart-Snes (Foundational Integration)** This repository provided the essential technical bridge between the SNES emulator and our Python-based agents. It served as the primary foundation for our integration files, most notably the **memory mapping (`data.json`)** required to extract real-time telemetry such as Cartesian coordinates ($x, y, z$), velocity, and checkpoint progress. By utilizing its baseline **reward definitions (`scenario.json`)** and **Lua-based state logic**, we were able to implement a robust training pipeline that transitioned from simple lap-completion goals to our complex, multi-tiered reward hierarchies.  
  *Source: [https://github.com/esteveste/gym-SuperMarioKart-Snes](https://github.com/esteveste/gym-SuperMarioKart-Snes)*
- **Nature DQN Architecture (CNN Design):** Our DQN agent utilizes the seminal CNN architecture described in *"Human-level control through deep reinforcement learning"* (Mnih et al., 2015). We specifically adopted the three-layer convolutional structure (32, 64, and 64 filters) to ensure the agent could effectively generalize spatial features from the game's raw 84x84 pixel input. This established architecture provided the mathematical foundation for the agent's feature extraction and Q-value approximation.  
  *Source: [Mnih et al. (2015)](https://www.nature.com/articles/nature14236)*
- **Stable Retro** — SNES emulation backend wrapping the environment 
  in the Gymnasium API
- **Stable Baselines3** — Imported as a dependency but agents were 
  implemented from scratch rather than using SB3's built-in algorithms
- **PyTorch** — Deep learning framework used for all neural network 
  implementation and training
- **CleanRL `ppo_atari.py` — Huang et al. (2022)** https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_atari.py Our PPO implementation is a substantial adaptation of this file. The following components are directly derived from it: the ActorCritic CNN architecture (modified: FC layer reduced from 512→256), orthogonal weight initialization scheme, GAE advantage computation loop, clipped surrogate policy loss, clipped value loss, advantage normalization, Adam optimizer with eps=1e-5, and learning rate annealing formula. Our original contributions on top of this base: single-environment episodic rollout buffer (vs. vectorized pre-allocated tensors), custom _process_reward() log-transform for reward stability, scheduled entropy coefficient decay, Mario Kart-specific action mapping (SIMPLE_ACTIONS / DISCOVERY_ACTIONS), custom Gymnasium wrapper chain for SNES environment, checkpoint save/load system, and early stopping via approximate KL divergence threshold and no-improvement tolerance.


### Plotting / Matplotlib
- Matplotlib example gallery: https://matplotlib.org/stable/gallery/index.html
- Matplotlib `bar` documentation: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.bar.html
- Matplotlib `fill_between` documentation: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.fill_between.html
- Matplotlib grouped bar chart example: https://matplotlib.org/stable/gallery/lines_bars_and_markers/barchart.html
- Matplotlib `fill_between` / confidence-band example: https://matplotlib.org/stable/gallery/lines_bars_and_markers/fill_between_demo.html




## AI Tool Usage

We used Claude and Gemini during development. Specifically:

- **PPO Hyperparameter:** We used AI tools to help us reason through how to tune the PPO training process when the agent was unstable or plateauing. In particular, AI assistance helped us understand the role of learning rate, rollout length, minibatch size, number of PPO epochs, entropy coefficient, clipping coefficient, and reward scaling/processing. AI tools were useful for explaining why certain settings could make training unstable, suggesting more conservative update schedules, and helping interpret noisy reward curves. However, the final hyperparameter choices were selected through our own experimentation on the Mario Kart environment by comparing training curves, checkpoint progression, and evaluation results across tracks.

- **total_test.py:**  AI tools were used to help build and refine the general evaluation scaffold for this file, including reusable episode-evaluation loops, checkpoint aggregation, CSV/table summaries, and matplotlib plotting helpers (there are inline comments further describing the use and locations). The Mario Kart-specific parts of the file were customized for this project, including the wrapped random baseline, checkpoint-based progress metrics, in-distribution vs out-of-distribution track evaluation, custom failure categories, and the split error analysis by algorithm and track. AI assistance mainly helped with code structure and plotting boilerplate; the actual experimental design, metric choices, and evaluation logic were adapted manually for this project.


- **Overall AI Development**: We used AI tools to brainstorm our multi-tiered reward hierarchy and refine our early-termination logic to improve training throughput. AI assistance was particularly helpful in implementing the "Nature DQN" CNN architecture and creating a custom "crash dump" diagnostic pipeline for forensic error analysis. We also leveraged AI tools to troubleshoot complex environment issues, including WSL kernel setup, PyTorch security protocols, and tensor dimension alignment (e.g., $[1, 4, 84, 84]$). While AI tools provided technical blueprints and debugging strategies, the final code integration, forensic analysis of failure modes like "Replay Buffer Collapse," and overall project synthesis were driven by our own experimentation and evaluation of the agent’s behavior.


