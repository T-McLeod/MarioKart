import stable_retro
from .agents.random_agent import MarioKartRandomAgent
from .agents.deep_rl_agent import Deep_RL_Agent
from .agents.ppo_agent import PPO_Agent
from . import config as cfg
import numpy as np
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # standard matplotlib setting for running without a display
import matplotlib.pyplot as plt
from gymnasium.wrappers import FrameStackObservation
from .wrapper import (
    get_checkpoint,
    MarioResize,
    MarioToPyTorch,
    MaxAndSkipEnv,
    DiscreteActionWrapper,
    EarlyTermination,
    SpeedReward,
    CompleteLapReward,
)


# The Mario Kart-specific setup — track splits, checkpoint metrics, OOD logic,
# wrapped random baseline, and failure categories

GAME_NAME  = "SuperMarioKart-Snes"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EVAL_EPISODES = 50

# Same 5-action control space used by the trained agents.
# Kept here so this file can run on its own without depending on agent internals.
SIMPLE_ACTIONS = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 0: No Action
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 1: Gas
    [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # 2: Gas + Left
    [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # 3: Gas + Right
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 4: Brake / Reverse
]

# Saved PPO checkpoints for the first track.
PPO_CIRCUIT_M = [
    ("500",        "models/mario_cluster_ckpt_500_model_MarioCircuit_M"),
    ("1000",       "models/mario_cluster_ckpt_1000_model_MarioCircuit_M"),
    ("1500/early", "models/mario_cluster_ckpt_early_stop_model_MarioCircuit_M"),
]

# Saved PPO checkpoints for the second track.
PPO_CIRCUIT4_M = [
    ("500",   "models/mario_cluster_ckpt_MarioCircuit4_M_500"),
    ("1000",  "models/mario_cluster_ckpt_MarioCircuit4_M_1000"),
    ("1500",  "models/mario_cluster_ckpt_MarioCircuit4_M_1500"),
    ("2000",  "models/mario_cluster_ckpt_MarioCircuit4_M_2000"),
    ("2500",  "models/mario_cluster_ckpt_MarioCircuit4_M_2500"),
    ("final", "models/mario_cluster_ckpt_MarioCircuit4_M_final"),
]

# Saved DQN checkpoints for the first track.
DQN_CIRCUIT1 = [
    ("250",  "models/dqn_circuit1_ckpt_250"),
    ("500",  "models/dqn_circuit1_ckpt_500"),
    ("750",  "models/dqn_circuit1_ckpt_750"),
    ("1000", "models/dqn_circuit1_ckpt_1000"),
    ("1250", "models/dqn_circuit1_ckpt_1250"),
    ("1500", "models/dqn_circuit1_ckpt_1500"),
    ("1750", "models/dqn_circuit1_ckpt_1750"),
    ("2000", "models/dqn_circuit1_ckpt_2000"),
    ("2250", "models/dqn_circuit1_ckpt_2250"),
    ("2500", "models/dqn_circuit1_ckpt_2500"),  
    ("2750", "models/dqn_circuit1_ckpt_2750"),
    ("3000", "models/dqn_circuit1_ckpt_3000"),
]

# Saved DQN checkpoints for the second track.
DQN_CIRCUIT4 = [
    ("500",  "models/mario_run4_ckpt_500"),
    ("1500", "models/mario_run4_ckpt_1500"),
    ("2000", "models/mario_run4_ckpt_2000"),
    ("2250", "models/mario_run4_ckpt_2250"),
    ("2500", "models/mario_run4_ckpt_2500"),
    ("2750", "models/mario_run4_ckpt_2750"),
    ("3000", "models/mario_run4_ckpt_3000"),
]


def make_env(state):
    # Basic Retro environment constructor for a chosen saved state / track.
    return stable_retro.make(
        game=GAME_NAME,
        state=state,
        scenario=cfg.scenario if hasattr(cfg, "scenario") else "scenario",
        render_mode=cfg.render_mode,
        inttype=stable_retro.data.Integrations.ALL,
    )

def wrap_eval_env(env):
    '''this is inspired by existing RL examples, even though the Mario-specific wrappers are ours.
    We only use this helper for the random baseline, since DQN and PPO call their own wrap_env().'''
    env = MarioResize(env)
    env = MaxAndSkipEnv(env, skip=4)
    env = FrameStackObservation(env, 4)
    env = MarioToPyTorch(env)

    action_map = [np.array(a, dtype=np.int8) for a in SIMPLE_ACTIONS]
    env = DiscreteActionWrapper(env, action_map=action_map)

    env = EarlyTermination(env, max_no_progress_steps=600, stuck_penalty=-5)
    env = SpeedReward(env, scale=0.0001)
    env = CompleteLapReward(env)

    return env


def get_lap_threshold(agent_label):
    # We use a larger checkpoint threshold for Circuit4 because the lap is longer there.
    if "Circ4" in agent_label:
        return 205
    return 150

def categorise_failure(row):
    #failure logic
    agent      = row["agent"]
    ckpt       = row["final_checkpoint"]
    lap_thresh = get_lap_threshold(agent)

    if ckpt >= 5 and ckpt < lap_thresh:
        return "timeout"
    if ckpt <= 0:
        return "never_moved"
    if ckpt < 5:
        return "stuck_early"
    return "success"


def run_episodes(agent, env, agent_label, train_state, eval_state):
    """Run EVAL_EPISODES greedy episodes. Returns list of per-episode dicts."""
    #overall scaffold is generic RL evaluation boilerplate, but we adapted it to
    #log Mario-specific things like checkpoint progress and OOD status.
    #AI was used to help build this
    episode_log = []
    ood = (train_state != eval_state)   # marks when a model is evaluated on a different track than it was trained on

    for episode in range(EVAL_EPISODES):
        state, info = env.reset()
        episode_over  = False
        t             = 0
        episode_return = 0
        final_info    = info

        while not episode_over and (cfg.max_timesteps <= 0 or t < cfg.max_timesteps):
            action = agent.action_select(state) if hasattr(agent, 'action_select') else agent.select_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            episode_return += reward
            final_info      = info
            episode_over    = terminated or truncated
            state           = next_state
            t              += 1

        final_ckpt = get_checkpoint(final_info)
        timed_out  = (cfg.max_timesteps > 0 and t >= cfg.max_timesteps)
        stuck      = terminated and not timed_out

        episode_log.append({
            "agent":            agent_label,
            "episode":          episode,
            "return":           episode_return,
            "length":           t,
            "final_checkpoint": final_ckpt,
            "timed_out":        timed_out,
            "stuck":            stuck,
            "train_state":      train_state,
            "eval_state":       eval_state,
            "ood":              ood,
        })

    avg = float(np.mean([r["return"] for r in episode_log]))
    std = float(np.std ([r["return"] for r in episode_log]))
    ood_tag = " [OOD]" if ood else ""
    print(f"{agent_label:<35s}{ood_tag}  avg={avg:8.2f}  ±{std:.2f}")
    return episode_log, avg, std


def eval_ppo(label, checkpoint_path, train_state, eval_state):
    # The hyperparameters here match the PPO training run so the checkpoint loads correctly
    env = make_env(eval_state)
    agent = PPO_Agent(
        env,
        learning_rate=5e-5,
        rollout_steps=2048,
        minibatch_size=256,
        n_epochs=4,
        ent_coef_start=0.03,
        ent_coef_end=0.01,
        gae_lambda=0.95,
        clip_coef=0.1,
        max_grad_norm=0.5,
        total_timesteps=3_000_000,
        no_improve_tolerance=999999,
    )
    agent.load_checkpoint(checkpoint_path)
    env = agent.wrap_env(env)
    logs, avg, std = run_episodes(agent, env, label, train_state, eval_state)
    env.close()
    return logs, avg, std


def eval_dqn(label, checkpoint_path, train_state, eval_state):
    #greedy DQN evaluation. Setting epsilon to 0 makes this pure exploitation.
    env = make_env(eval_state)
    agent = Deep_RL_Agent(
        env,
        discount=0.99,
        learning_rate=0,
        epsilon_start=0.0,
        epsilon_min=0.0,
    )
    agent.load_checkpoint(checkpoint_path)
    env = agent.wrap_env(env)
    logs, avg, std = run_episodes(agent, env, label, train_state, eval_state)
    env.close()
    return logs, avg, std


def eval_random(eval_state):
    # Random baseline in the same wrapped environment so the comparison is fair
    env = make_env(eval_state)
    env = wrap_eval_env(env)
    agent = MarioKartRandomAgent(env)
    logs, avg, std = run_episodes(agent, env, "Random", eval_state, eval_state)
    env.close()
    return logs, avg, std


color_map = {
    "success":     "green",
    "never_moved": "red",
    "stuck_early": "orange",
    "timeout":     "royalblue",
}
failure_types = ["never_moved", "stuck_early", "timeout", "success"]


def _stacked_bar(ax, agents, df, fontsize=7):
    """Stacked bar chart of failure types for a list of agents."""
    #AI was used to help make sure it would generate a clean looking graph
    bottom = np.zeros(len(agents))
    for ftype in failure_types:
        counts = [(df[df["agent"] == a]["failure_type"] == ftype).sum()
                  for a in agents]
        ax.bar(range(len(agents)), counts, bottom=bottom,
               label=ftype, color=color_map.get(ftype, "gray"), alpha=0.85)
        bottom += np.array(counts, dtype=float)
    ax.set_xticks(range(len(agents)))
    ax.set_xticklabels(agents, rotation=45, ha="right", fontsize=fontsize)
    ax.set_ylabel("Episode Count")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")


def _add_ood_background(ax, agents):
    """Shade OOD columns with light red so they're visually distinct."""
    #visualization helper
    for i, a in enumerate(agents):
        if "OOD" in a:
            ax.axvspan(i - 0.5, i + 0.5, color="mistyrose", alpha=0.5, zorder=0)


def _add_divider(ax, in_dist_agents):
    """Dashed divider + labels between in-dist and OOD sections."""
    #AI was used to help add this so it would be easier to read
    split_idx = len(in_dist_agents) - 0.5
    ax.axvline(split_idx, color="black", linewidth=1.5, linestyle="--")
    ymax = ax.get_ylim()[1]
    ax.text(split_idx - 0.1, ymax * 0.95,
            "In-Dist", ha="right", fontsize=8,
            color="steelblue", fontweight="bold")
    ax.text(split_idx + 0.1, ymax * 0.95,
            "OOD →", ha="left", fontsize=8,
            color="darkorange", fontweight="bold")


def plot_progression(results_list, title, filename, out_dir):
    """Line chart of avg return vs checkpoint label for a single agent series."""
    #AI was used to edit this function to make sure we were generating a clean and easily interpreted visualization
    # standard matplotlib helper structure for line charts plus a shaded uncertainty band
    labels = [r["label"] for r in results_list]
    avgs   = [r["avg"]   for r in results_list]
    stds   = [r["std"]   for r in results_list]
    x      = range(len(labels))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, avgs, marker="o", color="steelblue", linewidth=2)
    ax.fill_between(x,
                    [a - s for a, s in zip(avgs, stds)],
                    [a + s for a, s in zip(avgs, stds)],
                    alpha=0.2, color="steelblue")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Avg Eval Return (50 episodes)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, filename), dpi=120)
    plt.close(fig)
    print(f"Saved {filename}")


def plot_comparison_bar(results_dict, title, filename, out_dir):
    """Bar chart comparing avg return across agents — shows best checkpoint per agent."""
    #AI was used here to make sure everything fit neatly and helped with formatting
    agents = list(results_dict.keys())
    avgs   = [results_dict[a]["avg"] for a in agents]
    stds   = [results_dict[a]["std"] for a in agents]

    fig, ax = plt.subplots(figsize=(max(10, len(agents) * 1.2), 5))
    ax.bar(range(len(agents)), avgs, yerr=stds, capsize=5,
           color="steelblue", alpha=0.8)
    ax.set_xticks(range(len(agents)))
    ax.set_xticklabels(agents, rotation=25, ha="right")
    ax.set_ylabel("Avg Eval Return (50 episodes)")
    ax.set_title(title)
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, filename), dpi=120)
    plt.close(fig)
    print(f"Saved {filename}")


def plot_ood_comparison(in_dist_results, ood_results, out_dir):
    """Side-by-side bars: in-distribution vs OOD track performance."""
    #AI was used for this function to help make sure everything fit nicely
    #cross-track generalization
    labels   = [r["label"] for r in in_dist_results]
    in_avgs  = [r["avg"]   for r in in_dist_results]
    ood_avgs = [r["avg"]   for r in ood_results]
    in_stds  = [r["std"]   for r in in_dist_results]
    ood_stds = [r["std"]   for r in ood_results]

    x     = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width/2, in_avgs,  width, yerr=in_stds,  capsize=4,
           label="In-distribution (MarioCircuit_M)",  color="steelblue", alpha=0.85)
    ax.bar(x + width/2, ood_avgs, width, yerr=ood_stds, capsize=4,
           label="Out-of-distribution (MarioCircuit4_M)", color="darkorange", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Ep {l}" for l in labels], rotation=15, ha="right")
    ax.set_ylabel("Avg Eval Return (50 episodes)")
    ax.set_title("PPO Generalisation: In-Distribution vs Out-of-Distribution Track")
    ax.legend()
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "ood_generalisation.png"), dpi=120)
    plt.close(fig)
    print("Saved ood_generalisation.png")


def plot_error_analysis(all_logs, out_dir):
    """
    6 separate error analysis figures split by track and OOD status:
      error_scatter.png         — all agents, coloured by failure type
      error_random.png          — random baseline only
      error_ppo_circuitM.png    — PPO Circuit_M: in-dist | OOD
      error_ppo_circuit4M.png   — PPO Circuit4_M: in-dist only
      error_dqn_circuitM.png    — DQN Circuit_M: in-dist | OOD
      error_dqn_circuit4M.png   — DQN Circuit4_M: in-dist only
    """
    #AI was used here so we ould generate clean and readable graphs based on the data collected
    #for the error analysis of each category
    # The actual split by algorithm, track, and OOD status is much more project-specific.
    df = pd.DataFrame(all_logs)
    df["failure_type"] = df.apply(categorise_failure, axis=1)

    agents_all = df["agent"].unique().tolist()

    agents_ppo_circM_in  = [a for a in agents_all if a.startswith("PPO-CircM") and "OOD" not in a]
    agents_ppo_circM_ood = [a for a in agents_all if a.startswith("PPO-CircM") and "OOD" in a]
    agents_ppo_circ4     = [a for a in agents_all if a.startswith("PPO-Circ4M")]
    agents_dqn_circ1_in  = [a for a in agents_all if a.startswith("DQN-Circ1") and "OOD" not in a]
    agents_dqn_circ1_ood = [a for a in agents_all if a.startswith("DQN-Circ1") and "OOD" in a]
    agents_dqn_circ4     = [a for a in agents_all if a.startswith("DQN-Circ4")]

    #All-agent scatter plot
    fig, ax = plt.subplots(figsize=(14, 6))
    for ftype, group in df.groupby("failure_type"):
        ax.scatter(group.index, group["return"],
                   label=ftype, alpha=0.4,
                   color=color_map.get(ftype, "gray"), s=12)
    ax.set_xlabel("Episode (across all agents)")
    ax.set_ylabel("Return")
    ax.set_title("Episode Returns by Failure Type (All Agents)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "error_scatter.png"), dpi=120)
    plt.close(fig)
    print("Saved error_scatter.png")

    # Random-only failure breakdown
    fig, ax = plt.subplots(figsize=(5, 6))
    _stacked_bar(ax, ["Random"], df, fontsize=10)
    ax.set_title("Failure Breakdown — Random Baseline")
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "error_random.png"), dpi=120)
    plt.close(fig)
    print("Saved error_random.png")

    #PPO on Circuit_M, showing in-distribution vs OOD side by side.
    agents_combined = agents_ppo_circM_in + agents_ppo_circM_ood
    fig, ax = plt.subplots(figsize=(10, 6))
    _stacked_bar(ax, agents_combined, df, fontsize=7)
    _add_ood_background(ax, agents_combined)
    _add_divider(ax, agents_ppo_circM_in)
    ax.set_title("Failure Breakdown — PPO MarioCircuit_M\n(In-Distribution | Out-of-Distribution)")
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "error_ppo_circuitM.png"), dpi=120)
    plt.close(fig)
    print("Saved error_ppo_circuitM.png")

    #PPO on Circuit4 only
    fig, ax = plt.subplots(figsize=(10, 6))
    _stacked_bar(ax, agents_ppo_circ4, df, fontsize=8)
    ax.set_title("Failure Breakdown — PPO MarioCircuit4_M (In-Distribution)")
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "error_ppo_circuit4M.png"), dpi=120)
    plt.close(fig)
    print("Saved error_ppo_circuit4M.png")

    #DQN on Circuit_M, showing in-distribution vs OOD side by side
    agents_combined = agents_dqn_circ1_in + agents_dqn_circ1_ood
    fig, ax = plt.subplots(figsize=(18, 6))
    _stacked_bar(ax, agents_combined, df, fontsize=6)
    _add_ood_background(ax, agents_combined)
    _add_divider(ax, agents_dqn_circ1_in)
    ax.set_title("Failure Breakdown — DQN MarioCircuit_M\n(In-Distribution | Out-of-Distribution)")
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "error_dqn_circuitM.png"), dpi=120)
    plt.close(fig)
    print("Saved error_dqn_circuitM.png")

    #DQN on Circuit4 only
    fig, ax = plt.subplots(figsize=(10, 6))
    _stacked_bar(ax, agents_dqn_circ4, df, fontsize=8)
    ax.set_title("Failure Breakdown — DQN MarioCircuit4_M (In-Distribution)")
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "error_dqn_circuit4M.png"), dpi=120)
    plt.close(fig)
    print("Saved error_dqn_circuit4M.png")

    #Text summary for the console and logs
    print("\n=== Failure Type Summary ===")
    summary = df.groupby(["agent", "failure_type"]).size().unstack(fill_value=0)
    print(summary.to_string())
    return df


def main():
    #AI was used to help form this based off of original matplot scaffolding and our ideas
    out_dir = "eval_results"
    os.makedirs(out_dir, exist_ok=True)

    all_logs    = []
    all_results = {}  

    #Random baseline
    print("\n── Random Baseline ──")
    logs, avg, std = eval_random("MarioCircuit_M")
    all_logs.extend(logs)
    all_results["Random"] = {"avg": avg, "std": std}

    #PPO on Circuit_M in-distribution
    print("\n── PPO MarioCircuit_M checkpoints (in-distribution) ──")
    circuit_m_in_dist = []
    for ep_label, ckpt_path in PPO_CIRCUIT_M:
        label = f"PPO-CircM-{ep_label}"
        logs, avg, std = eval_ppo(label, ckpt_path,
                                  train_state="MarioCircuit_M",
                                  eval_state ="MarioCircuit_M")
        all_logs.extend(logs)
        all_results[label] = {"avg": avg, "std": std}
        circuit_m_in_dist.append({"label": ep_label, "avg": avg, "std": std})

    #PPO trained on Circuit_M, tested on Circuit4
    print("\n── PPO MarioCircuit_M checkpoints (OUT-OF-DISTRIBUTION on Circuit4_M) ──")
    circuit_m_ood = []
    for ep_label, ckpt_path in PPO_CIRCUIT_M:
        label = f"PPO-CircM-{ep_label}-OOD"
        logs, avg, std = eval_ppo(label, ckpt_path,
                                  train_state="MarioCircuit_M",
                                  eval_state ="MarioCircuit4_M")
        all_logs.extend(logs)
        all_results[label] = {"avg": avg, "std": std}
        circuit_m_ood.append({"label": ep_label, "avg": avg, "std": std})

    #PPO on Circuit4 in-distribution
    print("\n── PPO MarioCircuit4_M checkpoints (in-distribution) ──")
    circuit4_m_series = []
    for ep_label, ckpt_path in PPO_CIRCUIT4_M:
        label = f"PPO-Circ4M-{ep_label}"
        logs, avg, std = eval_ppo(label, ckpt_path,
                                  train_state="MarioCircuit4_M",
                                  eval_state ="MarioCircuit4_M")
        all_logs.extend(logs)
        all_results[label] = {"avg": avg, "std": std}
        circuit4_m_series.append({"label": ep_label, "avg": avg, "std": std})

    #DQN on Circuit_M in-distribution
    print("\n── DQN Circuit1 checkpoints (in-distribution) ──")
    dqn_circuit1_series = []
    for ep_label, ckpt_path in DQN_CIRCUIT1:
        label = f"DQN-Circ1-{ep_label}"
        logs, avg, std = eval_dqn(label, ckpt_path,
                                  train_state="MarioCircuit_M",
                                  eval_state ="MarioCircuit_M")
        all_logs.extend(logs)
        all_results[label] = {"avg": avg, "std": std}
        dqn_circuit1_series.append({"label": ep_label, "avg": avg, "std": std})

    #DQN trained on Circuit_M, tested on Circuit4
    print("\n── DQN Circuit1 checkpoints (OUT-OF-DISTRIBUTION on Circuit4_M) ──")
    dqn_circuit1_ood = []
    for ep_label, ckpt_path in DQN_CIRCUIT1:
        label = f"DQN-Circ1-{ep_label}-OOD"
        logs, avg, std = eval_dqn(label, ckpt_path,
                                  train_state="MarioCircuit_M",
                                  eval_state ="MarioCircuit4_M")
        all_logs.extend(logs)
        all_results[label] = {"avg": avg, "std": std}
        dqn_circuit1_ood.append({"label": ep_label, "avg": avg, "std": std})

    #DQN on Circuit4 in-distribution
    print("\n── DQN Circuit4 checkpoints (in-distribution) ──")
    dqn_circuit4_series = []
    for ep_label, ckpt_path in DQN_CIRCUIT4:
        label = f"DQN-Circ4-{ep_label}"
        logs, avg, std = eval_dqn(label, ckpt_path,
                                  train_state="MarioCircuit4_M",
                                  eval_state ="MarioCircuit4_M")
        all_logs.extend(logs)
        all_results[label] = {"avg": avg, "std": std}
        dqn_circuit4_series.append({"label": ep_label, "avg": avg, "std": std})

    # Save the full episode-by-episode log
    df_all = pd.DataFrame(all_logs)
    df_all.to_csv(os.path.join(out_dir, "eval_log.csv"), index=False)
    print(f"\nRaw log saved → eval_results/eval_log.csv  ({len(df_all)} episodes)")

    #Console summary table
    print("\n=== Full Model Comparison Table ===")
    print(f"{'Agent':<35} {'Avg Return':>12} {'Std Dev':>10} "
          f"{'Avg Checkpoint':>16} {'Avg Length':>12}")
    print("-" * 90)
    for agent_label, vals in all_results.items():
        agent_rows = df_all[df_all["agent"] == agent_label]
        avg_ckpt   = agent_rows["final_checkpoint"].mean()
        avg_len    = agent_rows["length"].mean()
        ood_tag    = " [OOD]" if agent_rows["ood"].any() else ""
        print(f"{agent_label + ood_tag:<35} {vals['avg']:>12.2f} "
              f"{vals['std']:>10.2f} {avg_ckpt:>16.1f} {avg_len:>12.1f}")

    #Progression plots for both agents on both tracks
    plot_progression(circuit_m_in_dist,
                     "PPO Learning Progression — MarioCircuit_M",
                     "ppo_circuitM_progression.png", out_dir)
    plot_progression(circuit4_m_series,
                     "PPO Learning Progression — MarioCircuit4_M",
                     "ppo_circuit4M_progression.png", out_dir)
    plot_progression(dqn_circuit1_series,
                     "DQN Learning Progression — MarioCircuit_M",
                     "dqn_circuit1_progression.png", out_dir)
    plot_progression(dqn_circuit4_series,
                     "DQN Learning Progression — MarioCircuit4_M",
                     "dqn_circuit4_progression.png", out_dir)

    #PPO OOD comparison plot
    plot_ood_comparison(circuit_m_in_dist, circuit_m_ood, out_dir)

    #DQN OOD plot 
    labels   = [r["label"] for r in dqn_circuit1_series]
    in_avgs  = [r["avg"]   for r in dqn_circuit1_series]
    ood_avgs = [r["avg"]   for r in dqn_circuit1_ood]
    in_stds  = [r["std"]   for r in dqn_circuit1_series]
    ood_stds = [r["std"]   for r in dqn_circuit1_ood]
    x        = np.arange(len(labels))
    width    = 0.35
    fig, ax  = plt.subplots(figsize=(12, 5))
    ax.bar(x - width/2, in_avgs,  width, yerr=in_stds,  capsize=4,
           label="In-distribution (MarioCircuit_M)",      color="steelblue",  alpha=0.85)
    ax.bar(x + width/2, ood_avgs, width, yerr=ood_stds, capsize=4,
           label="Out-of-distribution (MarioCircuit4_M)", color="darkorange", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Ep {l}" for l in labels], rotation=20, ha="right")
    ax.set_ylabel("Avg Eval Return (50 episodes)")
    ax.set_title("DQN Generalisation: In-Distribution vs Out-of-Distribution Track")
    ax.legend()
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "dqn_ood_generalisation.png"), dpi=120)
    plt.close(fig)
    print("Saved dqn_ood_generalisation.png")

    #Best checkpoint per agent for one clean summary chart
    summary_results = {
        "Random":               all_results["Random"],
        "DQN-Circ1-3000":      all_results["DQN-Circ1-3000"],
        "DQN-Circ4-3000":      all_results["DQN-Circ4-3000"],
        "PPO-CircM-1500/early": all_results["PPO-CircM-1500/early"],
        "PPO-Circ4M-final":    all_results["PPO-Circ4M-final"],
    }
    plot_comparison_bar(summary_results,
                        "Best Agent Comparison — Super Mario Kart",
                        "agent_comparison.png", out_dir)

    plot_error_analysis(all_logs, out_dir)

    print("\n── All plots saved to eval_results/ ──")


if __name__ == "__main__":
    custom_path = os.path.join(SCRIPT_DIR, "custom_integrations")
    stable_retro.data.Integrations.add_custom_path(custom_path)
    main()