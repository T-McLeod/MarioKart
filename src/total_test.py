import stable_retro
from .agents.random_agent import MarioKartRandomAgent
from .agents.deep_rl_agent import Deep_RL_Agent
from .agents.ppo_agent import PPO_Agent
from . import config as cfg
import numpy as np
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
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

GAME_NAME  = "SuperMarioKart-Snes"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EVAL_EPISODES = 50

SIMPLE_ACTIONS = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 0: No Action
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 1: Gas
    [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # 2: Gas + Left
    [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # 3: Gas + Right
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 4: Brake / Reverse
]

# ── PPO checkpoints ───────────────────────────────────────────────────────────
# Track 1: MarioCircuit_M  (early-stopped ~ep 1500)
PPO_CIRCUIT_M = [
    ("500",        "models/mario_cluster_ckpt_500_model_MarioCircuit_M"),
    ("1000",       "models/mario_cluster_ckpt_1000_model_MarioCircuit_M"),
    ("1500/early", "models/mario_cluster_ckpt_early_stop_model_MarioCircuit_M"),
]

# Track 2: MarioCircuit4_M  (ran to 2500 + final)
PPO_CIRCUIT4_M = [
    ("500",   "models/mario_cluster_ckpt_MarioCircuit4_M_500"),
    ("1000",  "models/mario_cluster_ckpt_MarioCircuit4_M_1000"),
    ("1500",  "models/mario_cluster_ckpt_MarioCircuit4_M_1500"),
    ("2000",  "models/mario_cluster_ckpt_MarioCircuit4_M_2000"),
    ("2500",  "models/mario_cluster_ckpt_MarioCircuit4_M_2500"),
    ("final", "models/mario_cluster_ckpt_MarioCircuit4_M_final"),
]


# ── DQN checkpoints ───────────────────────────────────────────────────────────
# Circuit 1 / MarioCircuit_M
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

# Circuit 4 / MarioCircuit4_M
DQN_CIRCUIT4 = [
    ("500",  "models/mario_run4_ckpt_500"),
    ("1500", "models/mario_run4_ckpt_1500"),
    ("2000", "models/mario_run4_ckpt_2000"),
    ("2250", "models/mario_run4_ckpt_2250"),
    ("2500", "models/mario_run4_ckpt_2500"),
    ("2750", "models/mario_run4_ckpt_2750"),
    ("3000", "models/mario_run4_ckpt_3000"),
]


# ── Environment factory ───────────────────────────────────────────────────────
def make_env(state):
    return stable_retro.make(
        game=GAME_NAME,
        state=state,
        scenario=cfg.scenario if hasattr(cfg, "scenario") else "scenario",
        render_mode=cfg.render_mode,
        inttype=stable_retro.data.Integrations.ALL,
    )

def wrap_eval_env(env):
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

# ── Failure categorisation ────────────────────────────────────────────────────
def categorise_failure(row):
    if row["final_checkpoint"] == 0 and not row["timed_out"]:
        return "never_moved"
    if row["stuck"] and row["final_checkpoint"] < 5:
        return "stuck_early"
    if row["timed_out"]:
        return "timeout"
    return "success"


# ── Core eval loop ────────────────────────────────────────────────────────────
def run_episodes(agent, env, agent_label, train_state, eval_state):
    """Run EVAL_EPISODES greedy episodes. Returns list of per-episode dicts."""
    episode_log = []
    ood = (train_state != eval_state)   # out-of-distribution flag

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


# ── PPO evaluator ─────────────────────────────────────────────────────────────
def eval_ppo(label, checkpoint_path, train_state, eval_state):
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


# ── DQN evaluator ─────────────────────────────────────────────────────────────
def eval_dqn(label, checkpoint_path, train_state, eval_state):
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


# ── Random baseline ───────────────────────────────────────────────────────────
def eval_random(eval_state):
    env = make_env(eval_state)
    env = wrap_eval_env(env)

    agent = MarioKartRandomAgent(env)
    logs, avg, std = run_episodes(agent, env, "Random", eval_state, eval_state)

    env.close()
    return logs, avg, std


# ── Plots ─────────────────────────────────────────────────────────────────────
def plot_progression(results_list, title, filename, out_dir):
    """Line chart of avg return vs checkpoint label for a single agent series."""
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
    """Bar chart comparing avg return across multiple agents/checkpoints."""
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
    """
    Side-by-side bars: MarioCircuit_M model tested on its own track vs
    MarioCircuit4_M (out-of-distribution). Shows generalisation gap.
    """
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
    """Two-panel failure breakdown across all evaluated agents."""
    df = pd.DataFrame(all_logs)
    df["failure_type"] = df.apply(categorise_failure, axis=1)

    color_map = {
        "success":     "green",
        "never_moved": "red",
        "stuck_early": "orange",
        "timeout":     "royalblue",
    }

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Panel 1 — scatter coloured by failure type
    for ftype, group in df.groupby("failure_type"):
        axes[0].scatter(group.index, group["return"],
                        label=ftype, alpha=0.5,
                        color=color_map.get(ftype, "gray"), s=15)
    axes[0].set_xlabel("Episode (across all agents)")
    axes[0].set_ylabel("Return")
    axes[0].set_title("Episode Returns by Failure Type")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Panel 2 — stacked bar per agent
    agents_order  = df["agent"].unique().tolist()
    failure_types = ["never_moved", "stuck_early", "timeout", "success"]
    bottom        = np.zeros(len(agents_order))

    for ftype in failure_types:
        counts = [(df[df["agent"] == a]["failure_type"] == ftype).sum()
                  for a in agents_order]
        axes[1].bar(agents_order, counts, bottom=bottom,
                    label=ftype, color=color_map.get(ftype, "gray"), alpha=0.85)
        bottom += np.array(counts, dtype=float)

    axes[1].set_xlabel("Agent")
    axes[1].set_ylabel("Episode Count")
    axes[1].set_title("Failure Type Breakdown per Agent")
    axes[1].legend()
    axes[1].tick_params(axis="x", rotation=30)
    axes[1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "error_analysis.png"), dpi=120)
    plt.close(fig)
    print("Saved error_analysis.png")

    # Print failure summary
    print("\n=== Failure Type Summary ===")
    summary = df.groupby(["agent", "failure_type"]).size().unstack(fill_value=0)
    print(summary.to_string())
    return df


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    out_dir = "eval_results"
    os.makedirs(out_dir, exist_ok=True)

    all_logs   = []
    all_results = {}   # label -> {avg, std}

    # ── 1. Random baseline (on both tracks) ──────────────────────────────────
    print("\n── Random Baseline ──")
    logs, avg, std = eval_random("MarioCircuit_M")
    all_logs.extend(logs)
    all_results["Random"] = {"avg": avg, "std": std}

    # ── 2. PPO — MarioCircuit_M (in-distribution eval) ───────────────────────
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

    # ── 3. PPO — MarioCircuit_M models tested on MarioCircuit4_M (OOD) ───────
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

    # ── 4. PPO — MarioCircuit4_M (in-distribution eval) ─────────────────────
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

    # ── 5. DQN — Circuit1/MarioCircuit_M (in-distribution) ───────────────────
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

    # ── 6. DQN — Circuit1 models tested on MarioCircuit4_M (OOD) ─────────────
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

    # ── 7. DQN — Circuit4/MarioCircuit4_M (in-distribution) ──────────────────
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

    # ── 8. Save raw data ──────────────────────────────────────────────────────
    df_all = pd.DataFrame(all_logs)
    df_all.to_csv(os.path.join(out_dir, "eval_log.csv"), index=False)
    print(f"\nRaw log saved → eval_results/eval_log.csv  ({len(df_all)} episodes)")

    # ── 9. Print comparison table ─────────────────────────────────────────────
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

    # ── 10. Plots ─────────────────────────────────────────────────────────────
    # PPO progressions
    plot_progression(
        circuit_m_in_dist,
        "PPO Learning Progression — MarioCircuit_M",
        "ppo_circuitM_progression.png", out_dir
    )
    plot_progression(
        circuit4_m_series,
        "PPO Learning Progression — MarioCircuit4_M",
        "ppo_circuit4M_progression.png", out_dir
    )

    # DQN progressions
    plot_progression(
        dqn_circuit1_series,
        "DQN Learning Progression — MarioCircuit_M",
        "dqn_circuit1_progression.png", out_dir
    )
    plot_progression(
        dqn_circuit4_series,
        "DQN Learning Progression — MarioCircuit4_M",
        "dqn_circuit4_progression.png", out_dir
    )

    # PPO OOD comparison
    plot_ood_comparison(circuit_m_in_dist, circuit_m_ood, out_dir)

    # DQN OOD comparison — reuse same function with new filename
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

    # Full agent comparison — PPO final vs DQN final vs Random only
    # (too many checkpoints to show all, so just show best of each)
    summary_results = {
        "Random":           all_results["Random"],
        "DQN-Circ1-3000":  all_results["DQN-Circ1-3000"],
        "DQN-Circ4-3000":  all_results["DQN-Circ4-3000"],
        "PPO-CircM-1500/early": all_results["PPO-CircM-1500/early"],
        "PPO-Circ4M-final": all_results["PPO-Circ4M-final"],
    }
    plot_comparison_bar(
        summary_results,
        "Best Agent Comparison — Super Mario Kart",
        "agent_comparison.png", out_dir
    )

    # Error analysis across all agents
    plot_error_analysis(all_logs, out_dir)

    print("\n── All plots saved to eval_results/ ──")


if __name__ == "__main__":
    custom_path = os.path.join(SCRIPT_DIR, "custom_integrations")
    stable_retro.data.Integrations.add_custom_path(custom_path)
    main()