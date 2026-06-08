# Phase 1 · Step 1 — PPO Hyperparameter Sweep (Nature-CNN baseline)

Bayesian W&B sweep over 4 PPO hyperparameters to find strong, architecture-neutral
hyperparameters before the Nature-CNN vs IMPALA comparison. ~20 runs × 6,000,000
timesteps on `MarioCircuit2_M`, with Hyperband early termination, deployed as a
SLURM job array of Apptainer containers (max 4 concurrent).

Files:
- `sweeps/phase1_nature_sweep.yaml` — the sweep definition.
- `cluster_scripts/submit_sweep.sh` — submit wrapper: makes the per-sweep directory
  and routes logs into it.
- `cluster_scripts/sweep_agent.sh` — SLURM array launcher that runs `wandb agent`.
- `sweeps/phase1_best.yaml` — (you create this) the locked winner for Step 2.

Each sweep gets its own directory `/usr/project/xtmp/tm419/sweep_<short_id>/`
containing the SLURM logs (named simply `<jobid>.out` / `<jobid>.err`) and the bound
`models/ videos/ wandb/` output dirs, where `<short_id>` is the last component of the
sweep id (e.g. `6epvfgdt`).

## How it works (two layers)

- **W&B cloud controller** (`wandb sweep`) owns the Bayesian search + Hyperband
  early-stop decisions and hands each worker a hyperparameter assignment.
- **SLURM array** (`cluster_scripts/sweep_agent.sh`): each array task = 1 GPU = 1
  Apptainer container = 1 `wandb agent --count 1` = one trial. `--array=0-19%4` runs
  20 trials total, at most 4 at a time (the `%4` is the concurrency cap).
- Each agent fills the sweep YAML's `${args}` with the chosen
  `--learning-rate=… --clip-coef=… --ent-coef-start=… --rollout-steps=…` and runs
  `python -m src.train --agent ppo_nature …` inside the container. argparse turns
  those flags into `PPO_HYPERPARAMS`.
- The map is chosen by `MK_STATE`, passed into the container via
  `APPTAINERENV_MK_STATE=MarioCircuit2_M`. W&B auth crosses the same way via
  `APPTAINERENV_WANDB_API_KEY`.

The **entity** and **project** are pinned in the sweep YAML
(`tannermcleod21-duke-university` / `mariokart-rl`) so the sweep and its runs land in
the right place. (Your entity is your W&B username/team; find it any time with
`python -c "import wandb; print(wandb.Api().default_entity)"` or from a run URL
`wandb.ai/<entity>/<project>/...`.)

## ⚠️ Prerequisite: rebuild the container image

The source code is **baked into the `.sif` image** (`Dockerfile: COPY . .`, and
`CLUSTER_DEPLOYMENT.md` keeps the in-image source). Two fixes required for the sweep
live in the repo, so the image **must be rebuilt and re-pulled/transferred** before
running:

1. `src/train.py` — sweep-aware `wandb.init()` (otherwise the run comes up as
   "resumed" and crashes with *"Cannot override hyperparameters when resuming!"*).
2. `Dockerfile` — creates `models/ videos/ wandb/` so the `--bind` mounts attach
   (otherwise they fall back to an ephemeral temp dir).

Rebuild + ship per `CLUSTER_DEPLOYMENT.md` (build/push the Docker image, then on the
cluster re-pull `my_model.sif`, or rebuild locally and `scp` it). Verify the new
image has the fix:
```bash
apptainer exec my_model.sif grep -n "in_sweep" /workspace/MarioKart/src/train.py
```

## Run procedure

**Step 1 — create the sweep locally** (your WSL `mk_ai` env, where wandb works; the
login node does not need wandb):
```bash
wandb login
wandb sweep sweeps/phase1_nature_sweep.yaml
# → prints: Created sweep with ID: <id>
#           Run sweep agent with: wandb agent tannermcleod21-duke-university/mariokart-rl/<id>
```
Copy the full `tannermcleod21-duke-university/mariokart-rl/<id>` string.

**Step 2 — submit the array on the cluster** (login node only schedules; wandb runs
inside the container). The wrapper creates `sweep_<short_id>/` and routes logs there:
```bash
export WANDB_API_KEY=...                # same W&B account that created the sweep
./cluster_scripts/submit_sweep.sh tannermcleod21-duke-university/mariokart-rl/<id>
```

**Monitor:**
```bash
squeue -u tm419                                 # confirm ≤ 4 running at once
tail -f /usr/project/xtmp/tm419/sweep_<id>/*.out   # logs are <jobid>.out per task
# ...and the W&B sweep page: parallel-coords + parameter-importance populate live.
```

Notes:
- The sweep object lives in the cloud, so creating it locally and submitting on the
  cluster is fully supported. The container nodes need outbound access to wandb.ai
  (they already have it — normal training streams metrics).
- The `WANDB_API_KEY` on the cluster must belong to the same account (or a member of
  the `tannermcleod21-duke-university` team) that created the sweep.
- Change `%4` in the `--array` line of `sweep_agent.sh` to widen/narrow concurrency.
  Re-run `submit_sweep.sh` with the same id to add more trials (agents rejoin the
  same sweep).
- **EDIT REQUIRED**: the `XTMP` scratch path (in both `sweep_agent.sh` and
  `submit_sweep.sh`) and the `IMAGE` / `my_model.sif` name (in `sweep_agent.sh`).

## Smoke test before the full run

Submit a single short trial to confirm wiring end-to-end:

```bash
# Temporarily set --total-timesteps to "40000" in sweeps/phase1_nature_sweep.yaml,
# and --array=0-0 in cluster_scripts/sweep_agent.sh, recreate the sweep, then submit.
```

In the W&B UI confirm:
- the run appears **under the sweep** in `mariokart-rl` (not standalone, not in
  `MarioKart-src`);
- it does **not** crash on the resume guard;
- `avg_return`, `pg_loss`, `entropy`, `approx_kl` are logging;
- swept params show on the run config as `learning-rate`, `clip-coef`,
  `ent-coef-start`, `rollout-steps`;
- config `state` is `MarioCircuit2_M`;
- no `wandb/ wasn't writable` warning (binds attached), and no mid-run eval/video.

Then restore `--total-timesteps` to `3000000` and `--array` to `0-19%4`, and recreate
the sweep for the full run.

## Selecting the winner

1. Export the **parallel-coordinates** plot and **parameter-importance** panel from
   the sweep page for the report.
2. Pick the run maximizing `avg_return`. If the top 2–3 are within noise, run a small
   confirmation set (≈3 seeds each of the contenders) before locking — the "separate
   evaluation set" path.
3. Record the winning values in `sweeps/phase1_best.yaml` for Step 2 to consume:
   ```yaml
   learning_rate: <best>
   clip_coef: <best>
   ent_coef_start: <best>
   rollout_steps: <best>
   # fixed for Phase 1: minibatch_size 256, n_epochs 2, num_envs 8, gae_lambda 0.95
   ```

## Troubleshooting (errors seen during bring-up)

- **`ValueError: Cannot override hyperparameters when resuming!`** — old image. The
  pre-fix `train.py` passed `resume="allow"`, so sweep runs came up as resumed and
  tripped the guard. Fixed by the sweep-aware `wandb.init()` in `src/train.py`;
  rebuild the image.
- **`Ignored wandb.init() arg project/id when running a sweep`** — harmless after the
  fix (the sweep owns project/id). The project is pinned via the YAML's `project:`.
- **Runs landing in project `MarioKart-src`** — the sweep YAML had no `project:`, so
  `wandb sweep` auto-named one. Now pinned to `mariokart-rl`; recreate the sweep.
- **`Path /workspace/MarioKart/wandb/ wasn't writable, using system temp directory`**
  — bind targets missing in the read-only `.sif`. Fixed by the `mkdir` in the
  `Dockerfile`; rebuild the image. (Non-fatal: metrics still sync to the cloud from
  the temp dir, but the final checkpoint save and local run dirs would be lost.)

## Hyperband bracketing (env-step based)

`src/train.py` logs with `wandb.log(metrics, step=global_step)`, so the x-axis is
**environment steps** and W&B Hyperband brackets on env-steps — runs with different
`rollout-steps` are compared at **equal experience** (this fixes the earlier unfair
early-culling of small-batch runs, which were judged at low update counts).

`min_iter` in the sweep YAML is therefore in **env-step units**. With `min_iter:
1000000` and `eta: 3`, bracket cuts fall at **1M** and **3M** env steps (of 6M
total): every run trains to ≥ 1M env steps before any cut. Lower `min_iter` to cut
earlier/more aggressively; raise it to cut later. After 2–3 runs, confirm the first
cut lands near the expected env-step count and adjust.
