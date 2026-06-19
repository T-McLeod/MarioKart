# Phase 1 · Step 1 — PPO Hyperparameter Sweep (Nature-CNN baseline)

Bayesian W&B sweep over 4 PPO hyperparameters to find strong, architecture-neutral
hyperparameters before the Nature-CNN vs IMPALA comparison. ~20 runs × 6,000,000
timesteps on `MarioCircuit2_M`, with Hyperband early termination, deployed as a
SLURM job array of Apptainer containers (max 4 concurrent).

Files:
- `sweeps/phase1_nature_sweep.yaml` — the sweep definition.
- `cluster_scripts/sweep_agent.sh` — SLURM array launcher that runs `wandb agent`.
- `sweeps/phase1_best.yaml` — (you create this after the sweep) the locked winner for Step 2.

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

The source code is **baked into the `.sif` image** (`Dockerfile: COPY . .`). The
image must be rebuilt and re-pulled/transferred to the cluster before running a new
sweep. Rebuild + ship per `CLUSTER_DEPLOYMENT.md`. Verify the new image has the
sweep-aware `wandb.init()` fix:
```bash
apptainer exec my_model.sif grep -n "in_sweep" /workspace/MarioKart/src/train.py
```

## Run procedure

**Step 1 — create the sweep locally** (your WSL `mk_ai` env; wandb does not need to
be installed on the cluster login node):
```bash
wandb login
wandb sweep sweeps/phase1_nature_sweep.yaml
# → prints: Created sweep with ID: <short_id>
#           Run sweep agent with: wandb agent tannermcleod21-duke-university/mariokart-rl/<short_id>
```
Copy the full `tannermcleod21-duke-university/mariokart-rl/<short_id>` string.

**Step 2 — submit the array on the cluster.** The cluster login node does not support
running arbitrary shell scripts, so submit via `sbatch` directly. You must first
`mkdir` the per-sweep directory (SLURM will not create it), then pass the paths on the
command line (SLURM does not expand env vars in `#SBATCH` directives):
```bash
# On the cluster login node:
export WANDB_API_KEY=...
SWEEP_ID="tannermcleod21-duke-university/mariokart-rl/<short_id>"
SHORT_ID="${SWEEP_ID##*/}"           # extracts just <short_id>
SWEEP_DIR="/usr/project/xtmp/tm419/sweep_${SHORT_ID}"

mkdir -p "$SWEEP_DIR"

sbatch \
  --export=ALL,SWEEP_ID="$SWEEP_ID",SWEEP_DIR="$SWEEP_DIR" \
  --output="$SWEEP_DIR/%j.out" \
  --error="$SWEEP_DIR/%j.err" \
  cluster_scripts/sweep_agent.sh
```

**Monitor:**
```bash
squeue -u tm419                                        # confirm ≤ 4 running at once
tail -f /usr/project/xtmp/tm419/sweep_<short_id>/*.out
# ...and the W&B sweep page: parallel-coords + parameter-importance populate live.
```

Notes:
- The sweep object lives in the cloud, so creating it locally and submitting on the
  cluster is fully supported. The container nodes need outbound access to wandb.ai
  (they already have it — normal training streams metrics).
- The `WANDB_API_KEY` must belong to the same account (or a member of the
  `tannermcleod21-duke-university` team) that created the sweep.
- Change `%4` in the `--array` line of `sweep_agent.sh` to widen/narrow concurrency.
  Resubmit the same `sbatch` command with the same `SWEEP_ID` to add more trials
  (agents automatically rejoin the existing sweep).
- **EDIT REQUIRED** in `sweep_agent.sh`: the `XTMP` scratch path and `IMAGE` /
  `my_model.sif` name.

## Smoke test before the full run

Temporarily set `--total-timesteps` to `"40000"` in `phase1_nature_sweep.yaml` and
`--array=0-0` in `sweep_agent.sh`, recreate the sweep (`wandb sweep …`), then submit
the single-task array as above. In the W&B UI confirm:

- the run appears **under the sweep** in `mariokart-rl` (not standalone);
- it does **not** crash on the resume guard;
- `avg_return`, `pg_loss`, `entropy`, `approx_kl` are logging against `global_step`;
- swept params show on the run config as `learning-rate`, `clip-coef`, `ent-coef-start`, `rollout-steps`;
- config `state` is `MarioCircuit2_M`;
- no `wandb/ wasn't writable` warning.

Then restore `--total-timesteps` to `6000000` and `--array` to `0-19%4`, and recreate
the sweep for the full run.

## Selecting the winner

1. Export the **parallel-coordinates** plot and **parameter-importance** panel from
   the sweep page for the report.
2. Pick the run maximizing `avg_return`. If the top 2–3 are within noise, run a small
   confirmation set (≈3 seeds each of the contenders) before locking.
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
- **`RuntimeError: GET was unable to find an engine to execute this computation`** —
  PyTorch was upgraded past 2.1.0 and dropped Volta/sm_70 support. Fixed by pinning
  `torch==2.1.0` in `requirements.txt`; rebuild the image.
- **`avg_return` not appearing as a Y-axis option in W&B line charts** — metrics
  logged with an explicit `step=` parameter are indexed against `global_step` rather
  than W&B's internal `_step`. In the panel's Y-axis picker, type `avg_return`
  directly into the search field rather than scrolling the dropdown. The `define_metric`
  call in `train.py` fixes this automatically for all future runs from a rebuilt image.

## Hyperband bracketing (env-step based)

`src/train.py` logs with `wandb.log(metrics, step=global_step)`, so the x-axis is
**environment steps** and W&B Hyperband brackets on env-steps — runs with different
`rollout-steps` are compared at **equal experience** (this fixes the earlier unfair
early-culling of small-batch runs, which were judged at low update counts).

`min_iter` in the sweep YAML is therefore in **env-step units**. With `min_iter:
1000000` and `eta: 3`, bracket cuts fall at **1M** and **3M** env steps (of 6M
total): every run trains to ≥ 1M env steps before any cut. Lower `min_iter` to cut
earlier/more aggressively; raise it to cut later.
