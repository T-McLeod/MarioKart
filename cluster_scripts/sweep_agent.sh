#!/bin/bash
#SBATCH --job-name=mk-sweep
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00            # 6M env-steps/run for Hyperband survivors (~14h + margin)
#SBATCH --array=0-19%4              # 20 trials total, max 4 concurrent containers
#
# NOTE: --output / --error are intentionally NOT set here. SLURM does not expand
# env vars in #SBATCH directives, so the per-sweep log paths are passed on the
# `sbatch` command line by cluster_scripts/submit_sweep.sh (logs land in the
# per-sweep directory, named simply <jobid>.out / <jobid>.err).

set -euo pipefail

# EDIT REQUIRED: scratch space and image name.
XTMP=/usr/project/xtmp/tm419
IMAGE="$XTMP/my_model.sif"

# Require WANDB_API_KEY in the environment before submission.
if [ -z "${WANDB_API_KEY:-}" ]; then
    echo "Error: WANDB_API_KEY is not set. Please export it before submitting."
    exit 1
fi

# SWEEP_ID and SWEEP_DIR are provided by submit_sweep.sh. If sbatch was called
# directly, derive SWEEP_DIR from the sweep's short id (last path component).
if [ -z "${SWEEP_ID:-}" ]; then
    echo "Error: SWEEP_ID is not set. Submit via cluster_scripts/submit_sweep.sh <SWEEP_ID>."
    exit 1
fi
SWEEP_DIR="${SWEEP_DIR:-$XTMP/sweep_${SWEEP_ID##*/}}"

# Per-sweep output dirs (bound into the container).
mkdir -p "$SWEEP_DIR/models" "$SWEEP_DIR/videos" "$SWEEP_DIR/wandb"
cd "$SWEEP_DIR"

export APPTAINERENV_PYTHONUNBUFFERED=1
export APPTAINERENV_PYTHONNOUSERSITE=1
export APPTAINERENV_WANDB_API_KEY="$WANDB_API_KEY"   # auth inside the container
export APPTAINERENV_MK_STATE=MarioCircuit2_M         # select the map inside the container

# Each array task runs one sweep trial: the agent pulls a hyperparameter assignment
# from the W&B cloud controller and launches `python -m src.train` with those values.
apptainer exec --nv --no-home \
  --bind "$SWEEP_DIR/models":/workspace/MarioKart/models,"$SWEEP_DIR/videos":/workspace/MarioKart/videos,"$SWEEP_DIR/wandb":/workspace/MarioKart/wandb \
  --pwd /workspace/MarioKart "$IMAGE" \
  bash -c "wandb agent --count 1 $SWEEP_ID & wait"
