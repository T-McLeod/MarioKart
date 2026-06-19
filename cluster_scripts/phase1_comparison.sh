#!/bin/bash
#SBATCH --job-name=mk-phase1
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=18:00:00            # 5M env-steps/run (~12h on V100 with 8 envs + margin)
#SBATCH --array=0-5                # 6 runs: 2 architectures x 3 seeds, all concurrent
#
# Phase 1 Step 2 — Nature CNN vs IMPALA CNN architecture comparison.
# Locked hyperparameters from sweeps/phase1_best.yaml; 3 seeds per architecture.
#
# Array index → (agent, seed, run-name) mapping:
#   0: ppo_nature  seed=10  phase1-nature-s1
#   1: ppo_nature  seed=20  phase1-nature-s2
#   2: ppo_nature  seed=30  phase1-nature-s3
#   3: ppo_impala  seed=10  phase1-impala-s1
#   4: ppo_impala  seed=20  phase1-impala-s2
#   5: ppo_impala  seed=30  phase1-impala-s3
#
# NOTE: --output / --error are NOT set in the #SBATCH directives above because
# SLURM does not expand env vars there. Pass them on the sbatch command line.

set -euo pipefail

# EDIT REQUIRED: scratch space and image name.
XTMP=/usr/project/xtmp/tm419
IMAGE="$XTMP/my_model.sif"
COMP_DIR="$XTMP/phase1_comparison"

AGENTS=(ppo_nature ppo_nature ppo_nature ppo_impala ppo_impala ppo_impala)
SEEDS=(10 20 30 10 20 30)
NAMES=(phase1-nature-s1 phase1-nature-s2 phase1-nature-s3 phase1-impala-s1 phase1-impala-s2 phase1-impala-s3)

AGENT="${AGENTS[$SLURM_ARRAY_TASK_ID]}"
SEED="${SEEDS[$SLURM_ARRAY_TASK_ID]}"
RUN_NAME="${NAMES[$SLURM_ARRAY_TASK_ID]}"

if [ -z "${WANDB_API_KEY:-}" ]; then
    echo "Error: WANDB_API_KEY is not set. Export it before submitting."
    exit 1
fi

mkdir -p "$COMP_DIR/models" "$COMP_DIR/videos" "$COMP_DIR/wandb"
cd "$COMP_DIR"

export APPTAINERENV_PYTHONUNBUFFERED=1
export APPTAINERENV_PYTHONNOUSERSITE=1
export APPTAINERENV_WANDB_API_KEY="$WANDB_API_KEY"
export APPTAINERENV_MK_STATE=MarioCircuit2_M

echo "Starting: agent=$AGENT seed=$SEED name=$RUN_NAME"

apptainer exec --nv --no-home \
  --bind "$COMP_DIR/models":/workspace/MarioKart/models,"$COMP_DIR/videos":/workspace/MarioKart/videos,"$COMP_DIR/wandb":/workspace/MarioKart/wandb \
  --pwd /workspace/MarioKart "$IMAGE" \
  bash -c "python -u -m src.train \
    --agent $AGENT \
    --name $RUN_NAME \
    --seed $SEED \
    --learning-rate 3.5e-4 \
    --clip-coef 0.1 \
    --ent-coef-start 0.03 \
    --rollout-steps 2048 \
    --total-timesteps 5000000 \
    --num-envs 8 \
    --no-improve-tolerance 999999 \
    & wait"
