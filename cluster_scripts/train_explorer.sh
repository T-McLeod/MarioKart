#!/bin/bash
#SBATCH --job-name=mk_explorer
#SBATCH --output=mk_explorer_%j.out      
#SBATCH --error=mk_explorer_%j.err       
#SBATCH --partition=compsci-gpu               
#SBATCH --gres=gpu:1                  
#SBATCH --mem=32G                     
#SBATCH --cpus-per-task=4             
#SBATCH --time=24:00:00               

# Navigate to your CS scratch space
# EDIT REQUIRED: Update the path below to point to your specific scratch space or working directory.
cd /usr/project/xtmp/tm419/

# Require WANDB_API_KEY to be set in the environment before submission
if [ -z "$WANDB_API_KEY" ]; then
    echo "Error: WANDB_API_KEY is not set. Please export it before running sbatch."
    exit 1
fi

export APPTAINERENV_PYTHONUNBUFFERED=1
export APPTAINERENV_PYTHONNOUSERSITE=1

# Create output directories on the host so they can be bound
mkdir -p models videos wandb container_home

# EDIT REQUIRED: Update 'my_model.sif' in the command below to match your compiled Apptainer image name if different.
apptainer exec --nv --no-home --bind models:/workspace/MarioKart/models,videos:/workspace/MarioKart/videos,wandb:/workspace/MarioKart/wandb --pwd /workspace/MarioKart my_model.sif bash -c "python -u -m src.ppo_train --name \"exploration-focused-v3\" --ent-coef-start 0.05 --ent-coef-end 0.01 --total-timesteps 50000000 --no-improve-tolerance 999999 & wait"
