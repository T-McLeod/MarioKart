# Deploying MarioKart RL to the Duke CS GPU Cluster

Because cluster compute nodes do not allow standard Docker root privileges, deployment requires wrapping your Docker image into an Apptainer (Singularity) image format (`.sif`).

You can either pull the image directly from Docker Hub while logged into the cluster, or build the image locally and manually transfer it. Both methods require setting up your cluster scratch space first.

## Phase 1: Set Up Cluster Scratch Space
The Computer Science cluster (`login.cs.duke.edu`) uses a dedicated, high-speed temporary storage drive called `xtmp` for heavy training workloads. Home directories (`/home/users/...`) have strict quotas and should not be used for model training or storing `.sif` files.

1. **SSH into the CS cluster:**
   ```bash
   ssh tm419@login.cs.duke.edu
   ```

2. **Create and navigate to your scratch folder:**
   ```bash
   mkdir -p /usr/project/xtmp/tm419
   cd /usr/project/xtmp/tm419
   ```

## Phase 2: Get the Container onto the Cluster (Choose One)

### Option A: Pull Directly from Docker Hub (On the Cluster)
When pulling a large image directly on the Duke CS login nodes, you must bypass two system restrictions: the strict `/tmp` storage quota and a kernel `ptrace` security block.

Run these commands inside your `xtmp` directory:

```bash
# 1. Create a local temporary directory to bypass system /tmp limits
mkdir -p apptainer_tmp

# 2. Route Apptainer's extraction and caching to your high-capacity xtmp folder
export APPTAINER_TMPDIR=$PWD/apptainer_tmp
export APPTAINER_CACHEDIR=$PWD/apptainer_tmp

# 3. Bypass the cluster's kernel security bug that blocks mksquashfs
export PROOT_NO_SECCOMP=1

# 4. Pull the image
apptainer pull my_model.sif docker://yourusername/mariokart-rl:latest

# 5. Clean up the massive temporary extraction files to save space
rm -rf apptainer_tmp
```

### Option B: Build Locally and Transfer via SCP (Manual Transfer)
If you hit network timeouts or prefer to build locally, use Windows Subsystem for Linux (WSL). To avoid Windows filesystem permission errors and WSL memory limits, build the image in your native Linux home directory (`~`).

On your local WSL terminal:

```bash
# 1. Navigate to your Linux home directory (NOT /mnt/c/...)
cd ~

# 2. Route temporary files to your home directory to avoid RAM/tmp limits
mkdir -p ~/apptainer_tmp
export APPTAINER_TMPDIR=~/apptainer_tmp
export APPTAINER_CACHEDIR=~/apptainer_tmp

# 3. Pull the image
apptainer pull my_model.sif docker://yourusername/mariokart-rl:latest

# 4. Transfer the compiled file directly to your cluster scratch space
scp my_model.sif tm419@login.cs.duke.edu:/usr/project/xtmp/tm419/

# 5. Clean up local temp files
rm -rf ~/apptainer_tmp
```

## Phase 3: Configure and Run the Slurm Batch Scripts

Instead of a generic `train.sh`, you should use the pre-configured scripts located in the `cluster_scripts/` directory of your local repository (`train_aggressive.sh`, `train_explorer.sh`, `train_long_horizon.sh`). 

Transfer these scripts to your cluster scratch space (`/usr/project/xtmp/tm419/`).

### Important Script Details
These scripts have been specifically modified for your workflow:
- **X11 / Headless Rendering:** X-server/`xvfb-run` dependencies have been removed as `stable-retro` handles rendering natively.
- **Isolating Host Packages:** We explicitly use `--no-home` and `export PYTHONNOUSERSITE=1` to prevent Apptainer from leaking your cluster's local python packages (like `cv2` / `opencv-python-headless`) into the container, preventing `glibc` mismatch crashes.
- **Selective Binding:** We explicitly run `mkdir -p models videos wandb` and only `--bind` those specific outputs back to your host machine. This preserves the original source code baked inside the container while safely exporting your training logs, MP4 replays, and `.pth` checkpoints.

### Queueing the Job
Before running any of the scripts, you must export your Weights & Biases API key. `sbatch` will automatically inherit this key and pass it to the compute node.

```bash
# 1. Export your API key
export WANDB_API_KEY="your_actual_key_here"

# 2. Submit the jobs
sbatch train_aggressive.sh
sbatch train_explorer.sh
sbatch train_long_horizon.sh
```

## Phase 4: Submit and Monitor
1. **Monitor job state (`PD` = Pending, `R` = Running):**
   ```bash
   squeue -u tm419
   ```

2. **Stream your training logs in real-time:**
   ```bash
   tail -f mk_aggressive_JOBID.out
   tail -f mk_aggressive_JOBID.err
   ```

> ### ⚠️ Cluster Storage Policy Reminder
> The `/usr/project/xtmp/` directory has an automated **180-day purge policy**. Any dataset, log, or model checkpoint that has not been modified or touched within 180 days will be permanently deleted. Once training is complete, copy your final model weights (from the `models/` directory) and gameplay clips (from `videos/`) back to your `/home` directory or download them locally for permanent storage.
