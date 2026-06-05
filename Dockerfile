# Use the official PyTorch image with CUDA support pre-installed
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Set the working directory inside the container
WORKDIR /workspace/MarioKart

# Install system-level dependencies (Crucial for emulator rendering and OpenCV)
# These prevent "missing shared library" errors when the environment boots
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy your exported Conda dependencies into the container
COPY requirements.txt .

# Install Python packages (using --no-cache-dir keeps the image size small)
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Ensure the host-bind mountpoints exist inside the image. They are gitignored
# (so absent from `COPY . .`), and Apptainer cannot create bind targets on a
# read-only .sif — without these dirs the --bind silently fails and wandb/models/
# videos fall back to an ephemeral temp dir.
RUN mkdir -p /workspace/MarioKart/models /workspace/MarioKart/videos /workspace/MarioKart/wandb

ENV PYTHONUNBUFFERED=1

# Set the default command to launch your training script
CMD ["python", "-u", "-m", "src.train", "--agent", "ppo_nature"]