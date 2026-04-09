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
    && rm -rf /var/lib/apt/lists/*

# Copy your exported Conda dependencies into the container
COPY requirements.txt .

# Install Python packages (using --no-cache-dir keeps the image size small)
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your Mario Kart repository code into the container
COPY . .

# Set the default command to launch your training script
# (Update 'train.py' if your main script has a different name, like 'main.py' or 'run.py')
CMD ["python", "train.py"]