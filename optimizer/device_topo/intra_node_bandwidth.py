"""
Step 1
# Step 1: Clone the CUDA Samples Repository
# Clone the repository to your local machine:
# git clone https://github.com/NVIDIA/cuda-samples.git /path/to/local/clone

# Step 2: Reset to a Suitable Version
# Navigate to the cloned repository and reset it to the commit that matches your CUDA version.
# For example, if your CUDA version is 12.3, run:
# cd /path/to/local/clone/cuda-samples
# git reset --hard e8568c417356f7e66bb9b7130d6be7e55324a519

# Step 3: Install Required Dependencies
# Install the necessary libraries and tools:
# sudo apt-get install freeglut3-dev build-essential libx11-dev libxmu-dev libxi-dev \
# libgl1-mesa-glx libglu1-mesa libglu1-mesa-dev libglfw3-dev libgles2-mesa-dev cmake

# Step 4: Build the CUDA Samples
# Navigate to the CUDA samples directory and compile the samples:
# cd /path/to/local/clone/cuda-samples
# make

# Step 5: Move the CUDA Samples
# Move the compiled CUDA samples to a desired path, for example:
# sudo mv /path/to/local/clone/cuda-samples /usr/local/cuda-samples

# Step 6: Update the PATH Environment Variable
# Add the CUDA samples binary directory to your PATH in .bashrc.
# If the CUDA samples are located in /usr/local/cuda-samples, the PATH will be:
# echo 'export PATH=/usr/local/cuda-samples/bin/x86_64/linux/release:$PATH' >> ~/.bashrc
# source ~/.bashrc

# Step 7: Run the Bandwidth Test
# Execute the bandwidth test with the following command:
# bandwidthTest --device=all --dtoh --htod --dtod
"""
import csv
import io
import os
import socket
import subprocess
import torch
import tensorflow as tf
from tensorflow.python.client import device_lib


def get_device_bandwidth():
    hostname = socket.gethostname()
    # check if GPU configuration is okay
    gpus = tf.config.experimental.list_physical_devices('GPU')
    # add env variables to path
    os.environ["PATH"] += os.pathsep + "/usr/local/cuda-samples/bin/x86_64/linux/release"

    if not gpus:
        raise ValueError("No GPUs found")
    """Run the bandwidthTest utility and return the raw CSV output."""
    try:
        result = subprocess.run(
            ["bandwidthTest", "-device=all", "-dtoh", "-htod", "-dtod", "-csv"],
            capture_output=True, text=True
        )
        csv_reader = csv.reader(io.StringIO(result.stdout))
        bandwidths = {}
        for line in csv_reader:
            if line and 'bandwidthTest' in line[0]:
                parts = line
                device_name = parts[0].split('-')[1]
                bandwidth = parts[1].split('=')[1].strip()
                # key = f"{device_name}-{hostname}"
                # Append the bandwidth data to the device entry
                bandwidths[device_name] = bandwidth

        # Function to extract device information
        def extract_device_info(device):
            return {
                'device_type': device.device_type,
                'memory_limit': device.memory_limit
                # Convert bytes to GB for GPU
            }

        # Get the list of local devices and map them to the desired format
        device_info = {}
        for device in device_lib.list_local_devices():
            device_info[f"{device.name}-{hostname}"] = extract_device_info(device)
        return bandwidths, device_info
    except subprocess.CalledProcessError as e:
        print(f"Error running bandwidthTest: {e}")
