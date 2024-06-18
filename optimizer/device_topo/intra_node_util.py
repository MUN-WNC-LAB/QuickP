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
import ast
import csv
import io
import os
import re
import socket
import subprocess
import sys

import torch
import tensorflow as tf
from networkx import DiGraph
from tensorflow.python.client import device_lib

# Ensure the parent directory is in the Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
sys.path.append(project_root)
from optimizer.model.graph import DeviceGraph


def phase_slurm_intra_2_DiGraphs(slurm_output: str) -> [DiGraph]:
    def check_slurm_row_pattern(row: str):
        pattern = re.compile(r"^bandwidths:\s+(\{.*\})\s+devices:\s+(\{.*\})$")
        match = pattern.match(row)
        if match:
            # ast.literal_eval convert string to dict
            bandwidths = ast.literal_eval(match.group(1))
            devices = ast.literal_eval(match.group(2))
            return bandwidths, devices
        else:
            return None

    # Function to get a key that includes a specific substring
    def get_key_including_substring(d, substring):
        for key in d:
            if substring in key:
                return key
        return None  # Return None if no such key is found

    graph_list = []
    lines = slurm_output.splitlines()
    for line in lines:
        bandwidths_part, devices_part = check_slurm_row_pattern(line)
        if bandwidths_part and devices_part:
            G = DeviceGraph()
            for (name, attributes) in devices_part.items():
                G.add_new_node(name, attributes["memory_limit"])
            for (direction, band) in bandwidths_part.items():
                if direction == "H2D":
                    from_device = get_key_including_substring(G.nodes, "CPU:0")
                    to_device = get_key_including_substring(G.nodes, "GPU:0")
                elif direction == "D2H":
                    from_device = get_key_including_substring(G.nodes, "GPU:0")
                    to_device = get_key_including_substring(G.nodes, "CPU:0")
                else:
                    continue
                if not from_device or not to_device:
                    raise ValueError("device not found")
                G.update_link_bandwidth(from_device, to_device, band)
            graph_list.append(G)
    return graph_list


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
