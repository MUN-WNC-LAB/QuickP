# git clone https://github.com/NVIDIA/cuda-samples.git sudo apt-get install freeglut3-dev build-essential libx11-dev
# libxmu-dev libxi-dev libgl1-mesa-glx libglu1-mesa libglu1-mesa-dev libglfw3-dev libgles2-mesa-dev build-essential
# cmake git reset --hard v12.3 since the current CUDA version is 12.3 but avoid this issue
# https://github.com/NVIDIA/cuda-samples/issues/235. Choose the next commit e8568c417356f7e66bb9b7130d6be7e55324a519
import csv
import io
import os
import socket
import subprocess
import torch
import tensorflow as tf
from tensorflow.python.client import device_lib

# /usr/local/cuda-samples/bin/x86_64/linux/release/bandwidthTest --device=all --dtoh --htod --dtod
sample_addr = "/usr/local/cuda-samples/bin/x86_64/linux/release"
bandwidth_addr = os.path.join(sample_addr, "bandwidthTest")
print(bandwidth_addr)


def get_device_bandwidth():
    hostname = socket.gethostname()
    # check if GPU configuration is okay
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if not gpus:
        raise ValueError("No GPUs found")
    """Run the bandwidthTest utility and return the raw CSV output."""
    try:
        result = subprocess.run(
            [bandwidth_addr, "-device=all", "-dtoh", "-htod", "-dtod", "-csv"],
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

        device_info = []

        for device in device_lib.list_local_devices():

            device_info.append({
                'name': device.name,
                'device_type': device.device_type,
                'memory_limit': device.memory_limit
            })
            print(device_lib.list_local_devices())
        print(bandwidths, device_info)
        return bandwidths, device_info
    except subprocess.CalledProcessError as e:
        print(f"Error running bandwidthTest: {e}")

get_device_bandwidth()