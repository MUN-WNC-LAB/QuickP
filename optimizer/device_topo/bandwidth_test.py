# git clone https://github.com/NVIDIA/cuda-samples.git sudo apt-get install freeglut3-dev build-essential libx11-dev
# libxmu-dev libxi-dev libgl1-mesa-glx libglu1-mesa libglu1-mesa-dev libglfw3-dev libgles2-mesa-dev build-essential
# cmake git reset --hard v12.3 since the current CUDA version is 12.3 but avoid this issue
# https://github.com/NVIDIA/cuda-samples/issues/235. Choose the next commit e8568c417356f7e66bb9b7130d6be7e55324a519

import subprocess
import re


# /usr/local/cuda-samples/bin/x86_64/linux/release$ ./bandwidthTest --device=all --dtoh --htod --dtod
test_url = "/usr/local/cuda-samples/bin/x86_64/linux/release"

def get_gpu_bandwidth():
    # Run the bandwidthTest utility
    result = subprocess.run(
        ["./bandwidthTest", "--device=all", "--dtoh", "--htod", "--dtod"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    if result.returncode != 0:
        print(f"Error running bandwidthTest: {result.stderr.decode('utf-8')}")
        return None

    # Parse the output
    output = result.stdout.decode('utf-8')
    lines = output.split('\n')

    # Extract bandwidth information
    bandwidths = {}
    for line in lines:
        match = re.search(r"Device\s+(\d+)\s+to\s+Device\s+(\d+)\s+Bandwidth,\s+([0-9.]+)\s+GB/s", line)
        if match:
            device_from = int(match.group(1))
            device_to = int(match.group(2))
            bandwidth = float(match.group(3))
            bandwidths[(device_from, device_to)] = bandwidth

    for (device_from, device_to), bandwidth in bandwidths.items():
        print(f"Bandwidth from GPU {device_from} to GPU {device_to}: {bandwidth} GB/s")

    return bandwidths


if __name__ == "__main__":
    bandwidths = get_gpu_bandwidth()
    if bandwidths:
        print("Measured GPU-to-GPU bandwidths:")
        for (device_from, device_to), bandwidth in bandwidths.items():
            print(f"GPU {device_from} -> GPU {device_to}: {bandwidth:.2f} GB/s")