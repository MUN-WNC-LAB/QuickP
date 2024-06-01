# git clone https://github.com/NVIDIA/cuda-samples.git sudo apt-get install freeglut3-dev build-essential libx11-dev
# libxmu-dev libxi-dev libgl1-mesa-glx libglu1-mesa libglu1-mesa-dev libglfw3-dev libgles2-mesa-dev build-essential
# cmake git reset --hard v12.3 since the current CUDA version is 12.3 but avoid this issue
# https://github.com/NVIDIA/cuda-samples/issues/235. Choose the next commit e8568c417356f7e66bb9b7130d6be7e55324a519
import os
import subprocess
import re

# /usr/local/cuda-samples/bin/x86_64/linux/release/bandwidthTest --device=all --dtoh --htod --dtod
sample_addr = "/usr/local/cuda-samples/bin/x86_64/linux/release"
bandwidth_addr = os.path.join(sample_addr, "bandwidthTest")
print(bandwidth_addr)


def get_gpu_bandwidth():
    # Run the bandwidthTest utility
    result = subprocess.run(
        [bandwidth_addr, "--device=all", "--dtoh", "--htod", "--dtod"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    if result.returncode != 0:
        print(f"Error running bandwidthTest: {result.stderr.decode('utf-8')}")
        return None

    # Parse the output
    output = result.stdout.decode('utf-8')
    '''
    output_lines = [
    '[CUDA Bandwidth Test] - Starting...',
    '',
    '!!!!!Cumulative Bandwidth to be computed from all the devices !!!!!!',
    '',
    'Running on...',
    '',
    ' Device 0: NVIDIA GeForce RTX 3070',
    ' Quick Mode',
    '',
    ' Host to Device Bandwidth, 1 Device(s)',
    ' PINNED Memory Transfers',
    '   Transfer Size (Bytes)\tBandwidth(GB/s)',
    '   32000000\t\t\t24.6',
    '',
    ' Device to Host Bandwidth, 1 Device(s)',
    ' PINNED Memory Transfers',
    '   Transfer Size (Bytes)\tBandwidth(GB/s)',
    '   32000000\t\t\t26.3',
    '',
    ' Device to Device Bandwidth, 1 Device(s)',
    ' PINNED Memory Transfers',
    '   Transfer Size (Bytes)\tBandwidth(GB/s)',
    '   32000000\t\t\t389.4',
    '',
    'Result = PASS',
    '',
    'NOTE: The CUDA Samples are not meant for performance measurements. Results may vary when GPU Boost is enabled.',
    ''
    ]
    '''
    lines = output.split('\n')
    # Extract bandwidth information
    bandwidths = {}
    current_section = None
    for line in lines:
        if "Host to Device Bandwidth" in line:
            current_section = "Host to Device"
        elif "Device to Host Bandwidth" in line:
            current_section = "Device to Host"
        elif "Device to Device Bandwidth" in line:
            current_section = "Device to Device"
        elif current_section and "Bandwidth(GB/s)" in line:
            # Skip the header line within the current section
            continue
        elif "PINNED Memory Transfers" in line:
            continue
        elif current_section and line.strip():
            # Extract the transfer size and bandwidth
            parts = line.split()
            if len(parts) >= 2:
                bandwidth = parts[-1]
                bandwidths[current_section] = float(bandwidth)
                current_section = None  # Reset section after extracting the value
    return bandwidths


if __name__ == "__main__":
    bandwidths = get_gpu_bandwidth()
    if bandwidths:
        for direction, bandwidth in bandwidths.items():
            print(f"{direction}: {bandwidth:.2f} GB/s")
