import subprocess
import time

from networkx import DiGraph

from optimizer.device_topo.intra_node_bandwidth import get_device_bandwidth
from optimizer.model.graph import DeviceGraph

output_path = "device_intra_node_output.txt"
sh_path = "all_device_intra.sh"


def get_slurm_available_nodes():
    try:
        # Run sinfo command to get the number of idle nodes
        result = subprocess.run(['sinfo', '--noheader', '--states=idle', '--format=%D'],
                                stdout=subprocess.PIPE, text=True, check=True)

        # Parse the output to get the total number of available nodes
        available_nodes = sum(int(x) for x in result.stdout.split())

        return available_nodes
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running sinfo: {e}")
        return 0


def create_slurm_script(nodes, out_path):
    script_content = f"""#!/bin/bash
#SBATCH --job-name=All_Device_Intra_Node_Bandwidth
#SBATCH --time=00:30

#SBATCH --gpus={nodes}              
#SBATCH --gpus-per-node=1     
#SBATCH --nodes={nodes}           
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12    
#SBATCH --output={out_path}
#SBATCH --error=/dev/null     
#SBATCH --mem=1000            

### the command to run
srun python3 all_intra_node_topo_parallel.py
"""
    with open(sh_path, "w") as f:
        f.write(script_content)


def submit_slurm_script():
    try:
        result = subprocess.run(['sbatch', sh_path], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        job_id = result.stdout.decode().strip().split()[-1]
        print(f"Job {job_id} submitted successfully.")
        return job_id
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while submitting the job: {e}")
        return None


def retrieve_slurm_output():
    with open('device_intra_node_output.txt', 'r') as file:
        lines = file.readlines()

    for line in lines:
        print(line.strip())  # strip() to remove leading/trailing whitespace including newline characters


def phase_slurm_2_DiGraph() -> DiGraph:
    # Function to get a key that includes a specific substring
    def get_key_including_substring(d, substring):
        for key in d:
            if substring in key:
                return key
        return None  # Return None if no such key is found
    G = DeviceGraph()
    bandwidths, devices = get_device_bandwidth()
    for (name, attributes) in devices.items():
        G.add_new_node(name, attributes["memory_limit"])
    for (direction, band) in bandwidths.items():
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
    print("INFO ROW: ", G.edges.data())
    return G


if __name__ == "__main__":
    nodes = get_slurm_available_nodes()
    print(f"Number of available nodes: {nodes}")

    if nodes < 0:
        raise ValueError("No available nodes in Slurm to run the job.")

    create_slurm_script(nodes, output_path)
    submit_slurm_script()
    # guarantee the txt file contain the full info
    #time.sleep(5)
    retrieve_slurm_output()
