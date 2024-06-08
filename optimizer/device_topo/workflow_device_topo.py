import ast
import re
import subprocess

from networkx import DiGraph

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


def run_srun_command(nodes):
    try:
        result = subprocess.run([
            'srun',
            '--job-name=All_Device_Intra_Node_Bandwidth',
            '--time=00:30',
            f'--gpus={nodes}',
            '--gpus-per-node=1',
            f'--nodes={nodes}',
            '--ntasks-per-node=1',
            '--cpus-per-task=12',
            '--mem=1000',
            'python3', 'all_intra_node_topo_parallel.py'
        ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        if result.stdout:
            output = result.stdout.decode()
            return output
        else:
            print(f"Error: {result.stderr.decode()}")
            return None
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running the srun command: {e}")
        return None


def phase_slurm_2_DiGraphs(slurm_output: str) -> [DiGraph]:
    def check_slurm_row_pattern(row: str):
        pattern = re.compile(r"^bandwidths:  (\{.*\}) devices:  (\{.*\})$")
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
            print(bandwidths_part, devices_part)
            print(type(bandwidths_part), type(devices_part))
    '''
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
    '''
    return graph_list


if __name__ == "__main__":
    nodes = get_slurm_available_nodes()
    print(f"Number of available nodes: {nodes}")

    if nodes < 0:
        raise ValueError("No available nodes in Slurm to run the job.")

    output = run_srun_command(nodes)
    if output:
        phase_slurm_2_DiGraphs(output)
    else:
        raise ValueError("No available nodes in Slurm to run the job.")
