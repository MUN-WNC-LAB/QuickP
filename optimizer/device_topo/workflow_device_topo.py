import ast
import re
import socket
import subprocess

from networkx import DiGraph

from optimizer.model.graph import DeviceGraph, combine_graphs, visualize_graph
from slurm_util import get_server_ips, get_slurm_available_nodes

output_path = "device_intra_node_output.txt"
sh_path = "all_device_intra.sh"


def run_srun_command(num_nodes: int, intra: bool):
    path = 'all_intra_node_topo_parallel.py' if intra else 'all_intel_node_topo_parallel.py'
    try:
        result = subprocess.run([
            'srun',
            '--job-name=All_Device_Intra_Node_Bandwidth',
            '--time=00:30',
            f'--gpus={num_nodes}',
            '--gpus-per-node=1',
            f'--nodes={num_nodes}',
            '--ntasks-per-node=1',
            '--cpus-per-task=12',
            '--mem=1000',
            'python3', f'{path}'
        ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        if result.stdout:
            return result.stdout.decode()
        else:
            print(f"Error: {result.stderr.decode()}")
            return None
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running the srun command: {e}")
        return None


def gather_intel_bandwidth_data(servers: dict, num_nodes: int):

    all_results = {}
    local_hostname = socket.gethostname()
    other_servers = {key: value for key, value in servers.items() if key != local_hostname}
    print(servers, other_servers)
    '''
    for target in other_servers:
        print(f"bandwidth_test_{server}_to_{target}")
        result = run_srun_command(intra=False, num_nodes=num_nodes)
        server_results[target] = result

    all_results[server] = server_results
    '''
    return all_results


def phase_slurm_intra_2_DiGraphs(slurm_output: str) -> [DiGraph]:
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


if __name__ == "__main__":
    servers = get_server_ips()
    nodes = get_slurm_available_nodes()
    print(f"Number of available nodes: {nodes}")

    if nodes < 0:
        raise ValueError("No available nodes in Slurm to run the job.")
    if nodes != len(servers):
        raise ValueError("Number of nodes does not match the number of servers")

    output_intra = run_srun_command(nodes, intra=True)
    # output_intel = run_srun_command(nodes, intra=False)
    gather_intel_bandwidth_data(servers, nodes)
    if output_intra:
        graph_list_intra = phase_slurm_intra_2_DiGraphs(output_intra)
        graph_list_intel = []
        graph_combined = combine_graphs(graph_list_intra + graph_list_intel)
        visualize_graph(graph_combined)
    else:
        raise ValueError("No available nodes in Slurm to run the job.")
