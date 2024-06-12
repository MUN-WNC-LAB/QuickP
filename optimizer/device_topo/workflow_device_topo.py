import ast
import json
import re
import socket
import subprocess

from networkx import DiGraph

from optimizer.device_topo.intra_node_bandwidth import phase_slurm_intra_2_DiGraphs
from optimizer.model.graph import DeviceGraph, combine_graphs, visualize_graph
from slurm_util import get_server_ips, get_slurm_available_nodes

output_path = "device_intra_node_output.txt"
sh_path = "all_device_intra.sh"


def run_srun_command(num_nodes: int, intra: bool):
    path = 'all_intra_node_topo_parallel.py' if intra else 'all_intel_node_topo_parallel.py'
    command = [
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
        ]
    if not intra:
        # # Serialize the dictionary to a JSON string
        command.extend(['--dict', json.dumps(get_server_ips())])
    try:
        result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        if result.stdout:
            return result.stdout.decode()
        else:
            print(f"Error: {result.stderr.decode()}")
            return None
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running the srun command: {e}")
        return None


if __name__ == "__main__":
    servers = get_server_ips()
    nodes = get_slurm_available_nodes()
    print(f"Number of available nodes: {nodes}")

    if nodes < 0:
        raise ValueError("No available nodes in Slurm to run the job.")
    if nodes != len(servers):
        raise ValueError("Number of nodes does not match the number of servers")

    output_intra = run_srun_command(nodes, intra=True)
    output_intel = run_srun_command(nodes, intra=False)
    print("output_intel", output_intel)
    if output_intra:
        graph_list_intra = phase_slurm_intra_2_DiGraphs(output_intra)
        graph_list_intel = []
        graph_combined = combine_graphs(graph_list_intra + graph_list_intel)
        visualize_graph(graph_combined)
    else:
        raise ValueError("No available nodes in Slurm to run the job.")
