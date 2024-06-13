import argparse
import json
import socket
import sys

from optimizer.device_topo.workflow_device_topo import run_srun_command
from slurm_util import get_server_ips, get_slurm_available_nodes

sys.path.append("../../")
from optimizer.device_topo.intel_node_bandwidth import start_iperf_server, run_iperf_client


import warnings

warnings.filterwarnings("ignore")


def get_intel_node_topo(target_ip: str, from_node, to_node):
    # Start iperf3 server on the remote machine
    start_iperf_server(target_ip, "root", "1314520")
    duration = 3  # Duration in seconds for the test
    run_iperf_client(target_ip, duration, from_node, to_node)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some dictionary.')
    parser.add_argument('--dict', type=str, required=True,
                        help='all available servers in Slurm and their ips as a JSON string')

    args = parser.parse_args()

    # Deserialize the JSON string to a dictionary
    servers = json.loads(args.dict)

    all_results = {}
    local_hostname = socket.gethostname()
    other_servers = {key: value for key, value in servers.items() if key != local_hostname}
    for target_name, target_ip in other_servers.items():
        get_intel_node_topo(target_ip, from_node=local_hostname, to_node=target_name)
