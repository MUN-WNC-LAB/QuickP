import argparse
import json
import os
import socket
import sys
import warnings

warnings.filterwarnings("ignore")
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
sys.path.append(project_root)
from optimizer.device_topo.intel_node_util import start_iperf_server, run_iperf_client
from optimizer.ssh_parallel import execute_parallel, ParallelCommandType
from optimizer.cluster_info import host_ip_mapping


def get_intel_node_topo(target_ip: str, from_node, to_node):
    # Start iperf3 server on the remote machine
    start_iperf_server(target_ip, "root", "1314520")
    duration = 3  # Duration in seconds for the test
    run_iperf_client(target_ip, duration, from_node, to_node)


if __name__ == "__main__":
    # Deserialize the JSON string to a dictionary
    local_hostname = socket.gethostname()
    other_servers = {key: value for key, value in host_ip_mapping.items() if key != local_hostname}
    for target_name, target_ip in other_servers.items():
        get_intel_node_topo(target_ip, from_node=local_hostname, to_node=target_name)
