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
from optimizer.host_ip import host_ip_mapping
from optimizer.cluster_info import ServerInfo, servers


def get_intel_node_topo(info: ServerInfo):
    # Start iperf3 server on the remote machine
    start_iperf_server(info.ip, info.username, info.password)
    duration = 3  # Duration in seconds for the test
    run_iperf_client(info.ip, duration)


if __name__ == "__main__":
    # Deserialize the JSON string to a dictionary
    local_hostname = socket.gethostname()
    local_ip = host_ip_mapping.get(local_hostname)
    other_servers: list[ServerInfo] = [s_info for s_info in servers if s_info.ip != local_ip]
    for s_info in other_servers:
        get_intel_node_topo(s_info)
