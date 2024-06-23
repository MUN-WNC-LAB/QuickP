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
from optimizer.cluster_info import ServerInfo


def get_intel_node_topo(info: ServerInfo):
    # Start iperf3 server on the remote machine
    start_iperf_server(info.ip, info.username, info.password)
    duration = 3  # Duration in seconds for the test
    run_iperf_client(info.ip, duration)


if __name__ == "__main__":
    # Deserialize the JSON string to a dictionary
    parser = argparse.ArgumentParser(description='Process some dictionary.')
    parser.add_argument('--server_list', type=str, required=True,
                        help='all available servers in the current setting in the format of Service')

    args = parser.parse_args()
    # Convert the JSON string to a list of dictionaries
    server_list_dicts = json.loads(args.server_list)
    # Convert the list of dictionaries to a list of ServerInfo objects
    server_list = [ServerInfo(**server_dict) for server_dict in server_list_dicts]
    local_hostname = socket.gethostname()
    other_servers: list[ServerInfo] = [s_info for s_info in server_list if s_info.hostname != local_hostname]
    for s_info in other_servers:
        get_intel_node_topo(s_info)
