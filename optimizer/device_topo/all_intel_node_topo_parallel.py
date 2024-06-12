import argparse
import socket
import sys

from optimizer.device_topo.workflow_device_topo import run_srun_command
from slurm_util import get_server_ips, get_slurm_available_nodes

sys.path.append("../../")
from optimizer.device_topo.intel_node_bandwidth import start_iperf_server, run_iperf_client
from optimizer.device_topo.intra_node_bandwidth import get_device_bandwidth
from optimizer.model.graph import DeviceGraph

import warnings

warnings.filterwarnings("ignore")


def get_intel_node_topo(target_ip: str, target_port: int):
    # Start iperf3 server on the remote machine
    start_iperf_server(target_ip, target_port, "root", "1314520")
    duration = 3  # Duration in seconds for the test
    run_iperf_client(target_ip, duration, target_port)


servers = get_server_ips()
nodes = get_slurm_available_nodes()

all_results = {}
local_hostname = socket.gethostname()
other_servers = {key: value for key, value in servers.items() if key != local_hostname}
print(servers, local_hostname, other_servers)
