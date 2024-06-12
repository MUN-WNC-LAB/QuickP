import argparse
import sys

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Intel Node Topology Test.')
    parser.add_argument('--target_ip', type=str, required=True, help='Target IP address for the test')
    parser.add_argument('--target_port', type=int, required=True, help='Target port for the test')

    args = parser.parse_args()

    get_intel_node_topo(args.target_ip, args.target_port)
