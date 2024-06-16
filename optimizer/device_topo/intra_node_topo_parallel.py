"""
This file is called by a Slurm script .sh file
"""
import os
import sys

import warnings

warnings.filterwarnings("ignore")

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
sys.path.append(project_root)
from optimizer.device_topo.intra_node_util import get_device_bandwidth
from optimizer.model.graph import DeviceGraph


def get_intra_node_topo() -> DeviceGraph:
    print("fuck")
    G = DeviceGraph()
    bandwidths, devices = get_device_bandwidth()
    print("bandwidths: ", bandwidths, "devices: ", devices)
    return "fuck 3"


if __name__ == "__main__":
    print("fuck2")
    get_intra_node_topo()
