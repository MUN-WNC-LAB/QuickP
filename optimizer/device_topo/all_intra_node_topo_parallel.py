"""
This file is called by a Slurm script .sh file
"""
import os
import sys

sys.path.append("../../")
from optimizer.device_topo.intra_node_bandwidth import get_device_bandwidth
from optimizer.model.graph import DeviceGraph

import warnings

warnings.filterwarnings("ignore")


def get_intra_node_topo() -> DeviceGraph:
    G = DeviceGraph()
    bandwidths, devices = get_device_bandwidth()
    print("bandwidths: ", bandwidths, "devices: ", devices)
    return G


if __name__ == "__main__":
    get_intra_node_topo()
