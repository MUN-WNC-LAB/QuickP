"""
This file is called by a Slurm script .sh file
"""
import os
import sys

from optimizer.device_topo.intra_node_util import get_device_bandwidth
from optimizer.model.graph import DeviceGraph

import warnings

warnings.filterwarnings("ignore")


def get_intra_node_topo() -> DeviceGraph:
    print("fuck")
    G = DeviceGraph()
    bandwidths, devices = get_device_bandwidth()
    print("bandwidths: ", bandwidths, "devices: ", devices)
    return G


if __name__ == "__main__":
    get_intra_node_topo()
