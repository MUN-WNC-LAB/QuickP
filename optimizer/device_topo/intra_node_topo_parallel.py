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


def get_intra_node_topo() -> None:
    bandwidths, devices = get_device_bandwidth()
    print("bandwidths: ", bandwidths, "devices: ", devices)


if __name__ == "__main__":
    get_intra_node_topo()
