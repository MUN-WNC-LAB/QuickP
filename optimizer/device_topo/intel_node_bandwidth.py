# sudo apt-get install iperf3
import ast
import re
import subprocess
import json
import time
from typing import List, Any

import paramiko
from networkx import DiGraph
from paramiko.client import SSHClient

'''
Need update to RDMA but most hardware does not support. For now, it only supports TCP/UDP connections.
sudo apt-get install libmlx4-1 infiniband-diags rdmacm-utils ibutils ibverbs-utils perftest rdma-core
apt-get install tgt
https://developer.nvidia.com/networking/mlnx-ofed-eula?mtag=linux_sw_drivers&mrequest=downloads&mtype=ofed&mver=MLNX_OFED-24.04-0.6.6.0&mname=MLNX_OFED_LINUX-24.04-0.6.6.0-ubuntu22.04-x86_64.tgz
'''
default_port = 7100


# Sender side command example: iperf3 -c 192.168.0.6 -p 7575; -c state running as client end
# Receiver side command example: iperf3 -s -p 7575; -s state running as server end, -p specifies a self-defined port

# client end
def run_iperf_client(server_ip: str, duration: int, from_node: str, to_node: str):
    # Run the iperf3 client command
    command = ["iperf3", "-c", server_ip, "-t", str(duration), "-p", str(default_port), "-J"]  # '-J' for JSON output
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    if result.returncode != 0:
        print(f"Error running iperf3: {result.stderr.decode('utf-8')}")
        return None

    # Parse the JSON output
    output = result.stdout.decode('utf-8')
    iperf_data = json.loads(output)
    # Extract relevant information
    assert iperf_data is not None
    end = iperf_data["end"]
    assert end is not None
    duration = end["sum_received"]["seconds"]
    bandwidth_received = end["sum_received"]["bits_per_second"]
    band_dict = {
        "from": from_node,
        "to": to_node,
        "duration_seconds": duration,
        "bandwidth_received_GBs": bandwidth_received / (8 * 1_000_000_000),
    }
    print("Result: ", band_dict)


# server end
def start_iperf_server(hostname, username, password):
    global default_port

    def is_iperf3_active(client_obj: SSHClient):
        # Command to check what application is listening on the specified port
        global default_port
        cmd = f"sudo lsof -i :{default_port} | grep LISTEN"

        # Execute the command
        stdin, stdout, stderr = client_obj.exec_command(cmd)
        output = stdout.read().decode().strip()
        if output and "iperf3" in output:
            return True
        elif output and "iperf3" not in output:
            print(f"Wrong application is listening on port {default_port}. Run on new port")
            default_port += 1
            return False
        else:
            return False

    # Create an SSH client
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        # Connect to the remote server
        client.connect(hostname, username=username, password=password)
        if is_iperf3_active(client):
            return None
        # Start iperf3 server on the remote machine
        command = f"iperf3 -s -p {default_port}"
        client.exec_command(command)
        # give iperf3 two seconds to start
        time.sleep(2)
    except paramiko.SSHException as e:
        print(f"SSH connection failed: {e}")
    finally:
        client.close()


def slurm_output_intel_2_dict(slurm_output: str) -> list[dict]:
    def check_slurm_row_pattern(row: str):
        pattern = re.compile(r"Result:\s+(\{.*\})")
        match = pattern.match(row)
        if match:
            # ast.literal_eval convert string to dict
            bandwidths = ast.literal_eval(match.group(1))
            return bandwidths
        else:
            return None

    graph_list = []
    lines = slurm_output.splitlines()
    for line in lines:
        bandwidths_part = check_slurm_row_pattern(line)
        if bandwidths_part:
            graph_list.append(bandwidths_part)
    return graph_list


def update_intra_graph_with_intel(existing_graph: DiGraph, intel_dict_list: dict) -> None:
    # Function to get a key that includes a specific substring
    def find_nodes_with_substring(substring):
        return [node for node in existing_graph.nodes if substring in node]

    for band_dict in intel_dict_list:
        from_nodes = find_nodes_with_substring(band_dict["from"])
        to_nodes = find_nodes_with_substring(band_dict["to"])
        for from_node in from_nodes:
            for to_node in to_nodes:
                pass
