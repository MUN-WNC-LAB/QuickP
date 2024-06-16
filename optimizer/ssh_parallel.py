import json
import os
from enum import Enum

import paramiko
from concurrent.futures import ThreadPoolExecutor, as_completed

from slurm_util import get_server_ips

script_dir = os.path.dirname(os.path.abspath(__file__))

servers = [
    {"hostname": "192.168.0.6", "username": "root", "password": "1314520"},
    {"hostname": "192.168.0.66", "username": "root", "password": "1314520"},
    # Add more servers as needed
]


class SLURM_RUN_CONF(Enum):
    INTRA_NODE = {"path": 'optimizer/device_topo/intra_node_topo_parallel.py', "time": '00:30', "mem": '2000'}
    INTER_NODE = {"path": 'optimizer/device_topo/intel_node_topo_parallel.py', "time": '00:30', "mem": '2000'}
    COMPUTING_COST = {"path": 'optimizer/computing_graph/computing_cost_parallel.py', "time": "1:30", "mem": '3G'}


def command_builder(command_type: SLURM_RUN_CONF, model_type: str):
    global script_dir
    path = os.path.join(script_dir, command_type.value['path'])
    command = f"python3 {path}"
    if command_type == SLURM_RUN_CONF.INTER_NODE:
        command += f" --dict '{json.dumps(get_server_ips())}'"
    elif command_type == SLURM_RUN_CONF.COMPUTING_COST:
        command += f" --model {model_type}"
    return command


def execute_command_on_server(server, command_type: SLURM_RUN_CONF, model_type: str):
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(server["hostname"], username=server["username"], password=server["password"])
    stdin, stdout, stderr = ssh.exec_command(command_builder(command_type, model_type))
    output = stdout.read().decode()
    ssh.close()
    return f"Output from {server['hostname']}: {output}"


def execute_parallel(command_type: SLURM_RUN_CONF, model_type: str):
    with ThreadPoolExecutor(max_workers=len(servers)) as executor:
        futures = {executor.submit(execute_command_on_server, server, command_type, model_type): server for server in
                   servers}
        for future in as_completed(futures):
            server = futures[future]
            try:
                result = future.result()
                print(result)
            except Exception as e:
                print(f"Error on {server['hostname']}: {e}")
