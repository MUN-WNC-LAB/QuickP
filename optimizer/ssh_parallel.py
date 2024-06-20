import json
import os
from enum import Enum
from typing import Union

import paramiko
from concurrent.futures import ThreadPoolExecutor, as_completed
from slurm_util import get_server_ips
import warnings

warnings.filterwarnings("ignore")

script_dir = os.path.dirname(os.path.abspath(__file__))

servers = [

    {"ip": "192.168.0.66", "username": "root", "password": "1314520"},
    {"ip": "192.168.0.6", "username": "root", "password": "1314520"}
    # Add more servers as needed
]


class ParallelCommandType(Enum):
    INTRA_NODE = {"path": 'device_topo/intra_node_topo_parallel.py', "time": 30, "mem": '2000'}
    INTER_NODE = {"path": 'device_topo/intel_node_topo_parallel.py', "time": 30, "mem": '2000'}
    COMPUTING_COST = {"path": 'computing_graph/computing_cost_parallel.py', "time": 90, "mem": '3G'}
    IP_ADD_MAPPING = {"time": 30}


def graph_command_builder(command_type: ParallelCommandType, model_type: str) -> str:
    if command_type == ParallelCommandType.IP_ADD_MAPPING:
        return "python3 -c 'import socket; print(socket.gethostname())'"
    global script_dir
    path = os.path.join(script_dir, command_type.value['path'])
    command = f"python3 {path}"
    if command_type == ParallelCommandType.INTER_NODE:
        command += f" --dict '{json.dumps(get_server_ips())}'"
    elif command_type == ParallelCommandType.COMPUTING_COST:
        command += f" --model {model_type}"
    return command


def execute_command_on_server(server, command: str, timeout: int):
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(server["ip"], username=server["username"], password=server["password"])

    stdin, stdout, stderr = ssh.exec_command(command)
    stdout.channel.settimeout(timeout)
    stderr.channel.settimeout(timeout)
    output = stdout.read().decode()
    error = stderr.read().decode()

    ssh.close()
    '''
    if error:
        return f"Error from {server['ip']}: {error}"
    '''
    return output


def execute_commands_on_server(server, commands: list, timeout: int):
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(server["ip"], username=server["username"], password=server["password"])

    results = []
    for command in commands:
        stdin, stdout, stderr = ssh.exec_command(command)
        stdout.channel.settimeout(timeout)
        stderr.channel.settimeout(timeout)
        output = stdout.read().decode()
        error = stderr.read().decode()

        if error:
            results.append(f"Error from {server['ip']} for command '{command}': {error}")
        else:
            results.append(f"Output from {server['ip']} for command '{command}': {output}")

    ssh.close()
    return results


def execute_parallel(command_type: ParallelCommandType, model_type: str = None) -> dict:
    if model_type is None and command_type == ParallelCommandType.COMPUTING_COST:
        raise ValueError("model_type should not be None if getting COMPUTING_COST")
    results = {}
    with ThreadPoolExecutor(max_workers=len(servers)) as executor:
        exe_command = graph_command_builder(command_type, model_type)
        time_out = command_type.value['time']
        futures = {executor.submit(execute_command_on_server, server, exe_command, time_out): server for server in
                   servers}

        for future in as_completed(futures):
            server = futures[future]
            try:
                result = future.result()
                if command_type == ParallelCommandType.IP_ADD_MAPPING:
                    results[result] = server["ip"]
                else:
                    results[server["ip"]] = result
            except Exception as e:
                print(f"Error on {server['ip']}: {e}")

    return results


if __name__ == "__main__":
    print(execute_parallel(ParallelCommandType.IP_ADD_MAPPING))
    print(execute_parallel(ParallelCommandType.INTRA_NODE))
    print(execute_parallel(ParallelCommandType.INTER_NODE))
    print(execute_parallel(ParallelCommandType.COMPUTING_COST, "VGG16_tf"))
