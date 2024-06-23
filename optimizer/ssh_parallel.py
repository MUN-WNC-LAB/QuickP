import json
import os
import sys
from enum import Enum

import paramiko
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings

warnings.filterwarnings("ignore")

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.append(project_root)


class ServerInfo:
    def __init__(self, ip, username, password, hostname=None):
        self.ip = ip
        self.username = username
        self.password = password
        self.hostname = hostname

    def __repr__(self):
        return f"Server(ip='{self.ip}', hostname='{self.hostname}', username='{self.username}', password='{self.password}')"

    def to_json(self):
        # Convert the instance to a dictionary
        data = {
            'ip': self.ip,
            'username': self.username,
            'password': self.password,
            'hostname': self.hostname
        }
        # Serialize the dictionary to a JSON string
        return json.dumps(data)

    @classmethod
    def from_json(cls, json_str):
        # Parse the JSON string to a dictionary
        data = json.loads(json_str)
        # Create an instance of ServerInfo using the parsed data
        return cls(ip=data['ip'], username=data['username'], password=data['password'], hostname=data.get('hostname'))


class ParallelCommandType(Enum):
    INTRA_NODE = {"path": 'device_topo/intra_node_topo_parallel.py', "time": 30, "mem": '2000'}
    INTER_NODE = {"path": 'device_topo/intel_node_topo_parallel.py', "time": 30, "mem": '2000'}
    COMPUTING_COST = {"path": 'computing_graph/computing_cost_parallel.py', "time": 90, "mem": '3G'}
    IP_ADD_MAPPING = {"time": 30}


def graph_command_builder(command_type: ParallelCommandType, model_type: str, server_list: list[ServerInfo]) -> str:
    if command_type == ParallelCommandType.IP_ADD_MAPPING:
        return "python3 -c 'import socket; print(socket.gethostname())'"
    global script_dir
    path = os.path.join(script_dir, command_type.value['path'])
    command = f"python3 {path}"
    if command_type == ParallelCommandType.COMPUTING_COST:
        command += f" --model {model_type}"
    elif command_type == ParallelCommandType.INTER_NODE:
        server_list_dicts = [server.__dict__ for server in server_list]
        server_list_json = json.dumps(server_list_dicts)
        command += f" --server_list '{server_list_json}'"
    return command


def execute_command_on_server(server: ServerInfo, command: str, timeout: int):
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(server.ip, username=server.username, password=server.password)

    stdin, stdout, stderr = ssh.exec_command(command)
    stdout.channel.settimeout(timeout)
    stderr.channel.settimeout(timeout)
    output = stdout.read().decode()
    error = stderr.read().decode()

    ssh.close()

    if output:
        return output

    return f"Error from {server.ip}: {error}"


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


def execute_parallel(server_list, command_type: ParallelCommandType, model_type: str = None) -> dict:
    if model_type is None and command_type == ParallelCommandType.COMPUTING_COST:
        raise ValueError("model_type should not be None if getting COMPUTING_COST")
    results = {}
    with ThreadPoolExecutor(max_workers=len(server_list)) as executor:
        exe_command = graph_command_builder(command_type, model_type)
        time_out = command_type.value['time']
        futures = {executor.submit(execute_command_on_server, server, exe_command, time_out): server for server in
                   server_list}

        for future in as_completed(futures):
            server = futures[future]
            try:
                result = future.result()
                results[server.ip] = result.strip()
            except Exception as e:
                print(f"Error on {server.ip}: {e}")

    return results
