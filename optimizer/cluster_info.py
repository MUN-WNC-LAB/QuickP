from optimizer.ssh_parallel import execute_parallel, ParallelCommandType, ServerInfo

SERVERS_INFO = []
# Creating instances of the Server class
server1 = ServerInfo("192.168.0.66", "root", "1314520")
server2 = ServerInfo("192.168.0.6", "root", "1314520")
servers = [server1, server2]
host_ip_mapping = execute_parallel(servers, ParallelCommandType.IP_ADD_MAPPING)
for server in servers:
    if server.ip in host_ip_mapping.keys():
        server.hostname = host_ip_mapping[server.ip]


def add_server(ip, username, password):
    server = ServerInfo(ip, username, password)
    SERVERS_INFO.append(server)
    return server


def execute_commands():
    host_ip_mapping = execute_parallel(SERVERS_INFO, ParallelCommandType.IP_ADD_MAPPING)
    for server in SERVERS_INFO:
        if server.ip in host_ip_mapping.keys():
            server.hostname = host_ip_mapping[server.ip]


def get_servers_info():
    return SERVERS_INFO
