from optimizer.ssh_parallel import execute_parallel, ParallelCommandType, ServerInfo
# Creating instances of the Server class
server1 = ServerInfo("192.168.0.66", "root", "1314520")
server2 = ServerInfo("192.168.0.6", "root", "1314520")
servers = [server1, server2]
host_ip_mapping = execute_parallel(servers, ParallelCommandType.IP_ADD_MAPPING)
for server in servers:
    if server.ip in host_ip_mapping.keys():
        server.hostname = host_ip_mapping[server.ip]
