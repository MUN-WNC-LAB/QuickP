from optimizer.ssh_parallel import execute_parallel, ParallelCommandType


class ServerInfo:
    def __init__(self, ip, username, password):
        self.ip = ip
        self.username = username
        self.password = password

    def __repr__(self):
        return f"Server(ip='{self.ip}', username='{self.username}', password='{self.password}')"


# Creating instances of the Server class
server1 = ServerInfo("192.168.0.66", "root", "1314520")
server2 = ServerInfo("192.168.0.6", "root", "1314520")
servers = [server1, server2]
host_ip_mapping = execute_parallel(ParallelCommandType.IP_ADD_MAPPING)
