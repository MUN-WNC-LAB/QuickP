from optimizer.device_topo.intel_node_bandwidth import start_iperf_server, run_iperf_client
from optimizer.device_topo.intra_node_bandwidth import get_device_bandwidth
from optimizer.model.graph import DeviceGraph


def init_topo():
    G = DeviceGraph()
    bandwidths = get_device_bandwidth()
    G.a

    port = 7100
    server_ip = "192.168.0.6"  # Replace with the server's IP address
    # Start iperf3 server on the remote machine
    start_iperf_server(server_ip, port, "root", "1314520")
    duration = 10  # Duration in seconds for the test
    run_iperf_client(server_ip, duration, port)
