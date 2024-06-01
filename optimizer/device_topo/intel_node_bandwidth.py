# sudo apt-get install iperf3
import subprocess
import json
import time

import paramiko


# Sender side command example: iperf3 -c 192.168.0.6 -p 7575; -c state running as client end
# Receiver side command example: iperf3 -s -p 7575; -s state running as server end, -p specifies a self-defined port
def run_iperf_client(server_ip: str, duration=10, port=5201):
    # Run the iperf3 client command
    command = ["iperf3", "-c", server_ip, "-t", str(duration), "-p", str(port), "-J"]  # '-J' for JSON output
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
    bandwidth_sent = end["sum_sent"]["bits_per_second"]  # Mbps
    bandwidth_received = end["sum_received"]["bits_per_second"]  # Mbps
    band_dict = {
        "duration_seconds": duration,
        "bandwidth_sent_mbps": bandwidth_sent,
        "bandwidth_received_mbps": bandwidth_received,
    }
    print(band_dict)
    return band_dict


def start_iperf_server(hostname, port: int, username, password):
    # Create an SSH client
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        # Connect to the remote server
        client.connect(hostname, username=username, password=password)

        # Start iperf3 server on the remote machine
        command = f"iperf3 -s -p {port}"
        stdin, stdout, stderr = client.exec_command(command)
    except paramiko.SSHException as e:
        print(f"SSH connection failed: {e}")
    finally:
        client.close()


if __name__ == "__main__":
    port = 7100
    server_ip = "192.168.0.6"  # Replace with the server's IP address
    # Start iperf3 server on the remote machine
    start_iperf_server(server_ip, port, "root", "1314520")
    duration = 10  # Duration in seconds for the test
    run_iperf_client(server_ip, duration, port)
