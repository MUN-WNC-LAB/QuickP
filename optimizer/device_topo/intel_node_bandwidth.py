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
    start = iperf_data["start"]
    end = iperf_data["end"]
    intervals = iperf_data["intervals"]

    total_sent = end["sum_sent"]["bytes"] * 8 / 1_000_000  # Convert to Megabits
    total_received = end["sum_received"]["bytes"] * 8 / 1_000_000  # Convert to Megabits
    duration = end["sum_sent"]["seconds"]
    bandwidth_sent = total_sent / duration  # Mbps
    bandwidth_received = total_received / duration  # Mbps

    print(f"Total data sent: {total_sent:.2f} Mbits")
    print(f"Total data received: {total_received:.2f} Mbits")
    print(f"Duration: {duration:.2f} seconds")
    print(f"Bandwidth sent: {bandwidth_sent:.2f} Mbps")
    print(f"Bandwidth received: {bandwidth_received:.2f} Mbps")

    return {
        "total_sent_mbits": total_sent,
        "total_received_mbits": total_received,
        "duration_seconds": duration,
        "bandwidth_sent_mbps": bandwidth_sent,
        "bandwidth_received_mbps": bandwidth_received,
    }


def start_iperf_server(hostname, port, username, password):
    # Create an SSH client
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        # Connect to the remote server
        client.connect(hostname, username=username, password=password)

        # Start iperf3 server on the remote machine
        command = f"iperf3 -s -p {port}"
        stdin, stdout, stderr = client.exec_command(command)

        # Allow some time for the iperf3 server to start
        time.sleep(2)

        # Check if the server started successfully
        output = stdout.read().decode()
        errors = stderr.read().decode()
        if errors:
            print(f"Error starting iperf3 server: {errors}")
        else:
            print(f"iperf3 server started on {hostname}:{port}")

    except paramiko.SSHException as e:
        print(f"SSH connection failed: {e}")
    finally:
        client.close()


if __name__ == "__main__":
    port = 7575
    server_ip = "192.168.0.6"  # Replace with the server's IP address
    # Start iperf3 server on the remote machine
    start_iperf_server(server_ip, port, "root", "1314520")
    duration = 10  # Duration in seconds for the test
    run_iperf_client(server_ip, duration, port)
