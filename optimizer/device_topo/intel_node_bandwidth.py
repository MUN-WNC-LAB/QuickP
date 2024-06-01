# sudo apt-get install iperf3
import subprocess
import json

# Sender side command example: iperf3 -c 192.168.0.6 -p 7575; -c state running as client end
# Receiver side command example: iperf3 -s -p 7575; -s state running as server end, -p specifies a self-defined port
def run_iperf(server_ip, duration=10):
    # Run the iperf3 client command
    command = ["iperf3", "-c", server_ip, "-t", str(duration), "-J"]  # '-J' for JSON output
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


if __name__ == "__main__":
    server_ip = "192.168.0.6"  # Replace with the server's IP address
    duration = 10  # Duration in seconds for the test
    run_iperf(server_ip, duration)
