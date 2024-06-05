import paramiko
import scp


def execute_remote_script(hostname, username, password, local_script, remote_script_path):
    # Set up the SSH client
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    # Connect to the remote server
    ssh.connect(hostname, username=username, password=password)

    # Transfer the script
    with scp.SCPClient(ssh.get_transport()) as scp_client:
        scp_client.put(local_script, remote_script_path)

    # Execute the script
    stdin, stdout, stderr = ssh.exec_command(f'python3 {remote_script_path}')

    # Print the output
    print(stdout.read().decode())
    print(stderr.read().decode())

    # Close the connection
    ssh.close()


# Example usage
if __name__ == "__main__":
    hostname = 'remote_host'
    username = 'username'
    password = 'password'  # It's recommended to use key-based authentication for better security
    local_script = 'device_info.py'
    remote_script_path = '/path/to/destination/device_info.py'

    execute_remote_script(hostname, username, password, local_script, remote_script_path)