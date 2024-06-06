import subprocess

output_path = "device_intra_node_output.txt"
sh_path = "all_device_intra.sh"


def get_slurm_available_nodes():
    try:
        # Run sinfo command to get the number of idle nodes
        result = subprocess.run(['sinfo', '--noheader', '--states=idle', '--format=%D'],
                                stdout=subprocess.PIPE, text=True, check=True)

        # Parse the output to get the total number of available nodes
        available_nodes = sum(int(x) for x in result.stdout.split())

        return available_nodes
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running sinfo: {e}")
        return 0


def create_slurm_script(nodes, out_path):
    script_content = f"""#!/bin/bash
#SBATCH --job-name=All_Device_Intra_Node_Bandwidth
#SBATCH --time=00:30

#SBATCH --gpus={nodes}              
#SBATCH --gpus-per-node=1     
#SBATCH --nodes={nodes}           
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12    
#SBATCH --output={out_path}
#SBATCH --error=/dev/null     
#SBATCH --mem=1000            

### the command to run
srun python3 all_intra_node_topo_parallel.py
"""
    with open(sh_path, "w") as f:
        f.write(script_content)


def submit_slurm_script():
    try:
        subprocess.run(['sbatch', sh_path], check=True)
        print("Job submitted successfully.")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while submitting the job: {e}")


if __name__ == "__main__":
    nodes = get_slurm_available_nodes()
    print(f"Number of available nodes: {nodes}")

    if nodes > 0:
        create_slurm_script(nodes, output_path)
        submit_slurm_script()
    else:
        print("No available nodes to run the job.")
