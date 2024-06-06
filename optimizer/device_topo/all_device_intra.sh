#!/bin/bash
#SBATCH --job-name=All_Device_Intra_Node_Bandwidth
#SBATCH --time=00:30

### e.g. request 2 nodes with 1 gpu each, totally 2 gpus (WORLD_SIZE==2)
#SBATCH --gpus=2              # specifies the total number of GPUs required for an entire job across all nodes
#SBATCH --gpus-per-node=1     # GPU number in each node; should equal to ntasks-per-node
#SBATCH --nodes=ALL           # Request all available nodes
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12    # accelerate Pytorch data loader
#SBATCH --output=device_intra_node_output.txt
#SBATCH --error=/dev/null     # discard warning
#SBATCH --mem=1000            #A memory size specification of zero is treated as a special case and grants the job access to all of the memory on each node.

### the command to run
srun python3 all_intra_node_topo_parallel.py