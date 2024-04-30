#!/bin/bash
#SBATCH --nodes=2             # minimum node number is 2
#SBATCH --gpus=2              # specifies the total number of GPUs required for an entire job across all nodes
#SBATCH --gres=gpu:1          # Generic resources required per node. We only have one GPU per node
#SBATCH --ntasks-per-node=1   # The number of tasks (GPUs) used in each node.
#SBATCH --ntasks=2            # total number of tasks (GPUs) used in this job across all nodes
#SBATCH --cpus-per-task=12    # accelerate data-loader workers to load data in parallel.
#SBATCH --mem=12G
#SBATCH --time=0-05:00
#SBATCH --output=%N-%j.out

export MASTER_ADDR=192.168.0.66 #Store the master node’s IP address in the MASTER_ADDR environment variable.
export MASTER_PORT=3456
export WORLD_SIZE=2
echo "r$SLURM_NODEID master: $SLURM_SUBMIT_DIR"
echo "r$SLURM_NODEID Launching python script"
# Suppose we run our training in 2 servers and each server/node has 4 GPUs. The world size is 4*2=8.
# The ranks for the processes will be [0, 1, 2, 3, 4, 5, 6, 7]. In each node, the local rank will be [0, 1, 2, 3]
echo "SLURM_NTASKS_PER_NODE: $SLURM_NTASKS_PER_NODE"
echo "SLURM_JOB_NUM_NODES: $SLURM_JOB_NUM_NODES"
echo "NODELIST="${SLURM_NODELIST}
srun python3.10 gpipe.py