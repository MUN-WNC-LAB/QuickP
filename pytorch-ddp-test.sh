#!/bin/bash
#SBATCH --nodes 2
#SBATCH --gres=gpu:1          # Generic resources required per node. We only have one GPU per node
#SBATCH --ntasks-per-node=1   # Request 1 process per GPU. You will get 1 CPU per process by default. Request more CPUs with the "cpus-per-task" parameter to enable multiple data-loader workers to load data in parallel.
#SBATCH --mem=8G
#SBATCH --time=0-03:00
#SBATCH --output=%N-%j.out
#module load python/3.10 # Using Default Python version - Make sure to choose a version that suits your application
# the environment file is etc/environment-modules/modulespath; /usr/share/Modules/modulefiles
#source /etc/profile.d/modules.sh

# use sbatch pytorch-ddp-test.sh to execute
export TORCH_NCCL_BLOCKING_WAIT=1  #Set this environment variable if you wish to use the NCCL backend for inter-GPU communication.
export MASTER_ADDR=192.168.0.66 #Store the master node’s IP address in the MASTER_ADDR environment variable.

echo "r$SLURM_NODEID master: $SLURM_SUBMIT_DIR"
echo "r$SLURM_NODEID Launching python script"
# Suppose we run our training in 2 servers and each server/node has 4 GPUs. The world size is 4*2=8.
# The ranks for the processes will be [0, 1, 2, 3, 4, 5, 6, 7]. In each node, the local rank will be [0, 1, 2, 3]

# The $((SLURM_NTASKS_PER_NODE * SLURM_JOB_NUM_NODES)) variable tells the script how many processes are available for this execution. “srun” executes the script <tasks-per-node * nodes> times
# SLURM_NTASKS_PER_NODE correspond to --ntasks-per-node;
# SLURM_JOB_NUM_NODES is the total number of nodes in the job's resource allocation.
# Need to set GresTypes=gpu,mps in slurm.conf
echo "SLURM_NTASKS_PER_NODE: $SLURM_NTASKS_PER_NODE"
echo "SLURM_JOB_NUM_NODES: $SLURM_JOB_NUM_NODES"
srun python3.10 PyTorchDP.py --init_method tcp://$MASTER_ADDR:3456 --world_size $((SLURM_NTASKS_PER_NODE * SLURM_JOB_NUM_NODES))  --batch_size 256