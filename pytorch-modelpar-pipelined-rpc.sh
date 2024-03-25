#!/bin/bash
#SBATCH --nodes 1
#SBATCH --gres=gpu:2 # request 2 GPUs
#SBATCH --tasks-per-node=1 
#SBATCH --cpus-per-task=1 # change this parameter to 2,4,6,... and increase "--num_workers" accordingly to see the effect on performance
#SBATCH --mem=8G      
#SBATCH --time=0:10:00
#SBATCH --output=%N-%j.out
#SBATCH --account=<your account>

module load python # Using Default Python version - Make sure to choose a version that suits your application
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install torch torchvision --no-index

# This is needed to initialize pytorch's RPC module, required for the Pipe class which we'll use for Pipeline Parallelism
export MASTER_ADDR=$(hostname)
export MASTER_PORT=34567
 
echo "starting training..."
time python pytorch-modelpar-pipelined-rpc.py --batch_size=512 --num_workers=0