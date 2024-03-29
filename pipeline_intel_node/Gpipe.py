# Copyright (c) Meta Platforms, Inc. and affiliates
# Minimal effort to run this code:
# $ torchrun --nproc-per-node 3 example.py
import datetime
import os
import sys

import torch
from pippy import pipeline
from pippy.IR import annotate_split_points, SplitPoint
from pippy.PipelineSchedule import PipelineScheduleGPipe
from pippy.PipelineStage import PipelineStage

sys.path.append("../")
from PyUtil import getArgs, printPipelineSplitInfo, init_distributed_group, ExePipeStep
# Initialize distributed environment
import torch.distributed as dist

in_dim = 512
layer_dims = [512, 1024, 256]
out_dim = 10
beginning_time = None
ending_time = None


def add_split_points(model, world_size):
    for i in range(1, world_size):
        # the name should correspond to the layer name in the model
        annotate_split_points(
            model, {f"layer{i}": SplitPoint.BEGINNING})


# Single layer definition
class MyNetworkBlock(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.lin = torch.nn.Linear(in_dim, out_dim)

    def forward(self, x):
        x = self.lin(x)
        x = torch.relu(x)
        return x


# Full model definition
class MyNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.num_layers = len(layer_dims)

        prev_dim = in_dim
        # Add layers one by one
        for i, dim in enumerate(layer_dims):
            # layer name must be written correctly. Thus, the split point can be added
            super().add_module(f"layer{i}", MyNetworkBlock(prev_dim, dim))
            prev_dim = dim

        # Final output layer (with OUT_DIM projection classes)
        self.output_proj = torch.nn.Linear(layer_dims[-1], out_dim)

    def forward(self, x):
        for i in range(self.num_layers):
            layer = getattr(self, f"layer{i}")
            x = layer(x)

        return self.output_proj(x)


# To run a distributed training job, we must launch the script in multiple
# different processes. We are using `torchrun` to do so in this example.
# `torchrun` defines two environment variables: `RANK` and `WORLD_SIZE`,
# which represent the index of this process within the set of processes and
# the total number of processes, respectively.
#
# To learn more about `torchrun`, see
# https://pytorch.org/docs/stable/elastic/run.html
args = getArgs()

# Figure out device to use
if torch.cuda.is_available():
    device = torch.device(f"cuda:{args.rank % torch.cuda.device_count()}")
else:
    device = torch.device("cpu")
# print("nodeID", int(os.environ.get("SLURM_NODEID")), "distributed mode: ", args.distributed, " from rank: ",
# args.rank, " world_size: ", args.world_size, " num_workers: ", args.num_workers)

# Create the model
mn = MyNetwork().to(device)

# Add the model split point
add_split_points(mn, args.world_size)

batch_size = 32
example_input = torch.randn(batch_size, in_dim, device=device)
chunks = 4

pipe = pipeline(mn, chunks, example_args=(example_input,))

# make sure the stage number is equal to that of total devices
nstages = len(list(pipe.split_gm.children()))
assert nstages == args.world_size, f"nstages = {nstages} nranks = {args.world_size}"

# If there are two nodes, there can only be at most two stages
if args.rank == 0:
    print(" pipe ".center(80, "*"))
    print(pipe)
    print(" stage 0 ".center(80, "*"))
    print(pipe.split_gm.submod_0)
    print(" stage 1 ".center(80, "*"))
    print(pipe.split_gm.submod_1)

# printPipelineSplitInfo(args.rank, pipe)

init_distributed_group(args)

# Pipeline stage is our main pipeline runtime. It takes in the pipe object,
# the rank of this process, and the device.
# Put different stages on different devices
stage = PipelineStage(pipe, args.rank, device)

# Attach to a schedule
schedule = PipelineScheduleGPipe(stage, chunks)

# Input data
x = torch.randn(batch_size, in_dim, device=device)

# Run the pipeline with input `x`. Divide the batch into 4 micro-batches
# and run them in parallel on the pipeline
# This step triggers task 1: Segmentation fault (core dumped)
# Need to make sure the later node cannot run before the previous one
output = ExePipeStep(args, schedule, x)

if args.rank == args.world_size - 1:
    # Run the original code and get the output for comparison
    reference_output = mn(x)
    # Compare numerics of pipeline and original model
    torch.testing.assert_close(output, reference_output)
    print(" Pipeline parallel model ran successfully! ".center(80, "*"))