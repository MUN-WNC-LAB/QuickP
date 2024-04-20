import datetime
import os
import sys

import torch
from pippy import pipeline, split_into_equal_size, split_on_size_threshold
from pippy.IR import annotate_split_points, SplitPoint
from pippy.PipelineSchedule import PipelineScheduleGPipe
from pippy.PipelineStage import PipelineStage

sys.path.append("../")
from pyutil import getArgs, printPipelineSplitInfo, getStdCifar10DataLoader
# Initialize distributed environment
import torch.distributed as dist
from resnet import ResNet18

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
mn = ResNet18().to(device)

batch_size = 32
example_input = getStdCifar10DataLoader(num_workers=args.num_workers)
chunks = 4

pipe = pipeline(mn, chunks, example_args=(example_input,), split_policy=split_into_equal_size(args.world_size))

# make sure the stage number is equal to that of total devices
nstages = len(list(pipe.split_gm.children()))
assert nstages == args.world_size, f"nstages = {nstages} nranks = {args.world_size}"

# If there are two nodes, there can only be at most two stages
printPipelineSplitInfo(args.rank, pipe)

dist.init_process_group(backend=args.dist_backend, init_method=args.init_method, rank=args.rank,
                        world_size=args.world_size)

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
# rank == 0 => the first node
if args.rank == 0:
    beginning_time = datetime.datetime.now()
    schedule.step(x)
    ending_time = datetime.datetime.now()
    print("Rank", args.rank, " Beginning time ", beginning_time, " Ending time ", ending_time,
          " Elapsed time ", datetime.timedelta(seconds=ending_time.timestamp() - beginning_time.timestamp()))
# the last node
else:
    beginning_time = datetime.datetime.now()
    output = schedule.step()
    ending_time = datetime.datetime.now()
    print("Rank", args.rank, " Beginning time ", beginning_time, " Ending time ", ending_time,
          " Elapsed time ", datetime.timedelta(seconds=ending_time.timestamp() - beginning_time.timestamp()))

if args.rank == args.world_size - 1:
    # Run the original code and get the output for comparison
    reference_output = mn(x)
    # Compare numerics of pipeline and original model
    torch.testing.assert_close(output, reference_output)
    print(" Pipeline parallel model ran successfully! ".center(80, "*"))