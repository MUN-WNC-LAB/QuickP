import datetime
import os
import sys

import torch
import torchvision

'''
must download the latest version of Pippy form https://github.com/pytorch/PiPPy/tree/main
good example: https://github.com/pytorch/PiPPy/blob/main/examples/checkpoint/toy_model.py
'''
from pippy.PipelineSchedule import ScheduleGPipe
from pippy import pipeline, split_into_equal_size, split_on_size_threshold
from pippy.IR import annotate_split_points, SplitPoint
from pippy.PipelineStage import PipelineStage
from torchvision.transforms import transforms

sys.path.append("../")
from py_util import getArgs, printPipelineSplitInfo, getStdCifar10DataLoader, getStdModelForCifar10
# Initialize distributed environment
import torch.distributed as dist
from resnet import ResNet18
from vgg import vgg11

beginning_time = None
ending_time = None


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

# Create the model
mn = vgg11().to(device)

dataLoader = getStdCifar10DataLoader(num_workers=args.num_workers, batch_size=args.batch_size)
example_input = torch.randn(args.batch_size, 3, 32, 32, device=device)
example_output = torch.randn(args.batch_size, 10, device=device)

# An image is a 3*32*32 tensor
# A training set is a batch_size*3*32*32 tensor
for batch_idx, (inputs, targets) in enumerate(dataLoader, 0):
    if batch_idx == 1:
        break
    x = inputs.to(device)
    y = targets.to(device)
    print(x.shape)
# https://github.com/pytorch/PiPPy/blob/main/test/test_pipe.py
pipe = pipeline(mn, num_chunks=args.chunks, example_args=(example_input,), split_policy=split_into_equal_size(args.world_size))

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

# Define a loss function
loss_fn = torch.nn.MSELoss(reduction="sum")

# Attach to a schedule
schedule = ScheduleGPipe(stage, args.chunks, loss_fn=loss_fn)

# Run the pipeline with input `x`. Divide the batch into 4 micro-batches
# and run them in parallel on the pipeline
# rank == 0 => the first node
if args.rank == 0:
    beginning_time = datetime.datetime.now()
    schedule.step(x)
    ending_time = datetime.datetime.now()
    print("Rank", args.rank, " Beginning time ", beginning_time, " Ending time ", ending_time,
          " Elapsed time ", datetime.timedelta(seconds=ending_time.timestamp() - beginning_time.timestamp()))
# the last node
elif args.rank == args.world_size - 1:
    beginning_time = datetime.datetime.now()
    losses = []
    output = schedule.step()
    # output = schedule.step(target=y, losses=losses)
    ending_time = datetime.datetime.now()
    print("Rank", args.rank, " Beginning time ", beginning_time, " Ending time ", ending_time,
          " Elapsed time ", datetime.timedelta(seconds=ending_time.timestamp() - beginning_time.timestamp()))
# nodes in the middle
else:
    beginning_time = datetime.datetime.now()
    schedule.step()
    ending_time = datetime.datetime.now()
    print("Rank", args.rank, " Beginning time ", beginning_time, " Ending time ", ending_time,
          " Elapsed time ", datetime.timedelta(seconds=ending_time.timestamp() - beginning_time.timestamp()))

# Finish training
if args.rank == args.world_size - 1:
    # Run the original code and get the output for comparison
    reference_output = mn(x)
    # Compare numerics of pipeline and original model
    torch.testing.assert_close(output, reference_output)
    print(" Pipeline parallel model ran successfully! ".center(80, "*"))
