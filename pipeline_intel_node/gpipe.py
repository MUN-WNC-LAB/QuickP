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
from torch.profiler import profile, ProfilerActivity
from pippy.SaveModule import save_checkpoint

sys.path.append("../")
from py_util import getArgs, printPipelineSplitInfo, getStdCifar10DataLoader, getStdModelForCifar10
# Initialize distributed environment
import torch.distributed as dist
from resnet import ResNet18
from vgg import vgg11

beginning_time = None
ending_time = None
# https://github.com/pytorch/PiPPy/blob/main/examples/profiling/mlp_profiling.py is the profiling example
# https://github.com/pytorch/PiPPy/blob/main/examples/checkpoint/toy_model.py is a good example found

args = getArgs()

# Figure out device to use
if torch.cuda.is_available():
    device = torch.device(f"cuda:{args.rank % torch.cuda.device_count()}")
else:
    device = torch.device("cpu")

# Create the model
mn = vgg11().to(device)
'''
it is important batch size is dividable by the number of images
'''
dataLoader = getStdCifar10DataLoader(num_workers=args.num_workers, batch_size=args.batch_size)
example_input = torch.randn(args.batch_size, 3, 32, 32, device=device)

# https://github.com/pytorch/PiPPy/blob/main/test/test_pipe.py
pipe = pipeline(mn, num_chunks=args.chunks, example_args=(example_input,),
                split_policy=split_into_equal_size(args.world_size))

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

# Define a loss function and optimizer
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(stage.submod.parameters(), lr=1e-3, momentum=0.9)

# Attach to a schedule
schedule = ScheduleGPipe(stage, args.chunks, loss_fn=loss_fn)

# An image is a 3*32*32 tensor
# A training set is a batch_size*3*32*32 tensor
mn.train()
with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
) as prof:
    for epoch in range(args.epochs):  # change to no. epochs
        save_checkpoint(
            pipe,
            checkpoint_dir=os.path.join("checkpoints", f"{epoch + 1}"),
            optimizer=optimizer,
        )
        for batch_idx, (inputs, targets) in enumerate(dataLoader, 0):
            x = inputs.to(device)
            y = targets.to(device)

            optimizer.zero_grad()

            # Run the pipeline with input `x`. Divide the batch into 4 micro-batches
            # and run them in parallel on the pipeline
            # rank == 0 => the first node
            if args.rank == 0:
                if batch_idx == 0:
                    beginning_time = datetime.datetime.now()
                schedule.step(x)
                ending_time = datetime.datetime.now()
                if batch_idx == dataLoader.__len__() - 1:
                    ending_time = datetime.datetime.now()
                    print("Rank", args.rank, " Beginning time ", beginning_time, " Ending time ", ending_time,
                          " Elapsed time ",
                          datetime.timedelta(seconds=ending_time.timestamp() - beginning_time.timestamp()))
            # the last node
            elif args.rank == args.world_size - 1:
                if batch_idx == 0:
                    beginning_time = datetime.datetime.now()
                losses = []
                output = schedule.step(target=y, losses=losses)
                # Take an optimization step
                optimizer.step()
                if batch_idx == dataLoader.__len__() - 1:
                    ending_time = datetime.datetime.now()
                    print("Rank", args.rank, " Beginning time ", beginning_time, " Ending time ", ending_time,
                          " Elapsed time ",
                          datetime.timedelta(seconds=ending_time.timestamp() - beginning_time.timestamp()))
            # nodes in the middle
            else:
                schedule.step()
prof.export_chrome_trace(
    f"{os.path.splitext(os.path.basename(__file__))[0]}_{args.rank}.json"
)

# Finish training
dist.barrier()
print(f"Rank {args.rank} completes")
