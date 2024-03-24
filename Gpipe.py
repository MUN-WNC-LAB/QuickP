import argparse
import datetime
import os
import time

import torchvision
from torchvision import transforms

# This guide can only be run with the torch backend. must write when using both keras and pytorch
# sudo apt install python3-packaging
os.environ["KERAS_BACKEND"] = "torch"

import torch
import keras
import numpy as np
from torchgpipe import GPipe
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.distributed as dist
import torch.nn.functional as F

# Store argument values
parser = argparse.ArgumentParser(description='cifar10 classification models, distributed data parallel test')
# always 1 in our platform
parser.add_argument('-g', '--gpus', default=1, type=int, help='number of gpus per node')
parser.add_argument('--max_epochs', type=int, default=2, help='')
'''
Change the following accordingly
'''
parser.add_argument('--num_workers', type=int, default=1, help='')
parser.add_argument('--init_method', default='tcp://192.168.0.66:3456', type=str, help='')
parser.add_argument('--dist-backend', default='nccl', type=str, help='')
parser.add_argument('--world_size', default=1, type=int, help='')
args = parser.parse_args()
# Update world size
args.world_size = args.gpus * args.num_workers

# our module must be nn.Sequential as GPipe will automatically split the module into partitions with consecutive layers
# The previous layer's out_channels should match the next layer's in_channels
# https://stackoverflow.com/questions/68606661/what-is-difference-between-nn-module-and-nn-sequential;
# using nn-module or nn-sequential
model = nn.Sequential(
    nn.Conv2d(3, 6, 5),
    nn.MaxPool2d(2, 2),
    nn.Conv2d(6, 16, 5),
    nn.MaxPool2d(2, 2),
    nn.Flatten(),
    nn.Linear(16 * 5 * 5, 120),
    nn.Linear(120, 84),
    # the out_features number of the last layer should match the class number
    nn.Linear(84, 10))
model.cuda()

# balance's length is equal to the number of computing nodes
# model layers and sum of balance have the same length
# balance determines the number of layers in each node
# chunks means the number of micro-batches
model = GPipe(model, balance=[8], chunks=8)
# a Loss function and optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = torch.nn.CrossEntropyLoss().cuda()
batch_size = 64

# Creation of Distributed Data Parallel obj requires that torch.distributed (dist.init_process_group) to be initialized
# Backend includes mpi, gloo(CPU), nccl(GPU), and ucc. https://pytorch.org/docs/stable/distributed.html
# rank is the GPU index
# dist.init_process_group(backend=args.dist_backend, init_method=args.init_method, world_size=args.world_size, rank=rank)
# Wrap the model
# model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[current_device])
# train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=args.world_size)
'''
If Use keras dataset instead of torchvision 
https://keras.io/guides/writing_a_custom_training_loop_in_torch/ 
'''
# Data loading code for CiFar10
transform_train = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform_train, download=True)
# sampler=train_sampler; if sampler is defined, set the shuffle to false
train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True,
                                               pin_memory=True, num_workers=1)
# start training
epochs = 3
epoch_start = time.time()
for epoch in range(epochs):
    for step, (inputs, targets) in enumerate(train_dataloader):
        # cuda means Cross-GPU operations
        inputs = inputs.cuda()
        targets = targets.cuda()

        # Forward pass
        output = model(inputs)
        loss = loss_fn(output, targets)

        # Backward pass
        model.zero_grad()
        loss.backward()

        # Optimizer variable updates
        optimizer.step()

        elapse_time = datetime.timedelta(seconds=time.time() - epoch_start)
        print('From Node ID {}'.format(int(os.environ.get("SLURM_NODEID"))), f"Seen so far: {(step + 1) * batch_size} samples", "Training time {}".format(elapse_time))
