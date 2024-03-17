import os
import time
import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader

import torch.distributed as dist
import torch.utils.data.distributed

import argparse

# Store argument values
parser = argparse.ArgumentParser(description='cifar10 classification models, distributed data parallel test')
parser.add_argument('--lr', default=0.1, help='')
parser.add_argument('--batch_size', type=int, default=768, help='')
parser.add_argument('--max_epochs', type=int, default=2, help='')
parser.add_argument('--num_workers', type=int, default=2, help='')

parser.add_argument('--init_method', default='tcp://192.168.0.66:3456', type=str, help='')
parser.add_argument('--dist-backend', default='nccl', type=str, help='')
parser.add_argument('--world_size', default=1, type=int, help='')
parser.add_argument('--distributed', action='store_true', help='')


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train(epoch, net, criterion, optimizer, train_loader, train_rank):
    train_loss = 0
    correct = 0
    total = 0

    epoch_start = time.time()

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        start = time.time()

        inputs = inputs.cuda()
        targets = targets.cuda()
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        acc = 100 * correct / total

        batch_time = time.time() - start

        elapse_time = time.time() - epoch_start
        elapse_time = datetime.timedelta(seconds=elapse_time)
        print("From Rank: {}, Training time {}".format(train_rank, elapse_time))


def main():
    print("Starting...")

    args = parser.parse_args()

    ngpus_per_node = torch.cuda.device_count()

    """ This next line is the key to getting DistributedDataParallel working on SLURM:
		SLURM_NODEID is 0 or 1 in this example, SLURM_LOCALID is the id of the 
 		current process inside a node and is also 0 or 1 in this example."""

    local_rank = int(os.environ.get("SLURM_LOCALID"))
    rank = int(os.environ.get("SLURM_NODEID")) * ngpus_per_node + local_rank

    current_device = local_rank

    torch.cuda.set_device(current_device)

    """ this block initializes a process group and initiate communications
		between all processes running on all nodes """

    print('From Rank: {}, ==> Initializing Process Group...'.format(rank))
    # init the process group
    # Creation of DistributedDataParallel obj requires that torch.distributed (dist) to be initialized, by calling
    # Backend includes mpi, gloo(CPU), nccl(GPU), and ucc. https://pytorch.org/docs/stable/distributed.html
    dist.init_process_group(backend=args.dist_backend, init_method=args.init_method, world_size=args.world_size,
                            rank=rank)
    print("process group ready!")

    print('From Rank: {}, ==> Making model..'.format(rank))

    net = Net()

    net.cuda()
    net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[current_device])

    print('From Rank: {}, ==> Preparing data..'.format(rank))

    transform_train = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    dataset_train = CIFAR10(root='./data', train=True, download=True, transform=transform_train)

    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset_train)
    train_loader = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=(train_sampler is None),
                              num_workers=args.num_workers, sampler=train_sampler)

    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)

    for epoch in range(args.max_epochs):
        train_sampler.set_epoch(epoch)

        train(epoch, net, criterion, optimizer, train_loader, rank)


if __name__ == '__main__':
    main()
