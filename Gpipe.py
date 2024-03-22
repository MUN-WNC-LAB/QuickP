from torchgpipe import GPipe
import torch.nn as nn
# our module must be nn.Sequential as GPipe will automatically split the module into partitions with consecutive layers
model = nn.Sequential(
    nn.Conv2d(1,20,5),
    nn.ReLU(),
    nn.Conv2d(20,64,5),
    nn.ReLU())
# balance's length is equal to the number of computing nodes
# module and sum of balance have the same length
model = GPipe(model, balance=[4], chunks=8)
for input in data_loader:
    output = model(input)