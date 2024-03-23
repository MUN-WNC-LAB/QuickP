import os

import torch
import keras
import numpy as np
from torchgpipe import GPipe
import torch.nn as nn
from torch.utils.data import DataLoader

# our module must be nn.Sequential as GPipe will automatically split the module into partitions with consecutive layers
model = nn.Sequential(
    nn.Conv2d(1, 20, 5),
    nn.ReLU(),
    nn.Conv2d(20, 64, 5),
    nn.ReLU())

# balance's length is equal to the number of computing nodes
# module and sum of balance have the same length
model = GPipe(model, balance=[4], chunks=8)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
# Instantiate a torch loss function
loss_fn = torch.nn.CrossEntropyLoss()

# Create load up the MNIST dataset and put it in a torch DataLoader
# Prepare the training dataset.
batch_size = 32
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = np.reshape(x_train, (-1, 784)).astype("float32")
x_test = np.reshape(x_test, (-1, 784)).astype("float32")
y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)

# Reserve 10,000 samples for validation.
x_validation = x_train[-10000:]
y_validation = y_train[-10000:]
x_train = x_train[:-10000]
y_train = y_train[:-10000]

# Create torch Datasets
train_dataset = torch.utils.data.TensorDataset(
    torch.from_numpy(x_train), torch.from_numpy(y_train)
)
val_dataset = torch.utils.data.TensorDataset(
    torch.from_numpy(x_validation), torch.from_numpy(y_validation)
)

# Create DataLoaders for the Datasets
train_dataloader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True
)
val_dataloader = DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False
)

epochs = 3
for epoch in range(epochs):
    for step, (inputs, targets) in enumerate(train_dataloader):
        # cuda means
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

        # Log every 100 batches.
        if step % 100 == 0:
            print(
                f"Training loss (for 1 batch) at step {step}: {loss.detach().numpy():.4f}"
            )
            print(f"Seen so far: {(step + 1) * batch_size} samples")

