import numpy as np
import tensorflow as tf
from keras.src.losses import BinaryCrossentropy

# Define the number of classes and batch size
num_classes = 5
batch_size = 4

# Example of predictions (softmax probabilities) with shape (batch_size, num_classes)
# These represent the predicted probabilities for 5 classes for each of the 4 examples
predictions = tf.constant([
    [0.1, 0.2, 0.3, 0.15, 0.25],  # Probabilities for the first example
    [0.05, 0.1, 0.2, 0.5, 0.15],  # Probabilities for the second example
    [0.25, 0.25, 0.25, 0.15, 0.1],  # Probabilities for the third example
    [0.05, 0.1, 0.15, 0.2, 0.5]    # Probabilities for the fourth example
], dtype=tf.float32)  # Shape: (4, 5)

# Example of real labels with shape (batch_size,)
# These represent the true class labels for the 4 examples
real_labels = tf.constant([2, 3, 1, 4], dtype=tf.int32)  # Shape: (4,)

print(real_labels.shape)

# Define the loss function (since the output is probabilities, we use from_logits=False)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)

# Compute the loss between the predictions and real labels
loss = loss_fn(real_labels, predictions)

# Print the loss value
print(f"Loss: {loss.numpy()}")


# True labels (y_true): binary labels, e.g., 1 for positive, 0 for negative
y_true = np.array([1, 0, 1, 0])

# Predicted probabilities (y_pred): these are outputs from the model (sigmoid layer)
y_pred = np.array([0.9, 0.2, 0.8, 0.1])

# Initialize the binary crossentropy loss function
bce = BinaryCrossentropy()

# Calculate the binary crossentropy loss manually for better understanding
# Binary crossentropy formula: - (y_true * log(y_pred) + (1 - y_true) * log(1 - y_pred))
# Shape: (batch_size,) or (batch_size, 1)
loss = bce(y_true, y_pred).numpy()

# Display the loss value
print(f"Binary Crossentropy Loss: {loss}")