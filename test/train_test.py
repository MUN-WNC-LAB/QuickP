from keras import Sequential
from keras.src.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

from DNN_model_tf.vgg_tf import VGG16_tf
from tf_util import compile_model, getCifar, train_model
import tensorflow as tf
import datetime
import torch

from DNN_model_tf.vgg_tf import VGG16_tf
from tf_util import getCifar


def create_cifar10_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(10, activation='softmax')
    ])
    return model


def f1():
    model = create_cifar10_model()
    compile_model(model)

    class PrintBatchMetrics(tf.keras.callbacks.Callback):
        def on_train_batch_end(self, batch, logs=None):
            print(f'Batch {batch}, Loss: {logs["loss"]}, Accuracy: {logs["acc"]}')

    # Create a TensorBoard callback
    # profiling: https://www.tensorflow.org/tensorboard/tensorboard_profiling_keras
    # python3 -m tensorboard.main --logdir=~/my/training/dir
    tboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs", histogram_freq=1,
                                                     # a range of batches
                                                     profile_batch='50,55')
    print_batch_metrics = PrintBatchMetrics()
    (x_train, y_train), (x_test, y_test) = getCifar()
    train_model(model, x_train, y_train, x_test, y_test, [print_batch_metrics], batch_size=200)


def f2():
    model = VGG16_tf()

    # Example loss and optimizer
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam()

    # Example metrics
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    @tf.function
    def train_step(images, labels):
        with tf.GradientTape() as tape:
            predictions = model(images, training=True)
            loss = loss_object(labels, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_loss(loss)
        train_accuracy(labels, predictions)

    def train(dataset, epochs):
        for epoch in range(epochs):
            for images, labels in dataset:
                train_step(images, labels)

                template = 'Epoch {}, Loss: {}, Accuracy: {}'
                print(template.format(epoch + 1,
                                      train_loss.result(),
                                      train_accuracy.result() * 100))

            # Reset the metrics for the next epoch
            train_loss.reset_state()
            train_accuracy.reset_state()

    (train_images, train_labels), (test_images, test_labels) = getCifar()

    train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).batch(32)

    train(train_dataset, epochs=1)


f2()
