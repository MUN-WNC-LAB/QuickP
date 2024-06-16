import argparse
import sys
from DNN_model_tf.model_enum import model_mapping

import warnings

warnings.filterwarnings("ignore")

import keras
# it is weird that on my server, have to import torch to activate tensorflow
import tensorflow as tf
from keras import Sequential

sys.path.append("../../")
from optimizer.model.graph import CompGraph
from optimizer.computing_graph.op_graph_util import compile_model, train_loss, train_accuracy, parse_to_comp_graph, \
    process_op_df, profile_train, get_cifar_data_loader


def get_computation_graph(model: Sequential, optimizer=keras.optimizers.Adam(3e-4),
                          loss_fn=keras.losses.SparseCategoricalCrossentropy(), batch_size=200) -> CompGraph:
    compile_model(model, optimizer, loss_fn)

    # tf.function is a decorator that tells TensorFlow to create a graph from the Python function
    # https://www.tensorflow.org/guide/function
    # https://www.tensorflow.org/tensorboard/get_started
    @tf.function
    def training_step(train_x, train_y):
        # https://www.tensorflow.org/guide/autodiff
        with tf.GradientTape() as tape:
            # Forward pass
            predictions = model(train_x, training=True)
            loss = loss_fn(train_y, predictions)
            loss += sum(model.losses)
        gradients = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(gradients, model.trainable_weights))

        train_loss.update_state(loss)
        train_accuracy.update_state(train_y, predictions)
        return loss

    inputs_constraint = tf.TensorSpec(shape=[batch_size, 32, 32, 3], dtype=tf.float32, name="input")
    targets_constraint = tf.TensorSpec(shape=[batch_size, 1], dtype=tf.uint8, name="target")

    concrete_function = training_step.get_concrete_function(inputs_constraint, targets_constraint)
    parent_directory = profile_train(concrete_function, get_cifar_data_loader(batch_size, True), num_prof_step=20)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some dictionary.')
    parser.add_argument('--model', type=str, required=True,
                        help='specify the model type')

    args = parser.parse_args()
    get_computation_graph(model=model_mapping.get(args.model))
