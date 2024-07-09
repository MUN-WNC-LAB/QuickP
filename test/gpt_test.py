import torch
# https://discuss.tensorflow.org/t/tensorflow-operator-computational-graph/3759
import keras
# it is weird that on my server, have to import torch to activate tensorflow
import tensorflow as tf
from keras import Sequential
from networkx import is_directed_acyclic_graph
from transformers import TFGPT2LMHeadModel

from DNN_model_tf.openai_gpt2 import get_openai_gpt2_model_and_tokenizer
from optimizer.computing_graph.tool import Conf_TB, CONF
from optimizer.model.graph import CompGraph, visualize_graph
from optimizer.computing_graph.op_graph_util import compile_model, train_loss, train_accuracy, parse_to_comp_graph, \
    process_op_df, update_graph_with_prof, profile_train, get_cifar_data_loader, parse_tensorboard, \
    find_specific_pb_file, process_mem_dict, get_gpt_data_loader, get_proper_optimizer

model, tokenizer = get_openai_gpt2_model_and_tokenizer()

optimizer = get_proper_optimizer(model)


def test_gpt_data_loader():
    for index, (text, label) in enumerate(get_gpt_data_loader().take(10)):
        # EagerTensor is the type of text or label
        # < dtype: 'string' > < dtype: 'int64' >
        # (200,)(200, )
        print(text.dtype, label.dtype)
        print(text.shape, label.shape)


def get_computation_graph(model: keras.Model, optimizer=keras.optimizers.Adam(3e-4),
                          loss_fn=keras.losses.SparseCategoricalCrossentropy(), batch_size=200, tokenizer=None,
                          max_len=128) -> CompGraph:
    compile_model(model, optimizer, loss_fn)
    if (not tokenizer or not max_len) and isinstance(model, TFGPT2LMHeadModel):
        raise ValueError("tokenizer must be set when using TFGPT2LMHeadModel")

    # tf.function is a decorator that tells TensorFlow to create a graph from the Python function
    # https://www.tensorflow.org/guide/function
    # https://www.tensorflow.org/tensorboard/get_started
    @tf.function
    def training_step(train_x: tf.Tensor, train_y: tf.Tensor, attention_mask: tf.Tensor = None):
        # https://www.tensorflow.org/guide/autodiff
        with tf.GradientTape() as tape:
            # Forward pass
            if isinstance(model, TFGPT2LMHeadModel):
                outputs = model(train_x, attention_mask=attention_mask, labels=train_y)
                loss = outputs.loss
                predictions = outputs.logits
            else:
                predictions = model(train_x, training=True)
                loss = loss_fn(train_y, predictions)
                loss += sum(model.losses)
        gradients = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(gradients, model.trainable_weights))

        train_loss.update_state(loss)
        train_accuracy.update_state(train_y, predictions)
        return loss

    if isinstance(model, keras.Sequential):
        # based on the Cifar-10 dataset
        inputs_spec = [
            tf.TensorSpec(shape=[batch_size, 32, 32, 3], dtype=tf.float32, name="input"),
            tf.TensorSpec(shape=[batch_size, 1], dtype=tf.uint8, name="target")
        ]
    elif isinstance(model, TFGPT2LMHeadModel):
        # Make the tensor size unique max_len. If actual shape < max_len, use padding to fill the rest
        # based on the imdb dataset
        inputs_spec = [
            tf.TensorSpec(shape=[batch_size, max_len], dtype=tf.int32, name="input_ids"),
            tf.TensorSpec(shape=[batch_size, max_len], dtype=tf.int32, name="labels"),
            tf.TensorSpec(shape=[batch_size, max_len], dtype=tf.int32, name="attention_mask")
        ]
    else:
        raise ValueError("Unsupported model type")

    concrete_function = training_step.get_concrete_function(*inputs_spec)

    graph = parse_to_comp_graph(concrete_function)
    data_loader_func = get_cifar_data_loader if isinstance(model, keras.Sequential) else get_gpt_data_loader
    parent_directory = profile_train(concrete_function, data_loader_func(batch_size, True), num_prof_step=20)


get_computation_graph(model=model, optimizer=optimizer, tokenizer=tokenizer)
