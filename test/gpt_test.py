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
from optimizer.model.graph import CompGraph
from optimizer.computing_graph.op_graph_util import compile_model, train_loss, train_accuracy, parse_to_comp_graph, \
    process_op_df, update_graph_with_prof, profile_train, get_cifar_data_loader, parse_tensorboard, \
    find_specific_pb_file, process_mem_dict, get_gpt_data_loader, get_proper_optimizer

print("TensorFlow version:", tf.__version__)
print("Keras version:", keras.__version__)

model, tokenizer = get_openai_gpt2_model_and_tokenizer()

optimizer = get_proper_optimizer(model)


def get_computation_graph(model: keras.Model, optimizer=keras.optimizers.Adam(3e-4),
                          loss_fn=keras.losses.SparseCategoricalCrossentropy(), batch_size=200, tokenizer=None,
                          max_len=128) -> CompGraph:
    compile_model(model, optimizer, loss_fn)


get_computation_graph(model, optimizer=optimizer, tokenizer=tokenizer)
