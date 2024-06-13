import argparse
import json
import socket
import sys

from tensorflow.python.eager.polymorphic_function import concrete_function

from optimizer.computing_graph.tool import Conf_TB, CONF
from tf_util import profile_train, find_specific_pb_file, parse_tensorboard, process_op_df, process_mem_dict, \
    get_cifar_data_loader

sys.path.append("../../")


import warnings

warnings.filterwarnings("ignore")


def get_computing_cost_matrix():
    parent_directory = profile_train(concrete_function, get_cifar_data_loader(batch_size, True), num_prof_step=20)
    plane_pb_file = find_specific_pb_file(parent_directory, "xplane.pb")
    dataframe = parse_tensorboard(plane_pb_file, Conf_TB(CONF.OP))
    mem_data = parse_tensorboard(plane_pb_file, Conf_TB(CONF.MEM))
    op_dict = process_op_df(dataframe)
    mem_dict = process_mem_dict(mem_data)


if __name__ == "__main__":
    get_computing_cost_matrix()