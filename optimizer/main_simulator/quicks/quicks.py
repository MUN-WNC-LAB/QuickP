import networkx as nx
from gurobipy import Model, GRB
from networkx import Graph

from DNN_model_tf.tf_model_enum import TFModelEnum
from optimizer.main_simulator.gurobi_util import init_computing_and_device_graph
from optimizer.main_simulator.quicks.simulator_quicks import evaluate_quick
from optimizer.main_simulator.simulator_util import get_comp_cost_dict, get_comm_cost_dict
from optimizer.model.graph import CompGraph
from optimizer.operator_device_placement.metis.subgraph_util import WeightNormalizationFunction, init_graph_weight, \
    construct_sub_graph
from optimizer.operator_device_placement.metis.weight_functions import NodeWeightFunction, EdgeWeightFunction
from optimizer.operator_device_placement.placement import get_placement_info
from optimizer.scheduling.near_optimal_scheduling_with_sampling import get_device_unreachable_pairs_mapping, split_nodes
from optimizer.main_simulator.quicks.quicks_list_schedule import quicks_list_schedule
from optimizer.scheduling.scheduling_util import split_three_stage_subgraph


def quickS(comp_graph: CompGraph, deviceTopo):
    # Partition the computation graph
    operator_device_mapping, edge_cut_list, edge_cut_weight_sum = (
        get_placement_info("METIS", comp_graph, deviceTopo))
    # Update the op_id-subgraph_id mapping dict to op_id-device_id mapping dict
    device_subgraph_mapping = construct_sub_graph(comp_graph, operator_device_mapping)
    # Get computation and communication cost
    op_computing_cost_mapping = get_comp_cost_dict(comp_graph, operator_device_mapping)
    edge_cut_communication_cost_mapping = get_comm_cost_dict(comp_graph, deviceTopo, edge_cut_list,
                                                             operator_device_mapping)
    rank_map = calculate_rank_map(comp_graph,device_subgraph_mapping, edge_cut_list, operator_device_mapping, op_computing_cost_mapping)
    evaluate_quick(comp_graph, deviceTopo, operator_device_mapping, edge_cut_list, edge_cut_weight_sum, graph_init["model_type"], rank_map)


def calculate_rank_map(comp_graph: CompGraph,
                       device_subgraph_mapping: dict, edge_cut_list: list, operator_device_mapping: dict, computing_cost_dict,
                       ):
    rank_map = {}

    # form new device non-isolated part mapping
    # split into isolated and non-isolated part
    for device, subgraph in device_subgraph_mapping.items():
        # Simply the search space by
        # Simply the search space by
        stage_one, stage_two, stage_three, reliance_comm_map, reliance_node_map = split_three_stage_subgraph(
            subgraph, operator_device_mapping, edge_cut_list)
        # turn the reliance_map into computing score
        node_score_map = {
            node: (
                sum(computing_cost_dict[relied_node] for relied_node in
                    reliance_node_map[node])
            )
            for node in reliance_node_map}
        # give stage one the highest rank and the three the lowest rank
        for stage_one_node in stage_one:
            assert len(reliance_node_map[stage_one_node]) != 0
            rank_map[stage_one_node] = 10 + node_score_map[stage_one_node]
        for stage_two_node in stage_two:
            assert len(reliance_node_map[stage_two_node]) != 0
            rank_map[stage_two_node] = 10 + node_score_map[stage_two_node]
        for stage_three_node in stage_three:
            assert len(reliance_node_map[stage_three_node]) == 0
            rank_map[stage_three_node] = 0

    return rank_map


if __name__ == '__main__':
    graph_init = {
        "number_of_devices": 6,
        "model_type": TFModelEnum.ALEXNET,
        "node_weight_function": NodeWeightFunction.AVE_COMP_COST,
        "edge_weight_function": EdgeWeightFunction.SOURCE_OUTPUT_TENSOR,
        "weight_norm_function": WeightNormalizationFunction.MIN_MAX,
    }

    # init fake data
    deviceTopo, comp_graph = init_computing_and_device_graph(graph_init["number_of_devices"], None,
                                                             model_type=graph_init["model_type"])
    # init graph node/edge weight
    init_graph_weight(comp_graph, graph_init["node_weight_function"], graph_init["edge_weight_function"],
                      graph_init["weight_norm_function"])

    quickS(comp_graph, deviceTopo)
