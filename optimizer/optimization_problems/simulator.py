# python3 after_graph_partition_hetero.py
import argparse

import networkx as nx
from gurobipy import *
import torch
import tensorflow as tf

os.environ['GRB_LICENSE_FILE'] = '/home/hola/solverLicense/gurobi.lic'

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
sys.path.append(project_root)
from optimizer.optimization_problems.scheduling_algorithm import create_topological_position_dict, TopoSortFunction, \
    from_topo_list_to_dict
from optimizer.graph_partitioner.metis_partition import metis_partition
from optimizer.graph_partitioner.subgraph_util import construct_sub_graph, WeightNormalizationFunction, normalize_list, \
    map_subgraph_to_device
from optimizer.optimization_problems.gurobi_util import init_computing_and_device_graph, gurobi_setup, \
    show_optimization_solution, show_graph_partition_info, get_subgraph_topo_dict, sort_edges_by_topo_order, \
    get_subgraph_op_num_weight_sum_dict
from optimizer.graph_partitioner.weight_functions import NodeWeightFunction, EdgeWeightFunction
from optimizer.experiment_figure_generation.tf_model_enum import TFModelEnum


def optimize_after_graph_partition(number_of_devices=2, model_type: TFModelEnum = TFModelEnum.SMALL,
                                   node_weight_function=NodeWeightFunction.AVE_COMP_COST,
                                   edge_weight_function=EdgeWeightFunction.SOURCE_OUTPUT_TENSOR,
                                   adjust_matrix=None, weight_norm_function=WeightNormalizationFunction.MIN_MAX,
                                   scheduling_algorithm=TopoSortFunction.KAHN):
    # init fake data
    deviceTopo, comp_graph = init_computing_and_device_graph(number_of_devices, "comp_graph_after_partition.json",
                                                             100, model_type=model_type)
    device_computing_cost_dict = comp_graph.get_comp_cost_sum_ratio(number_of_devices)
    # Init solver
    model = gurobi_setup("minimize_maxload")

    # Partition the computation graph
    partition_dict, edge_cut_list, edge_cut_weight_sum = metis_partition(comp_graph, num_partitions=number_of_devices,
                                                                         edge_weight_function=edge_weight_function,
                                                                         node_weight_function=node_weight_function,
                                                                         adjust_matrix=adjust_matrix,
                                                                         weight_normalize=weight_norm_function)
    subgraph_dict = construct_sub_graph(comp_graph, partition_dict)
    operator_device_dict = map_subgraph_to_device(partition_dict, deviceTopo.getDeviceIDs(), device_computing_cost_dict)

    # two_dime_node_list is to test whether the
    two_dime_node_list: list[list] = [list(subgraph.nodes.keys()) for subgraph in subgraph_dict.values()]

    # Define variables
    x = model.addVars(comp_graph.getOperatorIDs(), deviceTopo.getDeviceIDs(), vtype=GRB.BINARY, name="x")  # [operator_id, device_id] == 1 means this operator is assigned to this device
    start = model.addVars(comp_graph.getOperatorIDs(), vtype=GRB.CONTINUOUS, lb=0.0, name="start")  # start[node_id] represent the starting time of this node
    finish = model.addVars(comp_graph.getOperatorIDs(), vtype=GRB.CONTINUOUS, lb=0.0, name="finish")  # finish[node_id] represent the finish time of this node


    '''
    Define Constraints
    '''

    for op_id, device_id in operator_device_dict.items():
        model.addConstr(x[op_id, device_id] == 1)


    # Add the constraint that each operator can only be placed on one device
    for op_id in comp_graph.getOperatorIDs():
        model.addConstr(
            quicksum(x[op_id, device_id] for device_id in deviceTopo.getDeviceIDs()) == 1,
            name=f"op_placement_{op_id}"
        )

    for node_id in comp_graph.getOperatorIDs():
        # Add constraints that each op's ending time = starting time + its computing time
        assigned_device = operator_device_dict[node_id]
        comp_cost = comp_graph.getOperatorCompCostByDevice(node_id, assigned_device)
        model.addConstr(finish[node_id] == start[node_id] + comp_cost, name=f"finish_start_{node_id}")

    # Add constraint that if op2 depends on op1, the starting time of op2 will be the ending time of op1 + communication delay if these two ops are not placed on the same device
    # device_pairs is a Set obj with unique device pair
    for edge_id_tuple in edge_cut_list:
        # only the edge in the edge_cut_list will bring communication cost since the source_op and destination-op are
        # placed on different devices
        source_op_ID, dest_op_ID = edge_id_tuple
        # Aggregate communication cost
        communication_cost = comp_graph.getEdgeTensorSize(source_op_ID, dest_op_ID) * deviceTopo.calUnitCommCostInUS(
            operator_device_dict[source_op_ID], operator_device_dict[dest_op_ID])

        # Ensures the communication starts only after the source operation finishes.
        model.addConstr(finish[source_op_ID] + communication_cost <= start[dest_op_ID])

    for edge in comp_graph.getEdgeIDs():
        model.addConstr(finish[edge[0]] <= start[edge[1]])

    # It is an SCHEDULING problem within each device.
    for subgraph_id, subgraph in subgraph_dict.items():
        topo_list = scheduling_algorithm(subgraph)
        # Since all nodes in a subgraph will be allocated to the same device, add constraint to ensure each device
        # processes only one operator at a time. Also, it indicates the data dependency
        for a, b in zip(topo_list, topo_list[1:]):
            model.addConstr(finish[a] <= start[b])

    # TotalLatency that we are minimizing
    TotalLatency = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0)
    for op_end in finish.values():
        model.addConstr(TotalLatency >= op_end, "satisfy each deice's latency")

    # Set the target of solver
    model.setObjective(TotalLatency, GRB.MINIMIZE)

    # Run the solver
    sys.stdout.flush()
    model.optimize()

    # Check optimization status
    if model.status == GRB.INFEASIBLE:
        print("Model is infeasible. Computing IIS...")
        model.computeIIS()
        model.write("model.ilp")
        print("IIS written to model.ilp")

        # Print the constraints that are in the IIS
        print("\nThe following constraints are in the IIS:")
        for constr in model.getConstrs():
            if constr.IISConstr:
                print(f"{constr.ConstrName}")
        if model is not None:
            model.dispose()
        disposeDefaultEnv()
    elif model.status == GRB.UNBOUNDED:
        print("Model is unbounded.")
    # this is the main process part after a solution is reached
    elif model.status == GRB.OPTIMAL:
        show_optimization_solution(model, x, comp_graph, deviceTopo, start, finish, True, two_dime_node_list)
        show_graph_partition_info(comp_graph, partition_dict, edge_cut_list, edge_cut_weight_sum)
        optimal_value = model.ObjVal
        if model is not None:
            model.dispose()
        disposeDefaultEnv()
        return optimal_value
    else:
        print(f"Optimization ended with status {model.status}")
        if model is not None:
            model.dispose()
        disposeDefaultEnv()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='arguments for optimization problem after graph partitioning')
    parser.add_argument('--number_of_device', type=int, default=2)
    parser.add_argument('--model', type=str, default='SMALL')
    parser.add_argument('--normalization_function', default='MinMax', type=str, help='')
    parser.add_argument('--node_weight_function', default='comp_cost', type=str, help='')
    parser.add_argument('--edge_weight_function', default='comm_cost', type=str, help='')
    parser.add_argument('--topo_sort_function', default='Kahn', type=str, help='it is regarding operator and communication scheduling')

    args = parser.parse_args()

    model_mapping_dict = {'VGG': TFModelEnum.VGG, 'SMALL': TFModelEnum.SMALL, "ALEXNET": TFModelEnum.ALEXNET}
    weight_normalization_dict = {'MinMax': WeightNormalizationFunction.MIN_MAX}
    topo_sort_dict = {"Kahn": TopoSortFunction.KAHN, "Default": TopoSortFunction.DEFAULT, "KahnPriority": TopoSortFunction.KAHN_PRIORITY}

    optimize_after_graph_partition(number_of_devices=args.number_of_device, model_type=model_mapping_dict[args.model],
                                   adjust_matrix={"node_enable": True, "edge_enable": False, 'adjustment_ratio': 0},
                                   weight_norm_function=weight_normalization_dict[args.normalization_function],
                                   scheduling_algorithm=topo_sort_dict[args.topo_sort_function])