# python3 after_graph_partition_hetero.py
import argparse

from gurobipy import *

os.environ['GRB_LICENSE_FILE'] = '/home/hola/solverLicense/gurobi.lic'

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
sys.path.append(project_root)
from optimizer.operator_device_placement.metis.subgraph_util import construct_sub_graph, WeightNormalizationFunction, \
    map_subgraph_to_device, init_graph_weight
from optimizer.optimization_problems.gurobi_util import init_computing_and_device_graph, gurobi_setup, \
    show_optimization_solution, show_graph_partition_info
from optimizer.operator_device_placement.metis.weight_functions import NodeWeightFunction, EdgeWeightFunction
from optimizer.experiment_figure_generation.tf_model_enum import TFModelEnum
from optimizer.operator_device_placement.placement import get_placement_info
from optimizer.scheduling.scheduling import execute_scheduling_function


def simulate(number_of_devices=2, model_type: TFModelEnum = TFModelEnum.SMALL,
             scheduling_function: str = "FIFO",
             placement: str = 'METIS',
             node_weight_function=NodeWeightFunction.AVE_COMP_COST,
             edge_weight_function=EdgeWeightFunction.SOURCE_OUTPUT_TENSOR,
             weight_norm_function=WeightNormalizationFunction.MIN_MAX,
             hetero_adjust_rate = None):
    # init fake data
    deviceTopo, comp_graph = init_computing_and_device_graph(number_of_devices, "comp_graph.json",
                                                             hetero_adjust_rate, model_type=model_type)
    # Init solver
    model = gurobi_setup("minimize_maxload")

    # init graph node/edge weight
    init_graph_weight(comp_graph, node_weight_function, edge_weight_function, weight_norm_function)


    # Partition the computation graph
    operator_device_mapping, edge_cut_list, edge_cut_weight_sum = (
        get_placement_info(placement, comp_graph, deviceTopo))

    # Update the op_id-subgraph_id mapping dict to op_id-device_id mapping dict
    device_subgraph_mapping = construct_sub_graph(comp_graph, operator_device_mapping)


    # two_dime_node_list is to test whether the
    two_dime_node_list: list[list] = [list(subgraph.nodes.keys()) for subgraph in device_subgraph_mapping.values()]

    # Define variables
    x = model.addVars(comp_graph.getOperatorIDs(), deviceTopo.getDeviceIDs(), vtype=GRB.BINARY,
                      name="x")  # [operator_id, device_id] == 1 means this operator is assigned to this device
    start = model.addVars(comp_graph.getOperatorIDs(), vtype=GRB.CONTINUOUS, lb=0.0,
                          name="start")  # start[node_id] represent the starting time of this node
    finish = model.addVars(comp_graph.getOperatorIDs(), vtype=GRB.CONTINUOUS, lb=0.0,
                           name="finish")  # finish[node_id] represent the finish time of this node
    comm_start = model.addVars(edge_cut_list, vtype=GRB.CONTINUOUS, lb=0.0,
                               name="comm_start")  # comm_start[source_op, dest_op] represent the communication
    comm_end = model.addVars(edge_cut_list, vtype=GRB.CONTINUOUS, lb=0.0, name="comm_end")
    comm_cost = model.addVars(edge_cut_list, vtype=GRB.CONTINUOUS, lb=0.0, name="comm_cost")

    '''
    Define Constraints
    '''

    # If we assume a homogeneous environment where each operator has the same time consumption on each device and the
    # bandwidth is also the same. Once we get the graph partition, the device-operation placement is already solved
    # because it does not matter where each sub-graph is placed.
    for op_id in comp_graph.getOperatorIDs():
        for device_id in deviceTopo.getDeviceIDs():
            if device_id == operator_device_mapping[op_id]:
                model.addConstr(x[op_id, device_id] == 1)
            else:
                model.addConstr(x[op_id, device_id] == 0)

    # Data dependency
    for source_op_ID, dest_op_ID in comp_graph.getEdgeIDs():
        model.addConstr(finish[source_op_ID] <= start[dest_op_ID])

    for node_id in comp_graph.getOperatorIDs():
        # Add constraints that each op's ending time = starting time + its computing time
        assigned_device = operator_device_mapping[node_id]
        comp_cost = comp_graph.getOperatorCompCostByDevice(node_id, assigned_device)
        model.addConstr(finish[node_id] == start[node_id] + comp_cost, name=f"finish_start_{node_id}")

    for edge_id_tuple in edge_cut_list:
        # only the edge in the edge_cut_list will bring communication cost since the source_op and destination-op are
        # placed on different devices
        source_op_ID, dest_op_ID = edge_id_tuple
        # Aggregate communication cost
        communication_cost = comp_graph.getEdgeTensorSize(source_op_ID, dest_op_ID) * deviceTopo.calUnitCommCostInUS(
            operator_device_mapping[source_op_ID], operator_device_mapping[dest_op_ID])

        # Ensures the communication starts only after the source operation finishes.
        model.addConstr(comm_start[source_op_ID, dest_op_ID] >= finish[source_op_ID],
                        f"bind_finish_to_comm_start_{source_op_ID}_{dest_op_ID}")

        # Ensures the communication ends before the destination operation starts.
        model.addConstr(comm_end[source_op_ID, dest_op_ID] <= start[dest_op_ID],
                        f"bind_comm_end_to_start_{source_op_ID}_{dest_op_ID}")

        # Ensures the communication duration covers the communication cost.
        model.addConstr(comm_end[source_op_ID, dest_op_ID] == comm_start[source_op_ID, dest_op_ID] + communication_cost,
                        f"data_dependency_{source_op_ID}_{dest_op_ID}")

        # just for verification
        model.addConstr(comm_cost[source_op_ID, dest_op_ID] == communication_cost,
                        f"comm_cost_{source_op_ID}_{dest_op_ID}")

    # It is an SCHEDULING problem within each device.
    execute_scheduling_function(scheduling_function, model, start=start, finish=finish, comm_start=comm_start,
                                comm_end=comm_end, comp_graph=comp_graph, device_subgraph_mapping=device_subgraph_mapping,
                                edge_cut_list=edge_cut_list, operator_device_mapping=operator_device_mapping)

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
        print(f"This is the optimal solution of such configuration: \n"
              f"number of operators: {comp_graph.number_of_nodes()} \n"
              f"number of devices: {deviceTopo.number_of_nodes()} \n"
              f"placement way of this DNN graph: {placement} \n"
              f"scheduling method in each device: {scheduling_function} \n"
              f"The environment is homogenous")
        show_graph_partition_info(comp_graph, operator_device_mapping, edge_cut_list, edge_cut_weight_sum)
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
    parser.add_argument('--number_of_device', type=int, default=5)
    parser.add_argument('--model', type=str, default='ALEXNET')
    parser.add_argument('--normalization_function', default='MinMax', type=str, help='')
    parser.add_argument('--scheduling', default='FIFO', type=str, help='')
    parser.add_argument('--placement', default='HETERO', type=str, help='')
    parser.add_argument('--hetero_rate', default=100, type=int, help='')

    args = parser.parse_args()

    model_mapping_dict = {'VGG': TFModelEnum.VGG, 'SMALL': TFModelEnum.SMALL, "ALEXNET": TFModelEnum.ALEXNET}
    weight_normalization_dict = {'MinMax': WeightNormalizationFunction.MIN_MAX}

    simulate(number_of_devices=args.number_of_device, model_type=model_mapping_dict[args.model],
             scheduling_function=args.scheduling,
             placement = args.placement,
             weight_norm_function=weight_normalization_dict[args.normalization_function],
             hetero_adjust_rate = args.hetero_rate)
