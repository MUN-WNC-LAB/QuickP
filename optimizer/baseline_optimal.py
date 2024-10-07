import argparse

import networkx as nx
from gurobipy import *

from DNN_model_tf.tf_model_enum import TFModelEnum
from optimizer.operator_device_placement.metis.subgraph_util import WeightNormalizationFunction

os.environ['GRB_LICENSE_FILE'] = '/home/hola/solverLicense/gurobi.lic'

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
sys.path.append(project_root)
from optimizer.main_simulator.gurobi_util import gurobi_setup, init_computing_and_device_graph, \
    show_optimization_solution_for_baseline, get_proper_M


def joint_optimize(comp_graph, deviceTopo) -> dict:
    def get_operator_device_mapping_through_x(x):
        mapping = {}
        for (operator_id, device_id), var in x.items():
            # Check if the variable has a value of 1 (operator is assigned to device)
            if var.X > 0.5:  # Since the variable is binary, this checks if it is assigned
                mapping[operator_id] = device_id
        return mapping

    # Init solver
    model = gurobi_setup("minimize_maxload")

    # Define variables
    x = model.addVars(comp_graph.getOperatorIDs(), deviceTopo.getDeviceIDs(), vtype=GRB.BINARY,
                      name="x")  # [operator_id, device_id] == 1 means this operator is assigned to this device
    start = model.addVars(comp_graph.getOperatorIDs(), vtype=GRB.CONTINUOUS, lb=0.0,
                          name="start")  # start[node_id] represent the starting time of this node
    finish = model.addVars(comp_graph.getOperatorIDs(), vtype=GRB.CONTINUOUS, lb=0.0,
                           name="finish")  # finish[node_id] represent the finish time of this node
    comm_start = model.addVars(comp_graph.getEdgeIDs(), vtype=GRB.CONTINUOUS, lb=0.0,
                               name="comm_start")  # comm_start[source_op, dest_op] represent the communication
    comm_end = model.addVars(comp_graph.getEdgeIDs(), vtype=GRB.CONTINUOUS, lb=0.0, name="comm_end")
    comm_cost = model.addVars(comp_graph.getEdgeIDs(), vtype=GRB.CONTINUOUS, lb=0.0, name="comm_cost")

    # Add constraints that schedule every node on exactly one machine
    for op in comp_graph.getOperatorIDs():
        model.addConstr(quicksum(x[op, device] for device in deviceTopo.getDeviceIDs()) == 1, name=f"one_device_{op}")

    # Add constraints that each op's ending time = starting time + its computing time
    for node_id in comp_graph.getOperatorIDs():
        comp_cost = quicksum(x[node_id, device_id] * comp_graph.getOperatorCompCostByDevice(node_id, device_id)
                             for device_id in deviceTopo.getDeviceIDs())
        model.addConstr(finish[node_id] == start[node_id] + comp_cost, name=f"finish_start_{node_id}")

    # Add constraint that if op2 depends on op1, the starting time of op2 will be the ending time of op1 + communication delay if these two ops are not placed on the same device
    device_pairs = {(src, dest) for src in deviceTopo.getDeviceIDs() for dest in deviceTopo.getDeviceIDs() if
                    src != dest}
    # unit_comm_costs[device_id_src, device_id_dest] means the com cost per bit from device with source device to dest device
    unit_comm_costs = {
        (src_device, dest_device): deviceTopo.calUnitCommCostInUS(src_device, dest_device)
        for src_device, dest_device in device_pairs
    }
    tensor_sizes = {
        (source_op_ID, dest_op_ID): comp_graph.getEdgeTensorSize(source_op_ID, dest_op_ID)
        for source_op_ID, dest_op_ID in comp_graph.getEdgeIDs()
    }
    for edge_id_tuple in list(comp_graph.getEdgeIDs()):
        source_op_ID, dest_op_ID = edge_id_tuple
        # Aggregate communication cost
        comm_cost_expr = quicksum(
            unit_comm_costs[device_id_src, device_id_dest] * tensor_sizes[source_op_ID, dest_op_ID] *
            x[source_op_ID, device_id_src] * x[dest_op_ID, device_id_dest]
            for device_id_src, device_id_dest in device_pairs
        )

        model.addConstr(comm_cost[source_op_ID, dest_op_ID] == comm_cost_expr, f"comm_cost_{source_op_ID}_{dest_op_ID}")

        # Ensures the communication starts only after the source operation finishes.
        model.addConstr(comm_start[source_op_ID, dest_op_ID] >= finish[source_op_ID],
                        f"bind_finish_to_comm_start_{source_op_ID}_{dest_op_ID}")

        # Ensures the communication ends before the destination operation starts.
        model.addConstr(comm_end[source_op_ID, dest_op_ID] <= start[dest_op_ID],
                        f"bind_comm_end_to_start_{source_op_ID}_{dest_op_ID}")

        # Ensures the communication duration covers the communication cost.
        model.addConstr(comm_end[source_op_ID, dest_op_ID] == comm_start[source_op_ID, dest_op_ID] + comm_cost[
            source_op_ID, dest_op_ID],
                        f"data_dependency_{source_op_ID}_{dest_op_ID}")

    M = get_proper_M(model_type)
    z = {}
    # Operator-scheduling
    for i, j in itertools.combinations(comp_graph.getOperatorIDs(), 2):
        # For each consecutive pair of operators, add a constraint for each device
        for d in deviceTopo.getDeviceIDs():
            if nx.has_path(comp_graph, i, j):
                model.addConstr(finish[i] <= start[j] + M * (2 - x[i, d] - x[j, d]))
            elif nx.has_path(comp_graph, j, i):
                model.addConstr(finish[j] <= start[i] + M * (2 - x[i, d] - x[j, d]))
            else:
                # Constraint to ensure non-overlapping periods
                z[i, j] = model.addVar(vtype=GRB.BINARY, name=f"order_{i}_{j}")
                model.addConstr(finish[i] <= start[j] + M * (1 - z[i, j] + 1 - x[i, d] + 1 - x[j, d]))
                model.addConstr(finish[j] <= start[i] + M * (z[i, j] + 1 - x[i, d] + 1 - x[j, d]))

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
    elif model.status == GRB.UNBOUNDED:
        print("Model is unbounded.")
    elif model.status == GRB.OPTIMAL:
        show_optimization_solution_for_baseline(model, x, comp_graph, deviceTopo, start, finish)
        operator_device_mapping = get_operator_device_mapping_through_x(x)
        del model
        disposeDefaultEnv()
        return operator_device_mapping
    else:
        print(f"Optimization ended with status {model.status}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='arguments for optimization problem after graph partitioning')
    parser.add_argument('--number_of_device', type=int, default=2)
    parser.add_argument('--model', type=str, default='SMALL')
    parser.add_argument('--hetero_rate', default=None, type=int, help='')

    args = parser.parse_args()
    model_type = getattr(TFModelEnum, args.model, None)

    model_mapping_dict = {'VGG': TFModelEnum.VGG, 'SMALL': TFModelEnum.SMALL, "ALEXNET": TFModelEnum.ALEXNET}
    weight_normalization_dict = {'MinMax': WeightNormalizationFunction.MIN_MAX}

    # init fake data
    deviceTopo, comp_graph = init_computing_and_device_graph(args.number_of_device, args.hetero_rate,
                                                             model_type=model_mapping_dict[args.model])
    joint_optimize(comp_graph, deviceTopo)
