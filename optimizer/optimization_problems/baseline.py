# python3 baseline.py

from gurobipy import *
import torch
import tensorflow as tf

os.environ['GRB_LICENSE_FILE'] = '/home/hola/solverLicense/gurobi.lic'

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
sys.path.append(project_root)
from optimizer.model.graph import determine_node_order
from optimizer.optimization_problems.scheduling_algorithm import create_topological_position_dict
from optimizer.optimization_problems.gurobi_util import gurobi_setup, init_computing_and_device_graph, \
    show_optimization_solution, sort_edges_by_topo_order
from optimizer.experiment_figure_generation.tf_model_enum import TFModelEnum


def optimize_baseline(number_of_devices=2, model_type: TFModelEnum = TFModelEnum.SMALL):
    # init fake data
    deviceTopo, comp_graph = init_computing_and_device_graph(number_of_devices, 'comp_graph_baseline.json',
                                                             model_type=model_type)
    # global_topo_dict will decide the
    global_topo_dict = create_topological_position_dict(comp_graph)
    edge_topo_ordered_list = sort_edges_by_topo_order(comp_graph.getEdgeIDs(), global_topo_dict)

    # Init solver
    model = gurobi_setup("minimize_maxload")

    # Define variables
    x = {}  # key will be (operator_id, machine_id), value will be 1 or 0; x[3, 1] = 1 means operator 3 get allocated to device 1
    start = {}  # start[node_id] represent the starting time of this node
    finish = {}  # finish[node_id] represent the finish time of this node
    comm_start = {}  # comm_start[source_op, dest_op] represent the communication
    comm_end = {}
    comm_cost = {}

    # Initialize all variables with names
    for node_id in comp_graph.getOperatorIDs():
        for machine_id in deviceTopo.getDeviceIDs():
            x[node_id, machine_id] = model.addVar(vtype=GRB.BINARY, name=f"x_{node_id}_{machine_id}")
        start[node_id] = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"start_{node_id}")
        finish[node_id] = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"finish_{node_id}")

    for edge_id_tuple in comp_graph.getEdgeIDs():
        source_op_ID, dest_op_ID = edge_id_tuple
        comm_start[source_op_ID, dest_op_ID] = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0,
                                                            name=f"comm_start_{source_op_ID}_{dest_op_ID}")
        comm_end[source_op_ID, dest_op_ID] = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0,
                                                          name=f"comm_end_{source_op_ID}_{dest_op_ID}")
        comm_cost[source_op_ID, dest_op_ID] = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0,
                                                           name=f"comm_cost_{source_op_ID}_{dest_op_ID}")

    # Add constraints that schedule every node on exactly one machine
    for op in comp_graph.getOperatorIDs():
        model.addConstr(quicksum(x[op, device] for device in deviceTopo.getDeviceIDs()) == 1, name=f"one_device_{op}")

    # Add constraints that operators assigned cannot exceed the capacity
    for machine_id in deviceTopo.getDeviceIDs():
        mem_sum = quicksum(x[node_id, machine_id] * comp_graph.getOperator(node_id)["mem"]
                           for node_id in comp_graph.getOperatorIDs())
        model.addConstr(mem_sum <= deviceTopo.getDeviceMaxMem(machine_id),
                        f"satisfy_memory_constraint_{machine_id}")

    # Add constraints that each device should have at least one operator assigned
    for machine_id in deviceTopo.getDeviceIDs():
        model.addConstr(quicksum(x[node_id, machine_id] for node_id in comp_graph.getOperatorIDs()) >= 1,
                        name=f"at_least_one_op_{machine_id}")

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
        (source_op_ID, dest_op_ID): comp_graph.getOperatorOutputInBit(source_op_ID)
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

    # Add constraint to ensure each device processes only one operator at a time. This is a SCHEDULING problem.
    for device in deviceTopo.getDeviceIDs():
        # ensures that each pair of operations is only considered once
        for op1, op2 in itertools.combinations(comp_graph.getOperatorIDs(), 2):
            node_order = determine_node_order(global_topo_dict, op1, op2)
            if node_order == 1:
                y = model.addVar(vtype=GRB.BINARY)
                model.addGenConstrIndicator(y, True, finish[op1] <= start[op2])
            elif node_order == 2:
                y = model.addVar(vtype=GRB.BINARY)
                model.addGenConstrIndicator(y, True, finish[op2] <= start[op1])
            else:
                raise ValueError("Invalid node order")

            # If on the same device, ensure that the operators do not overlap
            model.addConstr(y >= x[op1, device] + x[op2, device] - 1)

    # Add constraint to ensure each device can only send or receive from one link at a time, communication scheduling
    for (source_op_ID1, dest_op_ID1), (source_op_ID2, dest_op_ID2) in zip(edge_topo_ordered_list,
                                                                              edge_topo_ordered_list[1:]):
        for device_id_src, device_id_dest in itertools.combinations(deviceTopo.getDeviceIDs(), 2):
            no_overlap = model.addVar(vtype=GRB.BINARY)

            # Enforce non-overlapping constraints using indicator constraints
            model.addGenConstrIndicator(no_overlap, True, comm_end[source_op_ID1, dest_op_ID1] <= comm_start[
                source_op_ID2, dest_op_ID2])

            # if using the same link,
            # either x[source_op_ID1, device_id_src] + x[dest_op_ID1, device_id_dest] + x[source_op_ID2, device_id_src] + x[dest_op_ID2, device_id_dest]
            # or x[source_op_ID1, device_id_dest] + x[dest_op_ID1, device_id_src] + x[source_op_ID2, device_id_dest] + x[dest_op_ID2, device_id_src]
            # will be 4
            model.addConstr(
                no_overlap >= (
                    x[source_op_ID1, device_id_src] + x[dest_op_ID1, device_id_dest] + x[
                        source_op_ID2, device_id_src] + x[dest_op_ID2, device_id_dest] +
                    x[source_op_ID1, device_id_dest] + x[dest_op_ID1, device_id_src] + x[
                        source_op_ID2, device_id_dest] + x[dest_op_ID2, device_id_src] - 3
                )
            )

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
        show_optimization_solution(model, x, comp_graph, deviceTopo, start, finish)
        print("number of operators in total", len(comp_graph))
        optimal_value = model.ObjVal
        del model
        disposeDefaultEnv()
        return optimal_value
    else:
        print(f"Optimization ended with status {model.status}")


if __name__ == '__main__':
    optimize_baseline()
