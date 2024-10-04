from gurobipy import *
from networkx.classes import Graph

from optimizer.main_simulator.quicks.near_optimal_scheduling import sampling_based_near_optimal_schedule
from optimizer.scheduling.near_optimal_scheduling_with_sampling import SamplingFunction

os.environ['GRB_LICENSE_FILE'] = '/home/hola/solverLicense/gurobi.lic'

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
sys.path.append(project_root)

from optimizer.main_simulator.gurobi_util import init_computing_and_device_graph, gurobi_setup, \
    show_optimization_solution, show_graph_partition_info


def get_relied_component_execution_order(relied_graph: Graph, edge_cut_list: list, operator_device_mapping,
                                         op_computing_cost_mapping, edge_cut_communication_cost_mapping, heuristic_rank_map, device_relied_component_map, rho):
    # clean edge cut and operator_device_mapping after the non-exporting node removal
    edge_cut_list = [(u, v) for u, v in edge_cut_list if u in relied_graph.nodes and v in relied_graph.nodes]
    operator_device_mapping = {op: device for op, device in operator_device_mapping.items() if op in relied_graph.nodes}

    # Init solver
    model = gurobi_setup("minimize_maxload")

    # Define variables

    start = model.addVars(relied_graph.nodes, vtype=GRB.CONTINUOUS, lb=0.0,
                          name="start")  # start[node_id] represent the starting time of this node
    finish = model.addVars(relied_graph.nodes, vtype=GRB.CONTINUOUS, lb=0.0,
                           name="finish")  # finish[node_id] represent the finish time of this node
    comm_start = model.addVars(edge_cut_list, vtype=GRB.CONTINUOUS, lb=0.0,
                               name="")  # comm_start[source_op, dest_op] represent the communication
    comm_end = model.addVars(edge_cut_list, vtype=GRB.CONTINUOUS, lb=0.0, name="")

    '''
    Define Constraints
    '''

    for node_id in relied_graph.nodes:
        # Add constraints that each op's ending time = starting time + its computing time
        model.addConstr(finish[node_id] == start[node_id] + op_computing_cost_mapping[node_id], name=f"finish_start_{node_id}")

    # Data dependency for same-device communication
    non_edge_cut_list = [edge for edge in relied_graph.edges if edge not in edge_cut_list]
    for edge_id_tuple in non_edge_cut_list:
        source_op_ID, target_op_ID = edge_id_tuple
        model.addConstr(finish[source_op_ID] <= start[target_op_ID])

    # Data dependency for cross-device communication
    for edge_id_tuple in edge_cut_list:
        # only the edge in the edge_cut_list will bring communication cost since the source_op and destination-op are
        # placed on different devices
        source_op_ID, dest_op_ID = edge_id_tuple
        # Ensures the communication starts only after the source operation finishes.
        model.addConstr(finish[source_op_ID] <= comm_start[source_op_ID, dest_op_ID],
                        name = "")
        # Ensures the communication duration covers the communication cost.
        model.addConstr(comm_start[source_op_ID, dest_op_ID] + edge_cut_communication_cost_mapping[edge_id_tuple] == comm_end[source_op_ID, dest_op_ID],
                        name = "")
        # Ensures the communication ends before the destination operation starts.
        model.addConstr(comm_end[source_op_ID, dest_op_ID] <= start[dest_op_ID],
                        name = "")

    # It is an SCHEDULING problem within each device.
    sampling_based_near_optimal_schedule(model, start, finish, comm_start, comm_end, relied_graph,
                                         device_relied_component_map, edge_cut_list, operator_device_mapping, heuristic_rank_map,
                                         op_computing_cost_mapping, rho=rho, sampling_function=SamplingFunction.HEAVY_HITTER)

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
        print('latency', model.ObjVal)
        device_operator_execution_order = get_device_operator_execution_order(start, finish, operator_device_mapping)
        rank_map = calculate_extra_rank_of_relied_node(device_operator_execution_order)
        if model is not None:
            model.dispose()
        disposeDefaultEnv()
        return rank_map
    else:
        print(f"Optimization ended with status {model.status}")
        if model is not None:
            model.dispose()
        disposeDefaultEnv()

def get_device_operator_execution_order(start, finish, operator_device_placement):
    device_op_order = {}

    for op, device in operator_device_placement.items():
        if device not in device_op_order:
            device_op_order[device] = []
        device_op_order[device].append(op)

    # Sort operators by their start times for each device
    for device, ops in device_op_order.items():
        device_op_order[device] = sorted(ops, key=lambda node: start[node].X)

    return device_op_order


def calculate_extra_rank_of_relied_node(device_operator_execution_order):
    node_rank = {}
    for device, execution_order in device_operator_execution_order.items():
        for i, op in enumerate(reversed(execution_order)):
            node_rank[op] = i * 5  # 0 for the last, 5 for the second last, 10 for the third last, etc.

    return node_rank
