# python3 estimator_test.py

from gurobipy import *
import torch
import tensorflow as tf

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.append(project_root)
from optimizer.computing_graph.computing_graph import get_computation_graph
from py_util import tensor_shape_to_bits
from optimizer.model.graph import DeviceGraph, CompGraph
from DNN_model_tf.small import small_tf

# init fake data
if not os.path.exists('comp_graph.json'):
    model = small_tf()
    comp_graph = get_computation_graph(model=model)
    comp_graph.generata_random_cost(2)
    comp_graph.save_to_file('comp_graph.json')

comp_graph = CompGraph.load_from_file('comp_graph.json')

deviceTopo = DeviceGraph()
deviceTopo.generata_fat_tree_topo(2, 30, 20, 1)

# Init solver
model = Model("minimize_maxload")
model.setParam("LogToConsole", 0)
model.setParam("LogFile", "gurobi.log")
model.setParam("MIPGap", 0.01)
model.setParam("TimeLimit", 2400)
model.setParam("MIPFocus", 1)

# if this is too large, then the reformulated
# ex-quadratic constraints can behave funky
model.setParam("IntFeasTol", 1e-6)
model.setParam("MemLimit", 4096)  # Example: Limit memory usage to 4 GB
model.setParam("Threads", 4)  # Example: Use 4 threads

# Define variables
x = {}  # key will be (operator_id, machine_id), value will be 1 or 0; x[3, 1] = 1 means operator 3 get allocated to device 1
start = {}  # start[node_id] represent the starting time of this node
finish = {}  # finish[node_id] represent the finish time of this node
comm_active = {}  # comm_active[sourceDeviceID, destDeviceID] represent the communication
M = 1e9  # A sufficiently large number

# Initialize all variables with names
for node_id in comp_graph.getOperatorIDs():
    for machine_id in deviceTopo.getDeviceIDs():
        x[node_id, machine_id] = model.addVar(vtype=GRB.BINARY, name=f"x_{node_id}_{machine_id}")
    start[node_id] = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"start_{node_id}")
    finish[node_id] = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"finish_{node_id}")

# Add constraints that schedule every node on exactly one machine
for op in comp_graph.getOperatorIDs():
    model.addConstr(quicksum(x[op, device] for device in deviceTopo.getDeviceIDs()) == 1, name=f"one_device_{op}")

# Add constraints that operators assigned cannot exceed the capacity
for machine_id in deviceTopo.getDeviceIDs():
    mem_sum = LinExpr()
    for node_id in comp_graph.getOperatorIDs():
        mem_sum += x[node_id, machine_id] * comp_graph.getOperator(node_id)["mem"]
    model.addConstr(mem_sum <= deviceTopo.getDevice(machine_id)["memory_capacity"],
                    "satisfy each device's memory constraint")

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
# unit_comm_costs[device_id_src, device_id_dest] means the com cost per bit from device with source device to dest device
unit_comm_costs = {}
for device_id_src in deviceTopo.getDeviceIDs():
    for device_id_dest in deviceTopo.getDeviceIDs():
        unit_comm_costs[device_id_src, device_id_dest] = deviceTopo.calUnitCommCostInUS(device_id_src, device_id_dest)
for edge_id_tuple in list(comp_graph.getEdgeIDs()):
    # https://support.gurobi.com/hc/en-us/articles/360039628832-Constraint-has-no-bool-value-are-you-trying-lb-expr-ub
    # https://support.gurobi.com/hc/en-us/community/posts/360077951791-if-statement-in-constraint
    source_op_ID, dest_op_ID = edge_id_tuple
    shape, dtype = comp_graph.getOperatorOutputSizeAndType(source_op_ID)
    tensor_size = tensor_shape_to_bits(shape, dtype=dtype)
    comm_cost = model.addVar(vtype=GRB.CONTINUOUS, name=f"comm_cost_{source_op_ID}_{dest_op_ID}")

    # Aggregate communication cost
    comm_cost_expr = quicksum(
        unit_comm_costs[device_id_src, device_id_dest] * tensor_size * x[source_op_ID, device_id_src] * x[
            dest_op_ID, device_id_dest]
        for device_id_src in deviceTopo.getDeviceIDs()
        for device_id_dest in deviceTopo.getDeviceIDs()
        if device_id_src != device_id_dest
    )

    model.addConstr(comm_cost == comm_cost_expr, f"comm_cost_{source_op_ID}_{dest_op_ID}")

    # Add the data dependency constraint with communication cost
    model.addConstr(start[dest_op_ID] >= finish[source_op_ID] + comm_cost,
                    f"data_dependency_{source_op_ID}_{dest_op_ID}")

# Add constraint to ensure each device processes only one operator at a time. This is a SCHEDULING problem
operator_ids = comp_graph.getOperatorIDs()
for device in deviceTopo.getDeviceIDs():
    # ensures that each pair of operations is only considered once
    for i in range(len(operator_ids)):
        for j in range(i + 1, len(operator_ids)):
            op1 = operator_ids[i]
            op2 = operator_ids[j]
            # Create auxiliary binary variables
            y1 = model.addVar(vtype=GRB.BINARY, name=f"y1_{device}_{op1}_{op2}")
            y2 = model.addVar(vtype=GRB.BINARY, name=f"y2_{device}_{op1}_{op2}")
            model.addGenConstrIndicator(y1, True, finish[op1] <= start[op2])
            model.addGenConstrIndicator(y2, True, finish[op2] <= start[op1])

            # If on the same device, ensure that the operators do not overlap
            model.addConstr(y1 + y2 >= x[op1, device] + x[op2, device] - 1, name=f"non_overlap_{op1}_{op2}_{device}")

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
    print('Runtime = ', "%.2f" % model.Runtime, 's', sep='')
    print('Expected Traning time = ', TotalLatency.X, 's', sep='')
    # Assuming `start` and `finish` are dictionaries holding start and end times for each operator
    result = {'totalLatency': model.ObjVal, 'Assignment': {}, 'CommunicationCosts': []}

    for key, value in x.items():
        # key[1] is the device id
        if key[1] not in result['Assignment']:
            result['Assignment'][key[1]] = []
        # key[0] is the operator id. Put id into the list assigned to the device
        if value.X > 0.5:
            # Assignment: {device: [(op1, start[op1], finish[op1]), (...)]}
            result['Assignment'][key[1]].append((key[0], start[key[0]].X, finish[key[0]].X))

    # Sort operators by their start times for each device
    for device, ops in result['Assignment'].items():
        result['Assignment'][device] = sorted(ops, key=lambda x: x[1])

    # Extract communication costs
    for edge_id_tuple in list(comp_graph.getEdgeIDs()):
        source_op_ID, dest_op_ID = edge_id_tuple
        s_placement = None
        d_placement = None
        comm_cost_var = model.getVarByName(f"comm_cost_{source_op_ID}_{dest_op_ID}")
        if comm_cost_var:
            comm_cost = comm_cost_var.X
            shape, dtype = comp_graph.getOperatorOutputSizeAndType(source_op_ID)
            tensor_size = tensor_shape_to_bits(shape, dtype=dtype)
            for device, ops in result['Assignment'].items():
                if source_op_ID in [op[0] for op in ops]:
                    s_placement = device
                if dest_op_ID in [op[0] for op in ops]:
                    d_placement = device
                if s_placement and d_placement:
                    break
            # if both ops are placed on the same device, use 999 to represent that
            if s_placement == d_placement:
                bandwidth = 999
            else:
                bandwidth = deviceTopo.get_link_bandwidth(s_placement, d_placement)
            if comm_cost >= 0:  # Only include all communication costs to verify
                result['CommunicationCosts'].append(
                    (source_op_ID, s_placement, dest_op_ID, d_placement, comm_cost, tensor_size, bandwidth))

    # You can also format the output to display start and finish times more clearly
    for device, ops in result['Assignment'].items():
        print(f"Device: {device}")
        for op in ops:
            comp_cost = sum(x[op, device_id].X * comp_graph.getOperatorCompCostByDevice(op, device_id)
                            for device_id in deviceTopo.getDeviceIDs())
            print(f"  Operator: {op[0]}, Start: {op[1]}, Finish: {op[2]}, Comp Cost: {comp_cost}")

    # Print communication costs
    print("Communication Costs:")
    sum_of_communication = 0
    for source_op_ID, s_placement, dest_op_ID, d_placement, comm_cost, tensor_size, bandwidth in result['CommunicationCosts']:
        sum_of_communication += comm_cost
        print(
            f"  From {source_op_ID} with placement {s_placement} to {dest_op_ID} with placement {d_placement}, Cost: {comm_cost}, Tensor size: {tensor_size}, Bandwidth: {bandwidth} GB/s")
    print(f"Total Communication Cost: {sum_of_communication}")

    del model
    disposeDefaultEnv()
else:
    print(f"Optimization ended with status {model.status}")
