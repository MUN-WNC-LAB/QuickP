# python3 estimator_test.py

from gurobipy import *
import torch
import tensorflow as tf

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.append(project_root)
from optimizer.computing_graph.computing_graph import get_computation_graph
from py_util import tensor_shape_to_bits, convert_time
from optimizer.model.graph import DeviceGraph
from DNN_model_tf.small import small_tf

model = small_tf()
comp_graph = get_computation_graph(model=model)
deviceTopo = DeviceGraph()

# init fake data
comp_graph.generata_random_cost(2)
deviceTopo.generata_fat_tree_topo(2, 30, 20, 1)

# Init solver
model = Model("minimize_maxload")
model.setParam("LogToConsole", 0)
model.setParam("LogFile", "gurobi.log")
model.setParam("MIPGap", 0.05)
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
comm_active = {}  # comm_active[sourceDeviceID, destDeviceID] represent the communicati
order = {}
M = 1e9  # A sufficiently large number

# init all variables
for node_id in comp_graph.getOperatorIDs():
    for machine_id in deviceTopo.getDeviceIDs():
        x[node_id, machine_id] = model.addVar(vtype=GRB.BINARY)
        order[node_id, machine_id] = model.addVar(vtype=GRB.INTEGER, name=f"order_{node_id}_{machine_id}")
    start[node_id] = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0)
    finish[node_id] = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0)

# Add constraints that schedule every node on exactly one machine
for node_id in comp_graph.getOperatorIDs():
    times_scheduled = LinExpr()
    for machine_id in deviceTopo.getDeviceIDs():
        times_scheduled += x[node_id, machine_id]
    model.addConstr(times_scheduled == 1, "every node on exactly one machine")

# Add constraints that operators assigned cannot exceed the capacity
for machine_id in deviceTopo.getDeviceIDs():
    mem_sum = LinExpr()
    for node_id in comp_graph.getOperatorIDs():
        mem_sum += x[node_id, machine_id] * comp_graph.getOperator(node_id)["mem"]
    model.addConstr(mem_sum <= deviceTopo.getDevice(machine_id)["memory_capacity"],
                    "satisfy each device's memory constraint")

# Add constraints that each device should have at least one operator assigned
for machine_id in deviceTopo.getDeviceIDs():
    op_count = LinExpr()
    for node_id in comp_graph.getOperatorIDs():
        op_count += x[node_id, machine_id]
    model.addConstr(op_count >= 1, "each device should have at least one op")

# Add constraints that each op's ending time = starting time + its computing time
for node_id in list(comp_graph.getOperatorIDs()):
    comp_cost = LinExpr()
    # since there is one placement, only one x[node_id, device_id] will be 1
    for device_id in deviceTopo.getDeviceIDs():
        # consider the device heterogeneity
        comp_cost += x[node_id, device_id] * comp_graph.getOperator(node_id)["comp_cost"][device_id]
    model.addConstr(finish[node_id] == start[node_id] + comp_cost, "finish == start + process")

# Add constraint that if op2 depends on op1, the starting time of op2 will be the ending time of op1 + communication delay if these two ops are not placed on the same device
# since device id are str, map them to integers
device_id_mapping: dict[str, int] = {device_id: idx for idx, device_id in enumerate(deviceTopo.getDeviceIDs())}
unit_comm_costs = {}
for device_id_src, idx_src in device_id_mapping.items():
    for device_id_dest, idx_dest in device_id_mapping.items():
        unit_comm_costs[idx_src, idx_dest] = deviceTopo.calUnitCommCostInUS(device_id_src, device_id_dest)
for edge_id_tuple in list(comp_graph.getEdgeIDs()):
    # https://support.gurobi.com/hc/en-us/articles/360039628832-Constraint-has-no-bool-value-are-you-trying-lb-expr-ub
    # https://support.gurobi.com/hc/en-us/community/posts/360077951791-if-statement-in-constraint
    sourceID, destID = edge_id_tuple
    tensor_size = tensor_shape_to_bits(comp_graph.getOperatorOutputSize(sourceID), dtype=tf.float32)
    source_placement = model.addVar(vtype=GRB.INTEGER, name="w1")
    dest_placement = model.addVar(vtype=GRB.INTEGER, name="w1")
    # communication_costs[idx_src, idx_dest] means the com cost from device with int id idx_src to another with int id idx_dest
    comm_cost = model.addVar(vtype=GRB.CONTINUOUS, name=f"comm_cost_{sourceID}_{destID}")

    for idx_src in device_id_mapping.values():
        for idx_dest in device_id_mapping.values():
            # if source device is the same as the dest device, the communication cost
            if idx_src == idx_dest:
                comm_cost_src_dest = 0
            else:
                comm_cost_src_dest = unit_comm_costs[idx_src, idx_dest] * tensor_size
            # Create auxiliary binary variables for the conditions
            source_cond = model.addVar(vtype=GRB.BINARY, name=f"source_cond_{sourceID}_{idx_src}")
            dest_cond = model.addVar(vtype=GRB.BINARY, name=f"dest_cond_{destID}_{idx_dest}")
            # Add constraints to enforce these conditions
            # When source_cond == True(1), source_placement == int id of source device
            model.addGenConstrIndicator(source_cond, True, source_placement == idx_src)
            # When dest_cond == True(1), dest_placement == int id of destination device
            model.addGenConstrIndicator(dest_cond, True, dest_placement == idx_dest)
            # Create the AND variable
            and_var = model.addVar(vtype=GRB.BINARY, name=f"and_{sourceID}_{destID}_{idx_src}_{idx_dest}")
            # When source_cond == 1 and dest_cond == 1, and_var == 1
            model.addGenConstrAnd(and_var, [source_cond, dest_cond])
            model.addGenConstrIndicator(and_var, True, comm_cost == comm_cost_src_dest)

    # Add the data dependency constraint with communication cost
    model.addConstr(start[destID] >= finish[sourceID] + comm_cost, f"data_dependency_{sourceID}_{destID}")

# Add constraint to ensure each device processes only one operator at a time. This is a SCHEDULING problem
for device in deviceTopo.getDeviceIDs():
    op_ids = comp_graph.getOperatorIDs()
    # ensures that each pair of operations is only considered once
    for i in range(len(op_ids)):
        for j in range(i + 1, len(op_ids)):
            op1 = op_ids[i]
            op2 = op_ids[j]
            if op1 != op2:
                # Add AND constraint to check if both ops are on the same device
                same_device = model.addVar(vtype=GRB.BINARY, name=f"same_device_{device}_{op1}_{op2}")
                model.addGenConstrAnd(same_device, [x[op1, device], x[op2, device]])

                # Create auxiliary binary variables
                y1 = model.addVar(vtype=GRB.BINARY, name=f"y1_{device}_{op1}_{op2}")
                y2 = model.addVar(vtype=GRB.BINARY, name=f"y2_{device}_{op1}_{op2}")
                not_overlap = model.addVar(vtype=GRB.BINARY, name=f"not_both_{device}_{op1}_{op2}")
                model.addGenConstrIndicator(y1, True, finish[op1] <= start[op2])
                model.addGenConstrIndicator(y2, True, finish[op2] <= start[op1])

                # If on the same device, ensure that the operators do not overlap
                model.addGenConstrIndicator(same_device, True, y1 + y2 == 1)

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
        if value.X > 0.99:
            result['Assignment'][key[1]].append((key[0], start[key[0]].X, finish[key[0]].X))

    # Sort operators by their start times for each device
    for device, ops in result['Assignment'].items():
        result['Assignment'][device] = sorted(ops, key=lambda x: x[1])

    # Extract communication costs
    for edge_id_tuple in list(comp_graph.getEdgeIDs()):
        sourceID, destID = edge_id_tuple
        s_placement = None
        d_placement = None
        comm_cost_var = model.getVarByName(f"comm_cost_{sourceID}_{destID}")
        if comm_cost_var:
            comm_cost = comm_cost_var.X
            tensor_size = tensor_shape_to_bits(comp_graph.getOperatorOutputSize(sourceID), dtype=tf.float32)
            for device, ops in result['Assignment'].items():
                if sourceID in [op[0] for op in ops]:
                    s_placement = device
                if destID in [op[0] for op in ops]:
                    d_placement = device
                if s_placement and d_placement:
                    break
            # if both ops are placed on the same device, use 999 to represent that
            if s_placement == d_placement:
                bandwidth = 999
            else:
                bandwidth = deviceTopo.get_link_bandwidth(s_placement, d_placement)
            if comm_cost > 0:  # Only include non-zero communication costs
                result['CommunicationCosts'].append((sourceID, destID, comm_cost, tensor_size, bandwidth))

    # You can also format the output to display start and finish times more clearly
    for device, ops in result['Assignment'].items():
        print(f"Device: {device}")
        for op in ops:
            print(f"  Operator: {op[0]}, Start: {op[1]}, Finish: {op[2]}")

    # Print communication costs
    print("Communication Costs:")
    for sourceID, destID, comm_cost, tensor_size, bandwidth in result['CommunicationCosts']:
        print(f"  From {sourceID} to {destID}, Cost: {comm_cost}, Tensor size: {tensor_size}, Bandwidth: {bandwidth} GB/s")

    del model
    disposeDefaultEnv()
else:
    print(f"Optimization ended with status {model.status}")
