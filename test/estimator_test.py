# python3 estimator.py
import json

from gurobipy import *
import torch
import tensorflow as tf

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.append(project_root)
from DNN_model_tf.vgg_tf import VGG16_tf
from optimizer.computing_graph.computing_graph import get_computation_graph
from optimizer.device_topo.device_graph import get_device_topo_ssh
from optimizer.cluster_info import servers
from py_util import tensor_shape_to_bits
from optimizer.model.graph import DeviceGraph


model = VGG16_tf()
comp_graph = get_computation_graph(model=model)
deviceTopo = DeviceGraph()

# init fake data
comp_graph.generata_random_cost(2)
deviceTopo.generata_fat_tree_topo(2, 30, 20, 1)

# Init solver
model = Model("minimize_maxload")
model.setParam("LogToConsole", 0)
model.setParam("LogFile", "gurobi.log")
model.setParam("MIPGap", 0.01)
model.setParam("TimeLimit", 1200)
model.setParam("MIPFocus", 1)
model.setParam(GRB.Param.Threads, 12)  # Use a single thread
model.setParam(GRB.Param.MemLimit, 4096)  # Limit memory usage to 1GB

# if this is too large, then the reformulated
# ex-quadratic constraints can behave funky
model.setParam("IntFeasTol", 1e-6)

# Define variables
x = {}  # key will be (operator_id, machine_id), value will be 1 or 0; x[3, 1] = 1 means operator 3 get allocated to device 1
x1 = model.addVar(vtype=GRB.BINARY, name="w1")
x2 = model.addVar(vtype=GRB.BINARY, name="w2")
for node_id in comp_graph.getOperatorIDs():
    for machine_id in deviceTopo.getDeviceIDs():
        x[node_id, machine_id] = model.addVar(vtype=GRB.BINARY)

# Add constraints that schedule every node on exactly one machine
for node_id in comp_graph.getOperatorIDs():
    times_scheduled = LinExpr()
    for machine_id in deviceTopo.getDeviceIDs():
        times_scheduled += x[node_id, machine_id]
    model.addConstr(times_scheduled == 1,"every node on exactly one machine")

# Add constraints that operators assigned cannot exceed the capacity
for machine_id in deviceTopo.getDeviceIDs():
    mem_sum = LinExpr()
    for node_id in comp_graph.getOperatorIDs():
        mem_sum += x[node_id, machine_id] * comp_graph.getOperator(node_id)["mem"]
    model.addConstr(mem_sum <= deviceTopo.getDevice(machine_id)["memory_capacity"], "satisfy each device's memory constraint")

# Add constraints that each device should have at least one operator assigned
for machine_id in deviceTopo.getDeviceIDs():
    op_count = LinExpr()
    for node_id in comp_graph.getOperatorIDs():
        op_count += x[node_id, machine_id]
    model.addConstr(op_count >= 1, "each device should have at least one op")

# Add constraints that later operator cannot begin before all previous ones finish computing and transmission
start = {}
finish = {}
for node_id in list(comp_graph.getOperatorIDs()):
    start[node_id] = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0)
    finish[node_id] = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0)
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
for edge_id_tuple in list(comp_graph.getEdgeIDs()):
    sourceID = edge_id_tuple[0]
    destID = edge_id_tuple[1]
    tensor_size = tensor_shape_to_bits(comp_graph.getOperator(sourceID)["output_size"], dtype=tf.float32)
    source_placement = model.addVar(vtype=GRB.INTEGER, name="w1")
    dest_placement = model.addVar(vtype=GRB.INTEGER, name="w1")
    # https://support.gurobi.com/hc/en-us/articles/360039628832-Constraint-has-no-bool-value-are-you-trying-lb-expr-ub
    # https://support.gurobi.com/hc/en-us/community/posts/360077951791-if-statement-in-constraint
    # Enforce that source_placement and dest_placement match the binary variables
    # communication_costs[idx_src, idx_dest] means the com cost from device with int id idx_src to another with int id idx_dest
    communication_costs = {}
    for device_id_src, idx_src in device_id_mapping.items():
        for device_id_dest, idx_dest in device_id_mapping.items():
            communication_costs[idx_src, idx_dest] = deviceTopo.calculateCommunicationCost(tensor_size, device_id_src, device_id_dest)

    # Add constraints to link communication costs to source and destination placements
    comm_cost = model.addVar(vtype=GRB.CONTINUOUS, name=f"comm_cost_{sourceID}_{destID}")

    for idx_src in device_id_mapping.values():
        for idx_dest in device_id_mapping.values():
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
            model.addGenConstrIndicator(and_var, True, comm_cost == communication_costs[idx_src, idx_dest])

    # Add the data dependency constraint with communication cost
    model.addConstr(start[destID] >= finish[sourceID] + comm_cost, f"data_dependency_{sourceID}_{destID}")

# Add constraint to ensure each device processes only one operator at a time
for device in deviceTopo.getDeviceIDs():
    for op1 in comp_graph.getOperatorIDs():
        for op2 in comp_graph.getOperatorIDs():
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
                # not_both will be true when finish[op1] <= start[op2] OR finish[op2] <= start[op1]
                model.addGenConstrOr(not_overlap, [y1, y2])

                # If on the same device, ensure that the operators do not overlap
                model.addGenConstrIndicator(same_device, True, not_overlap == 1)


# TotalLatency that we are minimizing
TotalLatency = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0)
for op_end in finish.values():
    model.addConstr(TotalLatency >= op_end, "satisfy each deice's latency")

# Set the target of solver
model.setObjective(TotalLatency, GRB.MINIMIZE)

# Run the solver
sys.stdout.flush()
model.optimize()

if model.Status == GRB.Status.INFEASIBLE:
    raise "infeasible"
elif model.Status == GRB.Status.OPTIMAL:
    print("Value is:", TotalLatency.X)
else:
    raise "Wrong status code"

print('Runtime = ', "%.2f" % model.Runtime, 's', sep='')
#populate the result dict
result = {'totalLatency': TotalLatency.X, 'Assignment': {}}
for key, value in x.items():
    # key[1] is the device id
    if key[1] not in result['Assignment']:
        result['Assignment'][key[1]] = []
    # key[0] is the operator id. Put id into the list assigned to the device
    if value.X > 0.99:
        result['Assignment'][key[1]].append(key[0])

del model
disposeDefaultEnv()
print(json.dumps(result))
