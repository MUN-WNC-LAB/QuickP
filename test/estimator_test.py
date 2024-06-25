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
comp_graph.generata_random_cost(4)
deviceTopo.generata_fat_tree_topo(4, 30, 20, 2)

# Init solver
model = Model("minimize_maxload")
model.setParam("LogToConsole", 0)
model.setParam("LogFile", "gurobi.log")
model.setParam("MIPGap", 0.01)
model.setParam("TimeLimit", 1200)
model.setParam("MIPFocus", 1)

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
    for device_id, device_id_int in device_id_mapping.items():
        model.addConstr((x[sourceID, device_id] == 1) >> (source_placement == device_id_int))
        model.addConstr((x[destID, device_id] == 1) >> (dest_placement == device_id_int))
    # Add auxiliary variables for communication costs
    communication_costs = {}
    for device_id_src, idx_src in device_id_mapping.items():
        for device_id_dest, idx_dest in device_id_mapping.items():
            comm_cost = deviceTopo.calculateCommunicationCost(tensor_size, idx_src, idx_dest, device_id_mapping)
            communication_costs[(idx_src, idx_dest)] = comm_cost

    # Add constraints to link communication costs to source and destination placements
    comm_cost = model.addVar(vtype=GRB.CONTINUOUS, name=f"comm_cost_{sourceID}_{destID}")

    for idx_src in device_id_mapping.values():
        for idx_dest in device_id_mapping.values():
            model.addConstr((source_placement == idx_src) & (dest_placement == idx_dest) >> (
                        comm_cost == communication_costs[(idx_src, idx_dest)]))

    # Add the data dependency constraint with communication cost
    model.addConstr(start[destID] >= finish[sourceID] + comm_cost, f"data_dependency_{sourceID}_{destID}")

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
