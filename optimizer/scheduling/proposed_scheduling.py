import networkx as nx
from gurobipy import Model, GRB

from optimizer.model.graph import CompGraph, find_non_connected_pairs, is_not_connected
from optimizer.scheduling.scheduling_order_only import FIFO_scheduling_order


def near_optimal_scheduling(model: Model, start, finish, comm_start, comm_end, comp_graph: CompGraph,
                            device_subgraph_mapping: dict, edge_cut_list: list, operator_device_mapping: dict, threshold):
    # The global data dependency is already applied
    M = 1000000
    order = {}
    fifo_operator_order, _ = FIFO_scheduling_order(comp_graph, device_subgraph_mapping, edge_cut_list,
                                                   operator_device_mapping)
    for device, subgraph in device_subgraph_mapping.items():
        non_connected_pairs = find_non_connected_pairs(subgraph)
        high_cost_pairs, other_pairs = split_non_connected_pairs(comp_graph, device, non_connected_pairs, threshold)
        other_nodes = set(node for pair in other_pairs for node in pair)
        # get the FIFO order
        local_fifo_order = fifo_operator_order[device]
        node_order_dict = {op: idx for idx, op in enumerate(local_fifo_order)}
        # sort other_nodes based on the FIFO order
        other_nodes = sorted(other_nodes, key=lambda op: node_order_dict[op])

        for op_a, op_b in high_cost_pairs:
            order[op_a, op_b] = model.addVar(vtype=GRB.BINARY, name=f"order_{op_a}_{op_b}")
            model.addConstr(start[op_b] >= finish[op_a] - M * (1 - order[op_a, op_b]), name=f"NoOverlap1_{op_a}_{op_b}")
            model.addConstr(start[op_a] >= finish[op_b] - M * order[op_a, op_b], name=f"NoOverlap2_{op_a}_{op_b}")
        for op_a, op_b in zip(other_nodes, other_nodes[1:]):
            model.addConstr(finish[op_a] <= start[op_b])


def split_non_connected_pairs(graph: CompGraph, device, non_connected_pairs, threshold):
    computing_cost = graph.getOpCompCostMapByDevice(device)
    # List to store pairs where both nodes have a computing cost > 5
    high_cost_pairs = []

    # List to store other pairs
    other_pairs = []

    for node_a, node_b in non_connected_pairs:
        # Check if both nodes have a computing cost higher than 5
        # must use or instead of and because for a selected node, we must get the entire order chain
        if computing_cost[node_a] > threshold or computing_cost[node_b] > threshold:
            high_cost_pairs.append((node_a, node_b))
        else:
            other_pairs.append((node_a, node_b))

    return high_cost_pairs, other_pairs
