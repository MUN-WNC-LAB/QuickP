import networkx as nx
from gurobipy import Model, GRB

from optimizer.main_simulator.quicks.quicks_list_schedule import quicks_list_order_only
from optimizer.model.graph import CompGraph
from optimizer.scheduling.near_optimal_scheduling_with_sampling import split_nodes, get_device_unreachable_pairs_mapping


def sampling_based_near_optimal_schedule(model: Model, start, finish, comm_start, comm_end, comp_graph: CompGraph,
                                         device_relied_component_mapping: dict, edge_cut_list: list, operator_device_mapping: dict, heuristic_rank_map,
                                         computing_cost_dict, rho, sampling_function):
    # The global data dependency is already applied
    M = 1000000
    order = {}
    heuristic_order, _ = quicks_list_order_only(comp_graph, device_relied_component_mapping, edge_cut_list,
                                                   operator_device_mapping, heuristic_rank_map)

    device_unreachable_pairs_mapping, global_set_with_nr = get_device_unreachable_pairs_mapping(device_relied_component_mapping)
    global_node_split_by_device = split_nodes(comp_graph, global_set_with_nr, list(device_relied_component_mapping.keys()), operator_device_mapping, r=rho,
                                              sampling_function=sampling_function)

    for device, non_iso_part in device_relied_component_mapping.items():

        # flatten the pairs to get all the non-repeated nodes, convert to list
        selected_nodes, other_nodes = global_node_split_by_device.get(device)["selected_list"], \
        global_node_split_by_device.get(device)["unselected_list"]

        # use the FIFO order to sort other_nodes;
        local_fifo_order = heuristic_order[device]
        node_order_dict = {op: idx for idx, op in enumerate(local_fifo_order)}
        # sort other_nodes based on the FIFO order
        other_nodes = sorted(other_nodes, key=lambda op: node_order_dict[op])
        # Apply sequential constraint to non-selected nodes
        for op_a, op_b in zip(other_nodes, other_nodes[1:]):
            model.addConstr(finish[op_a] <= start[op_b])

        # apply optimization to all pairs involving these high-cost nodes, these pair also include the unselected nodes,
        # To optimize one node, all of its non-reachable node should be included
        important_pairs = [pair for pair in device_unreachable_pairs_mapping[device] if
                           pair[0] in selected_nodes or pair[1] in selected_nodes]
        for op_a, op_b in important_pairs:
            current_subgraph_dc = device_relied_component_mapping[device]
            assert op_a in current_subgraph_dc.nodes and op_b in current_subgraph_dc.nodes
            assert not nx.has_path(current_subgraph_dc, op_a, op_b)
            order[op_a, op_b] = model.addVar(vtype=GRB.BINARY, name=f"order_{op_a}_{op_b}")
            model.addConstr(start[op_b] >= finish[op_a] - M * (1 - order[op_a, op_b]), name=f"NoOverlap1_{op_a}_{op_b}")
            model.addConstr(start[op_a] >= finish[op_b] - M * order[op_a, op_b], name=f"NoOverlap2_{op_a}_{op_b}")