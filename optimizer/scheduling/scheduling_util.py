from typing import Tuple

import networkx as nx

from optimizer.model.graph import CompGraph

'''
Split a graph into three parts
Source Nodes: Nodes that serve as dependencies for other subgraphs.
Sink Nodes: Nodes that do not serve as dependencies for other subgraphs but have dependencies from other graphs. These can be seen as "end points."
Isolated Nodes: Nodes that neither serve as dependencies for other subgraphs nor have dependencies from other graphs. These are independent nodes within the graph structure.
'''


def split_subgraph(graph: CompGraph, operator_device_mapping, edge_cut_list) -> Tuple[CompGraph, CompGraph, set]:
    def get_depended_node_set(node):
        device = operator_device_mapping[node]
        outgoing_edges = [(u, v) for u, v in edge_cut_list if operator_device_mapping.get(u) == device]
        destination_node_depended = set(v for (u, v) in outgoing_edges if nx.has_path(graph, node, u))
        for node in destination_node_depended:
            assert operator_device_mapping.get(node) != device
        return destination_node_depended

    def get_depending_node_set(node):
        device = operator_device_mapping[node]
        incoming_edges = [(u, v) for u, v in edge_cut_list if operator_device_mapping.get(v) == device]
        source_node_depended = set(u for (u, v) in incoming_edges if nx.has_path(graph, v, node))
        for node in source_node_depended:
            assert operator_device_mapping.get(node) != device
        return source_node_depended

    # Create a copy of the original graph
    new_graph = graph.copy()

    # Iterate over the nodes and remove those with 0 related subgraphs
    non_source_node = set(node for node in graph.nodes if len(get_depended_node_set(node)) == 0)

    isolate_nodes = set(node for node in non_source_node if len(get_depending_node_set(node)) == 0)

    sink_nodes = non_source_node - isolate_nodes

    # Remove the nodes from the copied graph
    new_graph.remove_nodes_from(non_source_node)

    print('ff', len(non_source_node), len(isolate_nodes), len(sink_nodes))

    sink_components = graph.subgraph(sink_nodes)

    return new_graph, sink_components, isolate_nodes


def handle_sink_components(subgraph, sink_components: nx.DiGraph, device, operator_device_mapping, cut_off):
    weakly_connected_components = list(nx.weakly_connected_components(sink_components))
    sink_nodes = set(sink_components.nodes)
    incoming_nodes = set(v for u, v in cut_off if operator_device_mapping.get(v) == device)
    # node that directly connected with a cross device dependency
    joint_nodes = sink_nodes.intersection(incoming_nodes)
    other_nodes = sink_nodes - joint_nodes
    for weakly_connected_component in weakly_connected_components:
        weak_connected_subgraph = sink_components.subgraph(weakly_connected_component)
        topological_order = list(nx.topological_sort(weak_connected_subgraph))
        # start_node = list(weakly_connected_component)[0]  # Take the first node
        # assert topological_order[0] in incoming_nodes
