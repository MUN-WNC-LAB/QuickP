from collections import deque

import networkx as nx

from DNN_model_tf.vgg_tf import VGG16_tf
from optimizer.computing_graph.computing_graph import get_computation_graph
from optimizer.graph_partitioner.metis_partition import metis_partition
from optimizer.model.graph import DeviceGraph, visualize_graph, CompGraph, is_subgraph, keep_largest_component, \
    topo_order_until_node
import tensorflow as tf

from optimizer.graph_partitioner.subgraph_util import creates_cycle, construct_sub_graph
from py_util import convert_data_size, convert_time


def test_generata_fat_tree_topo():
    G = DeviceGraph()
    G.generata_fat_tree_topo(6, 50, 20, 2)
    visualize_graph(G, show_node_labels=False)
    print("is fully connected", G.is_fully_connected_bidirectional())


def test_generata_random_cost():
    model = VGG16_tf()
    comp_graph = get_computation_graph(model=model)
    comp_graph.generata_random_cost(6)
    print(comp_graph.getAllOperators())


def test_computation_graph():
    G = CompGraph()
    G.add_new_node(1, '', tf.TensorShape([2, 300]))
    G.add_new_node(2, '', tf.TensorShape([]))
    G.add_new_edge(1, 2)
    print(G.getEdgeIDs())
    for edge_id_tuple in list(G.getEdgeIDs()):
        print(type(edge_id_tuple))


def test_device_mapping():
    # Create mapping for devices to integers
    G = DeviceGraph()
    G.generata_fat_tree_topo(6, 50, 20, 2)
    device_id_mapping = {device_id: idx for idx, device_id in enumerate(G.getDeviceIDs())}
    print(device_id_mapping)


def test_conversion():
    print(convert_data_size(1, 'GB', 'bit'))
    print(convert_time(1, 's', 'us'))


def test_graph_json_conversion():
    # Create an example CompGraph
    G = CompGraph()
    G.add_node(1, label="A")
    G.add_node(2, label="B")
    G.add_edge(1, 2)

    # Save CompGraph to a file
    G.save_to_file('comp_graph.json')

    # Load CompGraph from the file
    loaded_graph = CompGraph.load_from_file('comp_graph.json')
    if loaded_graph:
        print("Loaded Graph:", loaded_graph.nodes(data=True), loaded_graph.edges(data=True))
    else:
        print("Failed to load graph.")


def generate_linear_dag(n):
    """
    Generate a fixed Directed Acyclic Graph (DAG) with n nodes.

    Parameters:
    n (int): Number of nodes in the graph.

    Returns:
    G (nx.DiGraph): A fixed DAG with n nodes. 1->2->3->4->5->...>n
    For example, list(G.successors(3)) => 4; G.predecessors(3) => 2
    """
    G = nx.DiGraph()

    # Add nodes to the graph
    G.add_nodes_from(range(1, n + 1))

    # Create a fixed pattern of edges
    # Example: Layered structure where each node connects to a few nodes in the next "layer"
    for i in range(1, n + 1):
        for j in range(i + 1, min(n + 1, i + 2)):  # Each node connects to the next 1 node
            G.add_edge(i, j)

    # Check if the generated graph is a DAG
    assert nx.is_directed_acyclic_graph(G), "The generated graph is not a DAG"

    return G


def test_linear_DAG():
    G = generate_linear_dag(5)
    print(list(G.successors(3)))
    print(list(G.predecessors(3)))


def test_if_create_cycle():
    G_DAG = generate_linear_dag(5)
    subG_DAG = generate_linear_dag(2)
    # Make a cycle
    cycle = G_DAG.copy()
    cycle.add_edge(3, 1)
    cycle.add_edge(6, 3)
    print("cycle is DAG", nx.is_directed_acyclic_graph(cycle))

    print("should create cycle ", creates_cycle(subG_DAG, 3, cycle))


def test_if_subgraph():
    G_DAG = generate_linear_dag(5)

    cycle = G_DAG.copy()
    cycle.add_edge(3, 1)

    subG_DAG = generate_linear_dag(3)
    # 1 → 2 → 3 is a subgraph of the larger graph 1 → 2 → 3 → 1.
    print(is_subgraph(subG_DAG, cycle))

    subG_DAG.add_edge(3, 2)
    print(is_subgraph(subG_DAG, cycle))


def test_metis_partition():
    comp_graph = CompGraph.load_from_file('../optimizer/comp_graph.json')
    metis_partition(comp_graph, num_partitions=3)


def test_metis_partition_subgraph_construction():
    comp_graph = CompGraph.load_from_file('comp_graph.json')
    # comp_graph = keep_largest_component(comp_graph)
    print('how many ops in total', len(comp_graph.nodes))
    partition_dict, edge_cut_list = metis_partition(comp_graph, num_partitions=3, visualization=True)
    subgraph_list = construct_sub_graph(comp_graph, partition_dict)
    for digraph in subgraph_list.values():
        print("after partitioning", len(digraph.nodes))
        visualize_graph(digraph, show_edge_labels=False, show_node_labels=False)


def test_topo_order_sequence():
    comp_graph = CompGraph.load_from_file('comp_graph.json')
    result = list(topo_order_until_node(comp_graph, 'gradient_tape/sequential_1/dense_1/Add/Shape'))
    print(result)


def create_topological_order_list(graph):
    # Compute the in-degree of each node
    in_degrees = {node: graph.in_degree(node) for node in graph.nodes()}

    # Queue of nodes with no incoming edges
    zero_in_degree_queue = deque([node for node in graph.nodes() if in_degrees[node] == 0])

    topo_order = []

    while zero_in_degree_queue:
        # Sort nodes in the queue based on in-degrees in ascending order
        zero_in_degree_queue = deque(sorted(zero_in_degree_queue, key=lambda node: in_degrees[node]))

        # Process the node with the lowest in-degree
        current = zero_in_degree_queue.popleft()
        topo_order.append(current)

        # Decrease the in-degree of neighboring nodes
        for neighbor in graph.successors(current):
            in_degrees[neighbor] -= 1
            if in_degrees[neighbor] == 0:
                zero_in_degree_queue.append(neighbor)

    return topo_order

# Create the graph
graph = nx.DiGraph()
graph.add_edges_from([("A", "B"), ("A", "C"), ("B", "D"), ("B", "E"), ("C", "F"),
                      ("D", "G"), ("E", "G"), ("F", "H"), ("F", "I"), ("G", "J"), ("H", "J"), ("I", "J")])

# Apply the function
topo_order = create_topological_order_list(graph)
print(topo_order)
print(list(nx.topological_sort(graph)))
