import networkx as nx

from DNN_model_tf.vgg_tf import VGG16_tf
from optimizer.computing_graph.computing_graph import get_computation_graph
from optimizer.graph_partitioner.metis_partition import metis_partition
from optimizer.model.graph import DeviceGraph, visualize_graph, CompGraph, is_subgraph
import tensorflow as tf

from optimizer.graph_partitioner.model_split_min_communication import creates_cycle, split_DAG_min_inter_subgraph_edges
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


def test_graph_split():
    G_DAG = generate_linear_dag(15)
    G_DAG.add_edge(2, 19)
    G_DAG.add_edge(2, 20)
    G_DAG.add_edge(22, 19)
    G_DAG.add_edge(2, 21)
    G_DAG.add_edge(21, 22)
    G_DAG.add_edge(22, 23)
    visualize_graph(G_DAG)
    subgraphs = split_DAG_min_inter_subgraph_edges(G_DAG, 5)
    for subgraph in subgraphs:
        print(subgraph.nodes)
        visualize_graph(subgraph)


def test_metis_partition():
    comp_graph = CompGraph.load_from_file('../optimizer/comp_graph.json')
    metis_partition(comp_graph)


test_metis_partition()
