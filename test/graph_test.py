from DNN_model_tf.vgg_tf import VGG16_tf
from optimizer.computing_graph.computing_graph import get_computation_graph
from optimizer.model.graph import DeviceGraph, visualize_graph, CompGraph
import tensorflow as tf

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


test_graph_json_conversion()
