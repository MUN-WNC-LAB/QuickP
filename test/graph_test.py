from DNN_model_tf.vgg_tf import VGG16_tf
from optimizer.computing_graph.computing_graph import get_computation_graph
from optimizer.model.graph import DeviceGraph, visualize_graph, CompGraph
import tensorflow as tf


def test_generata_fat_tree_topo():
    G = DeviceGraph()
    G.generata_fat_tree_topo(6, 50, 20, 2)
    visualize_graph(G, show_node_labels=False)


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
