from DNN_model_tf.vgg_tf import VGG16_tf
from optimizer.computing_graph.computing_graph import get_computation_graph
from optimizer.model.graph import DeviceGraph, visualize_graph, CompGraph


def test_generata_fat_tree_topo():
    G = DeviceGraph()
    G.generata_fat_tree_topo(6, 50, 20, 2)
    visualize_graph(G, show_node_labels=False)


test_generata_fat_tree_topo()


def test_generata_random_cost():
    model = VGG16_tf()
    comp_graph = get_computation_graph(model=model)
    comp_graph.generata_random_cost(6)
    print(comp_graph.getAllOperators())


test_generata_random_cost()
