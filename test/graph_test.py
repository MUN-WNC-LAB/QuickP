from DNN_model_tf.vgg_tf import VGG16_tf
from optimizer.computing_graph.computing_graph import get_computation_graph
from optimizer.model.graph import DeviceGraph, visualize_graph

G = DeviceGraph()
G.generata_fat_tree_topo(15, 50, 20, 5)
visualize_graph(G)
