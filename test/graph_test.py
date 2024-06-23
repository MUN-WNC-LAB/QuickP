from DNN_model_tf.vgg_tf import VGG16_tf
from optimizer.computing_graph.computing_graph import get_computation_graph

model = VGG16_tf()
comp_graph = get_computation_graph(model=model)
comp_graph.generata_random_cost(50)

print(comp_graph.getAllOperators())
