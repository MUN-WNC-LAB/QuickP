from optimizer.model.graph import DeviceGraph, visualize_graph, CompGraph


def test_generata_fat_tree_topo():
    G = DeviceGraph()
    G.generata_fat_tree_topo(6, 50, 20, 2)
    visualize_graph(G, show_node_labels=False)


test_generata_fat_tree_topo()


def test_generata_fat_tree_topo():
    G = CompGraph()
    G.generata_random_cost(6)
    print(G.getAllOperators())


test_generata_fat_tree_topo()
