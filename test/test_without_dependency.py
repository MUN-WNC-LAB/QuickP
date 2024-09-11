import networkx as nx
from networkx import DiGraph


def get_op_related_subgraph_mapping(graph: DiGraph, operator_device_mapping, device_subgraph_mapping, edge_cut_list):
    def get_related_subgraph_num(node):
        device = operator_device_mapping[node]
        current_subgraph = device_subgraph_mapping.get(device)

        outgoing_edges = [(u, v) for u, v in edge_cut_list if operator_device_mapping.get(u) == device]
        related_devices = set()
        outgoing_edges_depended = [(u, v) for u, v in outgoing_edges
                                   if nx.has_path(graph, node, u)]
        for u, v in outgoing_edges_depended:
            assert device_subgraph_mapping.get(operator_device_mapping.get(u)) == current_subgraph
            assert operator_device_mapping.get(v) != device
            related_devices.add(operator_device_mapping.get(v))
        return len(related_devices)

    return {op: get_related_subgraph_num(op) for op in graph.nodes}


def test_get_op_related_subgraph_mapping():
    # Create a small test graph as a networkx DAG
    graph = nx.DiGraph()
    graph.add_edges_from([(1, 2), (2, 3), (3, 4), (4, 5), (2, 6), (6, 7)])

    # Operator to device mapping
    operator_device_mapping = {
        1: 'device_1',
        2: 'device_1',
        3: 'device_2',
        4: 'device_2',
        5: 'device_3',
        6: 'device_3',
        7: 'device_4'
    }

    # Device to subgraph mapping
    device_subgraph_mapping = {
        'device_1': 'subgraph_1',
        'device_2': 'subgraph_2',
        'device_3': 'subgraph_3',
        'device_4': 'subgraph_4'
    }

    # Edge cut list (edges between different devices)
    edge_cut_list = [(2, 3), (4, 5), (2, 6), (6, 7)]

    # Expected output for this test case
    expected_output = {
        1: 2,  # device_1 (operator 1) is related to subgraph_2 and 3
        2: 2,  # device_1 (operator 2) is related to subgraph_2 and 3
        3: 1,
        4: 1,
        5: 0,
        6: 1,
        7: 0
    }

    # Call the function to test
    result = get_op_related_subgraph_mapping(graph, operator_device_mapping, device_subgraph_mapping, edge_cut_list)

    # Compare the result with expected output
    assert result == expected_output, f"Test failed! Expected {expected_output}, got {result}"

    print("Test passed!")


# Run the test
test_get_op_related_subgraph_mapping()
