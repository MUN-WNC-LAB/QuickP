import heapq
from collections import deque
from enum import Enum

from networkx import topological_sort


def topo_sort_Kahn(graph):
    # Compute the in-degree of each node
    in_degrees = {node: graph.in_degree(node) for node in graph.nodes()}

    # Queue of nodes with no incoming edges
    zero_in_degree_queue = deque([node for node in graph.nodes() if in_degrees[node] == 0])

    topo_order = []

    while zero_in_degree_queue:
        current = zero_in_degree_queue.popleft()
        topo_order.append(current)

        for neighbor in graph.successors(current):
            in_degrees[neighbor] -= 1
            if in_degrees[neighbor] == 0:
                zero_in_degree_queue.append(neighbor)
    assert is_topological_sort(graph, topo_order)
    return topo_order


def topo_sort_default(graph):
    return topological_sort(graph)


def topo_sort_Kahn_priority(graph, priority=None):
    if priority is None:
        # Default priority is 0 for all nodes if not specified
        priority = {node: 0 for node in graph.nodes()}

        # Compute the in-degree of each node
    in_degrees = {node: graph.in_degree(node) for node in graph.nodes()}

    # Priority queue of nodes with no incoming edges
    zero_in_degree_queue = [(priority[node], node) for node in graph.nodes() if in_degrees[node] == 0]
    heapq.heapify(zero_in_degree_queue)

    topo_order = []

    while zero_in_degree_queue:
        _, current = heapq.heappop(zero_in_degree_queue)
        topo_order.append(current)

        for neighbor in graph.successors(current):
            in_degrees[neighbor] -= 1
            if in_degrees[neighbor] == 0:
                heapq.heappush(zero_in_degree_queue, (priority[neighbor], neighbor))

    assert is_topological_sort(graph, topo_order)
    return topo_order


def is_topological_sort(graph, node_list):
    # Create a dictionary to record the position of each node in the node_list
    position = {node: i for i, node in enumerate(node_list)}

    # Iterate through all edges in the graph
    for u, v in graph.edges():
        # For a valid topological sort, u must come before v
        if position[u] > position[v]:
            return False

    return True


class TopoSortFunction(Enum):
    KAHN = topo_sort_Kahn
    DEFAULT = topo_sort_default
    KAHN_PRIORITY = topo_sort_Kahn_priority
