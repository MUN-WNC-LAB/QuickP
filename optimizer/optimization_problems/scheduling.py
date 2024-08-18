import itertools
from collections import deque
from enum import Enum
from itertools import combinations
from queue import PriorityQueue

import networkx as nx
from gurobipy import Model, GRB

from optimizer.model.graph import find_non_connected_pairs, CompGraph


def add_topo_order_constraints(model, original_topo_list, x, device_ids, finish, start):
    M = 1000000
    # Iterate over topologically sorted nodes
    for a, b in itertools.combinations(original_topo_list, 2):
        # For each consecutive pair of operators, add a constraint for each device
        for device_id in device_ids:
            # Ensure the correct order for each potential device assignment
            # This constraint will only apply if both a and b are assigned to the same device
            model.addConstr(finish[a] <= start[b] + M * (2 - x[a, device_id] - x[b, device_id]),
                            name=f"bigM_topo_order_{a}_{b}_on_device_{device_id}")


def optimal_scheduling(model: Model, start, finish, comm_start, comm_end, comp_graph, subgraph_dict, edge_cut_list):
    for source_op_ID, dest_op_ID in comp_graph.getEdgeIDs():
        model.addConstr(finish[source_op_ID] <= start[dest_op_ID])
    M = 100000
    order = {}
    for subgraph in subgraph_dict.values():
        non_connected_pairs = find_non_connected_pairs(subgraph)
        for op_a, op_b in non_connected_pairs:
            order[op_a, op_b] = model.addVar(vtype=GRB.BINARY, name=f"order_{op_a}_{op_b}")
            model.addConstr(start[op_b] >= finish[op_a] - M * (1 - order[op_a, op_b]), name=f"NoOverlap1_{op_a}_{op_b}")
            model.addConstr(start[op_a] >= finish[op_b] - M * order[op_a, op_b], name=f"NoOverlap2_{op_a}_{op_b}")

    # Add constraint to ensure each device can only send or receive from one link at a time, communication scheduling
    # Only edges in the edge_cut_list will bring communication cost
    order_link = {}
    for communication_a, communication_b in combinations(edge_cut_list, 2):
        order_link[communication_a, communication_b] = model.addVar(vtype=GRB.BINARY)
        model.addConstr(comm_start[communication_b] >= comm_end[communication_a] - M * (1 - order_link[communication_a, communication_b]))
        model.addConstr(comm_start[communication_a] >= comm_end[communication_b] - M * order_link[communication_a, communication_b])


def FIFO_scheduling(model: Model, start, finish, comm_start, comm_end, comp_graph: CompGraph,
                    device_subgraph_mapping: dict, edge_cut_list: list, operator_device_mapping: dict):

    def initialize_queues(subgraph_dict, dependency_graph):
        # Initialize a queue for each subgraph (device)
        device_queue_dict = {device: deque() for device, subgraph in subgraph_dict.items()}

        # Initialize with tasks that have no predecessors in the global graph
        for subgraph_id, subgraph in subgraph_dict.items():
            for operator_id in subgraph.nodes():
                # Check if the node has no predecessors in the global dependency graph
                global_predecessors = list(dependency_graph.predecessors(operator_id))

                # If the node has no predecessors in the global graph, it can be added to the queue
                if not global_predecessors:
                    # Add to the appropriate subgraph's queue
                    device_queue_dict[subgraph_id].append(operator_id)

        return device_queue_dict

    def update_queue(device_queues, finished_task, dependency_graph, completed_tasks, partition_dict):
        # Check all successors of the finished task in the global dependency graph
        successors = list(dependency_graph.successors(finished_task))
        for succ in successors:
            # Check if all predecessors are complete in the global dependency graph
            predecessors = list(dependency_graph.predecessors(succ))
            if all(predecessor in completed_tasks for predecessor in predecessors):
                # Enqueue the task to the task queue of this subgraph (device)
                subgraph_of_succ = partition_dict[succ]
                if subgraph_of_succ != partition_dict[finished_task]:
                    print(f"succ {succ} belongs to {subgraph_of_succ} while the current graph is {partition_dict[finished_task]}")
                # cannot use "if subgraph_of_succ" since subgraph id can be 0
                if subgraph_of_succ is not None:
                    # Enqueue the task to the task queue of the correct subgraph (device)
                    device_queues[subgraph_of_succ].append(succ)

    ready = model.addVars(comp_graph.getOperatorIDs(), vtype=GRB.CONTINUOUS, lb=0.0,
                          name="ready")  # ready[node_id] represent the ready time of this node, simulating Queue

    for node in comp_graph.nodes():
        for predecessor in comp_graph.predecessors(node):
            model.addConstr(ready[node] >= finish[predecessor], name=f"fifo_{predecessor}_to_{node}")

    # It is an SCHEDULING problem within each device.
    device_queues = initialize_queues(device_subgraph_mapping, comp_graph)
    total_items = sum(len(queue) for queue in device_queues.values())
    print("len of the init: ", total_items, 'The init device_queues is ', device_queues)

    # Initialize the set to track completed tasks
    completed_tasks = set()

    # This list will store all the constraints that we batch before optimization
    last_job_dict = {subgraph_id: None for subgraph_id in device_subgraph_mapping.keys()}
    last_communication_dict = {subgraph_id: None for subgraph_id in device_subgraph_mapping.keys()}
    # Process each subgraph independently
    while any(queue for queue in device_queues.values()):
        for subgraph_id, queue in device_queues.items():
            if queue:
                # Get the next task to execute for this subgraph
                current_op = queue.popleft()

                # check if this task get completed
                if current_op in completed_tasks:
                    raise ValueError("this is a repeated task")

                # check if all dependency satisfy
                for predecessor in nx.ancestors(comp_graph, current_op):
                    if predecessor not in completed_tasks:
                        raise ValueError(f"{current_op} 's dependency {predecessor} not satisfied")

                # Ensure the task starts after its ready time
                model.addConstr(start[current_op] >= ready[current_op],
                                name=f"start_after_ready_{current_op}_on_subgraph_{subgraph_id}")

                # Ensure that the task starts after the previous task finishes within the same subgraph
                # Operator scheduling within device
                if last_job_dict[subgraph_id] is not None:
                    model.addConstr(start[current_op] >= finish[last_job_dict[subgraph_id]], name=f"start_after_prev_finish_{current_op}_on_subgraph_{subgraph_id}")

                # Communication Scheduling. One device can only send or receive from up to one link at the same time
                for predecessor in comp_graph.predecessors(current_op):
                    if (predecessor, current_op) in edge_cut_list:
                        if last_communication_dict[subgraph_id] is not None:
                            model.addConstr(comm_start[predecessor, current_op] >= comm_end[last_communication_dict[subgraph_id]])
                        source_subgraph = operator_device_mapping[predecessor]
                        last_communication_dict[source_subgraph] = (predecessor, current_op)
                        last_communication_dict[subgraph_id] = (predecessor, current_op)

                # Track the finish time of the current task
                last_job_dict[subgraph_id] = current_op

                # Track task completion
                completed_tasks.add(current_op)

                # Update the queue based on the completion of the task
                update_queue(device_queues, current_op, comp_graph, completed_tasks, operator_device_mapping)

    # Get the collection of nodes that are in the graph but not in completed_tasks
    all_nodes = set(comp_graph.nodes())
    remaining_nodes = all_nodes - completed_tasks
    assert len(remaining_nodes) == 0, f"the remaining nodes {remaining_nodes} but all nodes should be scheduled"


def priority_queue_scheduling(model: Model, start, finish, comm_start, comm_end, comp_graph: CompGraph,
                              device_subgraph_mapping: dict, edge_cut_list: list, operator_device_mapping: dict):

    def initialize_queues(subgraph_dict, dependency_graph) -> dict[any, PriorityQueue]:
        # Initialize a queue for each device
        device_queue_dict = {device: PriorityQueue() for device, subgraph in subgraph_dict.items()}

        # Initialize with tasks that have no predecessors in the global graph
        for device, subgraph in subgraph_dict.items():
            for operator_id in subgraph.nodes():
                # Check if the node has no predecessors in the global dependency graph
                global_predecessors = list(dependency_graph.predecessors(operator_id))

                # If the node has no predecessors in the global graph, it can be added to the queue
                if not global_predecessors:
                    # Add to the appropriate subgraph's queue
                    device_queue_dict[device].put((comp_graph.getOperatorCompCostByDevice(operator_id, device), operator_id))

        return device_queue_dict

    def update_queue(device_queue_dict: dict[any, PriorityQueue], finished_task, dependency_graph, completed_tasks,
                     partition_dict):
        # Check all successors of the finished task in the global dependency graph
        successors = list(dependency_graph.successors(finished_task))
        for successor in successors:
            # Check if all predecessors are complete in the global dependency graph
            predecessors = list(dependency_graph.predecessors(successor))
            if all(predecessor in completed_tasks for predecessor in predecessors):
                # Enqueue the task to the task queue of this subgraph (device)
                device_of_successor = partition_dict[successor]
                if device_of_successor != partition_dict[finished_task]:
                    print(
                        f"{successor} belongs to {device_of_successor} while the current graph is {partition_dict[finished_task]}")
                # cannot use "if subgraph_of_succ" since subgraph id can be 0
                if device_of_successor is not None:
                    # Enqueue the task to the task queue of the correct subgraph (device)
                    device_queue_dict[device_of_successor].put((comp_graph.getOperatorCompCostByDevice(successor, device_of_successor), successor))

    ready = model.addVars(comp_graph.getOperatorIDs(), vtype=GRB.CONTINUOUS, lb=0.0,
                          name="ready")  # ready[node_id] represent the ready time of this node, simulating Queue

    for node in comp_graph.nodes():
        for predecessor in comp_graph.predecessors(node):
            model.addConstr(ready[node] >= finish[predecessor], name=f"fifo_{predecessor}_to_{node}")

    # It is an SCHEDULING problem within each device.
    device_queues = initialize_queues(device_subgraph_mapping, comp_graph)
    total_items = sum(queue.qsize() for queue in device_queues.values())
    print("len of the init: ", total_items, 'The init device_queues is ', device_queues)

    # Initialize the set to track completed tasks
    completed_tasks = set()

    # This list will store all the constraints that we batch before optimization
    last_job_dict = {subgraph_id: None for subgraph_id in device_subgraph_mapping.keys()}
    last_communication_dict = {subgraph_id: None for subgraph_id in device_subgraph_mapping.keys()}
    # Process each subgraph independently
    while any(not pqueue.empty() for pqueue in device_queues.values()):
        for device_id, queue in device_queues.items():
            if not queue.empty():
                # Get the next task to execute for this subgraph
                _, task = queue.get()

                # check if this task get completed
                if task in completed_tasks:
                    raise ValueError("this is a repeated task")

                # check if all dependency satisfy
                for predecessor in nx.ancestors(comp_graph, task):
                    if predecessor not in completed_tasks:
                        raise ValueError(f"{task} 's dependency {predecessor} not satisfied")

                # Ensure the task starts after its ready time
                model.addConstr(start[task] >= ready[task],
                                name=f"start_after_ready_{task}_on_subgraph_{device_id}")

                # Ensure that the task starts after the previous task finishes within the same subgraph
                # Operator scheduling within device
                if last_job_dict[device_id] is not None:
                    model.addConstr(start[task] >= finish[last_job_dict[device_id]], name=f"start_after_prev_finish_{task}_on_subgraph_{device_id}")

                # Communication Scheduling. One device can only send or receive from up to one link at the same time
                for predecessor in comp_graph.predecessors(task):
                    if (predecessor, task) in edge_cut_list:
                        if last_communication_dict[device_id] is not None:
                            model.addConstr(comm_start[predecessor, task] >= comm_end[last_communication_dict[device_id]])
                        source_device = operator_device_mapping[predecessor]
                        last_communication_dict[source_device] = (predecessor, task)
                        last_communication_dict[device_id] = (predecessor, task)

                # Track the finish time of the current task
                last_job_dict[device_id] = task

                # Track task completion
                completed_tasks.add(task)

                # Update the queue based on the completion of the task
                update_queue(device_queues, task, comp_graph, completed_tasks, operator_device_mapping)

    # Get the collection of nodes that are in the graph but not in completed_tasks
    all_nodes = set(comp_graph.nodes())
    remaining_nodes = all_nodes - completed_tasks
    assert len(remaining_nodes) == 0, f"the remaining nodes {remaining_nodes} but all nodes should be scheduled"


class SchedulingAlgorithm(Enum):
    FIFO = "FIFO"
    OPTIMIZED = "OPTIMIZED"
    PRIORITY_QUEUE = "PRIORITY_QUEUE"


def execute_scheduling_function(sch_fun_type: str, model: Model, **kwargs):
    # Define the required arguments for each scheduling algorithm
    required_args = {
        SchedulingAlgorithm.FIFO.value: ['start', 'finish', 'comm_start', 'comm_end', 'comp_graph',
                                         'device_subgraph_mapping', 'edge_cut_list', 'operator_device_mapping'],
        SchedulingAlgorithm.OPTIMIZED.value: ['start', 'finish', 'comm_start', 'comm_end', 'comp_graph',
                                              'device_subgraph_mapping', 'edge_cut_list'],
        SchedulingAlgorithm.PRIORITY_QUEUE.value: ['start', 'finish', 'comm_start', 'comm_end', 'comp_graph',
                                                   'device_subgraph_mapping', 'edge_cut_list', 'operator_device_mapping']
    }

    if sch_fun_type not in required_args:
        raise ValueError(f"Unknown scheduling algorithm: {sch_fun_type}")

    # Check if all required arguments are provided
    missing_args = [arg for arg in required_args[sch_fun_type] if arg not in kwargs]
    if missing_args:
        raise ValueError(f"Missing arguments for {sch_fun_type}: {', '.join(missing_args)}")

    # Select the appropriate arguments for the scheduling function
    selected_kwargs = {key: kwargs[key] for key in required_args[sch_fun_type]}

    # Dynamically dispatch to the appropriate scheduling function
    if sch_fun_type == SchedulingAlgorithm.FIFO.value:
        return FIFO_scheduling(model, **selected_kwargs)
    elif sch_fun_type == SchedulingAlgorithm.OPTIMIZED.value:
        return optimal_scheduling(model, **selected_kwargs)
    elif sch_fun_type == SchedulingAlgorithm.PRIORITY_QUEUE.value:
        return priority_queue_scheduling(model, **selected_kwargs)
