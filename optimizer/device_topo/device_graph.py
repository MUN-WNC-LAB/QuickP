from optimizer.device_topo.intel_node_util import update_intra_graph_with_intel, \
    ssh_intel_2_dict
from optimizer.device_topo.intra_node_util import ssh_intra_2_DiGraphs
from optimizer.host_ip import host_ip_mapping
from optimizer.model.graph import DeviceGraph, combine_graphs, visualize_graph
from optimizer.ssh_parallel import execute_parallel, ParallelCommandType


def get_device_topo_ssh():
    if len(host_ip_mapping) <= 0:
        raise ValueError("No available nodes in Slurm to run the job.")

    output_intra = execute_parallel([], ParallelCommandType.INTRA_NODE)
    output_intel = execute_parallel([], ParallelCommandType.INTER_NODE)
    if output_intra:
        graph_list_intra = ssh_intra_2_DiGraphs(output_intra)
        dict_list_intel = ssh_intel_2_dict(output_intel)
        graph_combined = combine_graphs(graph_list_intra)
        update_intra_graph_with_intel(graph_combined, dict_list_intel)
        visualize_graph(graph_combined)
    else:
        raise ValueError("No available nodes in Slurm to run the job.")
