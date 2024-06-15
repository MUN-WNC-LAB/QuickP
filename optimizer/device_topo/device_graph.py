from optimizer.device_topo.intel_node_util import slurm_output_intel_2_dict, update_intra_graph_with_intel
from optimizer.device_topo.intra_node_util import phase_slurm_intra_2_DiGraphs
from optimizer.model.graph import DeviceGraph, combine_graphs, visualize_graph
from slurm_util import get_server_ips, get_slurm_available_nodes, run_srun_command, SLURM_RUN_CONF


def get_device_topo():
    servers = get_server_ips()
    nodes = get_slurm_available_nodes()

    if nodes < 0:
        raise ValueError("No available nodes in Slurm to run the job.")
    if nodes != len(servers):
        raise ValueError("Number of nodes does not match the number of servers")

    output_intra = run_srun_command(nodes, SLURM_RUN_CONF.INTRA_NODE)
    output_intel = run_srun_command(nodes, SLURM_RUN_CONF.INTER_NODE)
    if output_intra:
        graph_list_intra = phase_slurm_intra_2_DiGraphs(output_intra)
        dict_list_intel = slurm_output_intel_2_dict(output_intel)
        graph_combined = combine_graphs(graph_list_intra)
        update_intra_graph_with_intel(graph_combined, dict_list_intel)
        visualize_graph(graph_combined)
    else:
        raise ValueError("No available nodes in Slurm to run the job.")
