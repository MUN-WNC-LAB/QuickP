from enum import Enum
import networkx as nx


class NodeWeightFunction(Enum):
    SUM_COMP_COST = 'sum_comp_cost'
    AVE_COMP_COST = 'ave_comp_cost'


class EdgeWeightFunction(Enum):
    MOCK_COMMUNICATION_COST = 'mock_communication_cost'
    SOURCE_OUTPUT_TENSOR = 'source_output_tensor'