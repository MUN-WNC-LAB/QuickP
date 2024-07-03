import unittest
from unittest.mock import Mock
import gurobipy as gp
from gurobipy import GRB


class TestSchedulingConstraints(unittest.TestCase):
    def setUp(self):
        # Initialize the model
        self.model = gp.Model("minimize_maxload")

        # Mock the comp_graph and deviceTopo
        self.comp_graph = Mock()
        self.deviceTopo = Mock()

        # Sample operator and device IDs
        self.operator_ids = ['op1', 'op2', 'op3']
        self.device_ids = ['device1', 'device2']

        self.comp_graph.getOperatorIDs.return_value = self.operator_ids
        self.deviceTopo.getDeviceIDs.return_value = self.device_ids

        # Sample data for operators and devices
        self.sample_operator_data = {
            'op1': {'mem': 10, 'comp_cost': {'device1': 5, 'device2': 10}},
            'op2': {'mem': 15, 'comp_cost': {'device1': 6, 'device2': 8}},
            'op3': {'mem': 20, 'comp_cost': {'device1': 7, 'device2': 9}}
        }
        self.sample_device_data = {
            'device1': {'memory_capacity': 30},
            'device2': {'memory_capacity': 50}
        }

        def mock_get_operator(node_id):
            return self.sample_operator_data[node_id]

        def mock_get_device(machine_id):
            return self.sample_device_data[machine_id]

        def mock_get_operator_comp_cost_by_device(node_id, device_id):
            return self.sample_operator_data[node_id]['comp_cost'][device_id]

        self.comp_graph.getOperator.side_effect = mock_get_operator
        self.deviceTopo.getDevice.side_effect = mock_get_device
        self.comp_graph.getOperatorCompCostByDevice.side_effect = mock_get_operator_comp_cost_by_device

        # Define start and finish times as Gurobi decision variables
        self.start = {op: self.model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"start_{op}") for op in
                      self.operator_ids}
        self.finish = {op: self.model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"finish_{op}") for op in
                       self.operator_ids}

        # Define x as a binary variable indicating if an operator is placed on a device
        self.x = {(op, device): self.model.addVar(vtype=GRB.BINARY, name=f"x_{op}_{device}") for op in self.operator_ids
                  for device in self.device_ids}

    def test_non_overlapping_constraints(self):
        # Add constraint to ensure each device processes only one operator at a time. This is a SCHEDULING problem
        for device in self.deviceTopo.getDeviceIDs():
            for i in range(len(self.operator_ids)):
                for j in range(i + 1, len(self.operator_ids)):
                    op1 = self.operator_ids[i]
                    op2 = self.operator_ids[j]
                    # Create auxiliary binary variables
                    y1 = self.model.addVar(vtype=GRB.BINARY, name=f"y1_{device}_{op1}_{op2}")
                    y2 = self.model.addVar(vtype=GRB.BINARY, name=f"y2_{device}_{op1}_{op2}")
                    self.model.addGenConstrIndicator(y1, True, self.finish[op1] <= self.start[op2])
                    self.model.addGenConstrIndicator(y2, True, self.finish[op2] <= self.start[op1])

                    # If on the same device, ensure that the operators do not overlap
                    self.model.addConstr(y1 + y2 >= self.x[op1, device] + self.x[op2, device] - 1,
                                         name=f"non_overlap_{op1}_{op2}_{device}")

        self.model.optimize()
        self.assertEqual(self.model.Status, GRB.OPTIMAL,
                         "Model is not optimal after adding non-overlapping constraints.")

    def test_total_latency_constraint(self):
        # TotalLatency that we are minimizing
        TotalLatency = self.model.addVar(vtype=GRB.CONTINUOUS, lb=0.0)
        for op_end in self.finish.values():
            self.model.addConstr(TotalLatency >= op_end, "satisfy_each_device_latency")

        self.model.setObjective(TotalLatency, GRB.MINIMIZE)

        self.model.optimize()
        self.assertEqual(self.model.Status, GRB.OPTIMAL, "Model is not optimal after adding total latency constraint.")


if __name__ == '__main__':
    unittest.main()
