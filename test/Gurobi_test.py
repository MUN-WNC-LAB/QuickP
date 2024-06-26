import gurobipy as gp
from gurobipy import GRB


def test_addGenConstrIndicator():
    # Create a new Gurobi model
    model = gp.Model("test_addGenConstrIndicator")

    # Add binary variables
    x1 = model.addVar(vtype=GRB.BINARY, name="x1")
    x2 = model.addVar(vtype=GRB.BINARY, name="x2")

    # Add continuous variable
    y = model.addVar(vtype=GRB.CONTINUOUS, name="y", ub=500)

    # Set objective (for demonstration purposes, let's maximize y)
    model.setObjective(y, GRB.MAXIMIZE)

    # Add indicator constraints
    # When x1 is 1, y should be <= 10
    model.addGenConstrIndicator(x1, True, y <= 10)
    # When x1 is 0, y should be >= 100
    model.addGenConstrIndicator(x1, False, y >= 100)

    # When x2 is 1, y should be <= 20
    model.addGenConstrIndicator(x2, True, y <= 20)
    model.addGenConstrIndicator(x2, False, y >= 200)

    # Also add a constraint to link x1 and x2 (for demonstration purposes)
    model.addConstr(x1 + x2 <= 1, "linking_constraint")

    # Optimize the model
    model.optimize()

    # Print the results
    if model.status == GRB.OPTIMAL:
        print("Optimal solution found:")
        for v in model.getVars():
            print(f"{v.VarName} = {v.X}")
    else:
        print(f"Optimization ended with status {model.status}")


# x5 = and(x1, x3, x4)
# model.addGenConstrAnd(x5, [x1, x3, x4], "andconstr")

# overloaded forms
# model.addConstr(x5 == and_([x1, x3, x4]), "andconstr")
# model.addConstr(x5 == and_(x1, x3, x4), "andconstr")

dict1 = {}
dict1[1,3] = "999"
print(dict1[1,3])
print(dict1[(1,3)])