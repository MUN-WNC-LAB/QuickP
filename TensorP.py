from torch.distributed.tensor.parallel import parallelize_module, ColwiseParallel, RowwiseParallel
from torch.distributed.device_mesh import init_device_mesh
# Define the module.
m = Model(...)
tp_mesh = init_device_mesh("cuda", (8,))
m = parallelize_module(m, tp_mesh, {"w1": ColwiseParallel(), "w2": RowwiseParallel()})