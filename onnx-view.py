import torch
import torch.onnx.utils

g = torch._C.Graph()
n = g.create('onnx::Constant', 1)
n.z_('value', torch.tensor(0))
