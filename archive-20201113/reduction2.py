import torch

a = torch.randn(8, 1, 128, 1024, 1024, device='cuda').transpose(-1, -2)
s = a.sum(1, keepdim=True)
print(a.stride())  # (134217728, 134217728, 1048576, 1, 1024)
print(s.stride())  # (134217728, 134217728, 1048576, 1024, 1)
