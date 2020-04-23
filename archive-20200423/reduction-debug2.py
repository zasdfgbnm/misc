import torch
dtype = torch.double
device = 'cuda'
a = torch.ones((1,), device='cuda', dtype=torch.bool)
print(a.all())
print(a)

