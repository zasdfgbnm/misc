import torch
dtype = torch.double
device = 'cuda'
a = torch.ones((), device='cuda', dtype=torch.bool)
print(a)
print(a.all())

