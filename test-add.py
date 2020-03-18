import torch

device = 'cuda'
m1 = torch.randn(100, 100, device=device)
v1 = torch.randn(100, device=device)
res1 = torch.add(m1[:, 4], v1)
