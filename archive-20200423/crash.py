import torch

a = torch.zeros(5555, device='cuda')
b = torch.zeros(5555, device='cuda', dtype=torch.float64)
c = torch.zeros(5555, device='cuda')
torch.where(a, b, c)
