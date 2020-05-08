import torch
t = torch.zeros(5, 14400, 14400, device='cuda')
t.sum(dim=0)
