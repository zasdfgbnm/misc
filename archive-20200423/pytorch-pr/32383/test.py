import torch

a = torch.zeros(1024, dtype=torch.double, device='cuda')
b = torch.zeros(1024, dtype=torch.double, device='cuda')

a[16:] + b[16:]