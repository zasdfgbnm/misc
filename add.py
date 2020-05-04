import torch

a = torch.zeros(10, 64, 256, 256, device='cuda')
b = torch.zeros(10, 64, 1, 1, device='cuda')
a + b