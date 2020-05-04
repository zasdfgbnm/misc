import torch

x = torch.rand(100, 300, 50, device='cuda')
print(x.stride())
mean2 = x.mean(dim=1)
