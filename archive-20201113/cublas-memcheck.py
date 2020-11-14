import torch

x = torch.randn((100, 100), device='cuda')
y = torch.randn((100, 100), device='cuda')
z = torch.randn((100, 100), device='cuda')
torch.addmm(x, y, z)
