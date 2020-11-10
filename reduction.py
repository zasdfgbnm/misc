import torch

a = torch.randn(8, 1, 128, 1024, 1024, device='cuda')
s = a.sum(1, keepdim=True)
print((a - s).abs().max())
