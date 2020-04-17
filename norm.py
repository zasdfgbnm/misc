import torch

device = 'cuda'
x = torch.randn(1001, device=device)
res = x.norm().cpu()