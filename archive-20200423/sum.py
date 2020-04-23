import torch

a = torch.randn(1024 * 1024 * 1024, device='cuda', dtype=torch.half)
a.sum()
torch.cuda.synchronize()

a = torch.randn(1024, 1024, 1024, device='cuda', dtype=torch.half)
a.sum()
torch.cuda.synchronize()