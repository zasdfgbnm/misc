import torch

torch.randperm(2**31, dtype=torch.float, device='cuda')
