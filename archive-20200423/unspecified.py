import torch
dtype = torch.complex64
a = torch.zeros(500, device='cuda', dtype=dtype)
b = torch.zeros(500, device='cuda', dtype=dtype)
a - b
torch.cuda.synchronize()