import torch
dtype = torch.float
device = 'cuda'
size = 1024 * 1024 * 64 + 3
x = torch.zeros(size, dtype=dtype, device=device)
x.sum()
torch.cuda.synchronize()
