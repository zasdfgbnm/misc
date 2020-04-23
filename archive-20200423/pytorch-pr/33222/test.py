import torch

a = torch.zeros(5, device='cuda')
b = torch.zeros(5, device='cuda')
a == b
torch.cuda.synchronize()
