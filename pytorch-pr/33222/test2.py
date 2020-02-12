import torch

s = 2048
a = torch.zeros(s, device='cuda')
b = torch.ones(s, device='cuda')
c = a + b
torch.cuda.synchronize()
