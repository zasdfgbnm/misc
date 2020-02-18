import torch

a = torch.zeros(5, 5, device='cuda')
b = a.t()
print(b.stride())
torch.arange(25, out=b)
print(b.flatten())