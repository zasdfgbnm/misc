import torch
a = torch.ones(5, dtype=torch.uint8, device='cuda')
b = torch.ones(5, dtype=torch.uint8, device='cuda')
print(a - b)