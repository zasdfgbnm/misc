import torch
for i in range(2000):
    x = torch.ones(2000, dtype=torch.uint8, device='cuda')
    x[i] = 0
    print(x.all())