import torch

dtype = torch.float

t = torch.tensor([
    [5, 2, 3],
    [8, 5, 6],
], device='cuda', dtype=torch.float)

v, i = t.min(1)
torch.cuda.synchronize()
print(v, v.dtype)
print(i, i.dtype)


t = torch.tensor([
    [5, 2, 3],
    [8, 5, 6],
], device='cpu', dtype=torch.float)

v, i = t.min(1)
torch.cuda.synchronize()
print(v, v.dtype)
print(i, i.dtype)

t = torch.tensor([
    [5, 2, 3],
    [8, 5, 6],
], device='cuda', dtype=torch.float)

v, i = t.min(0)
torch.cuda.synchronize()
print(v, v.dtype)
print(i, i.dtype)


t = torch.tensor([
    [5, 2, 3],
    [8, 5, 6],
], device='cpu', dtype=torch.float)

v, i = t.min(0)
torch.cuda.synchronize()
print(v, v.dtype)
print(i, i.dtype)