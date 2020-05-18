import torch
t = torch.randn(10000, 10000, device='cuda')
a = t.sum(dim=0)
b = t.sum(dim=0)
print((a == b).all())
c = t.sum(dim=1)
d = t.sum(dim=1)
print((c == d).all())
