import torch
import math
device = 'cpu'
a = torch.randn((3, 2, 0), device=device)
b = torch.randn((3, 0, 4), device=device)
c = torch.full((3, 2, 4), math.nan, device=device)
out = torch.full((3, 2, 4), math.nan, device=device)
print(torch.baddbmm(c, a, b, beta=0, alpha=1, out=out))
