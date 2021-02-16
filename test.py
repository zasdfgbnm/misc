import torch
import math
x = torch.tensor(complex(-math.inf, math.nan), device='cuda:0')
print(x, x.asin(), x.cpu().asin())
# tensor(-inf+nanj, device='cuda:0') tensor(nan-infj, device='cuda:0') (nan+infj)
print(math.inf + math.nan * 1.0j)
