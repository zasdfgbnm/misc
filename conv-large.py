import torch
from torch import nn
device = 'cuda'
dtype = torch.half
conv = nn.Conv2d(2, 2, 8, 8, bias=False).to(device).to(dtype)
input_large = torch.randn(4097, 2, 512, 512, dtype=dtype, device=device)
ret = conv(input_large)
