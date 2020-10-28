import torch
import torch.nn as nn

def assertEqual(x, y, **kwargs):
    diff = (x.cpu() - y.cpu()).abs().max().item()
    print(diff)
    assert diff < 1e-3


n, c, h, w, k, filter_size = 4, 2, 8, 8, 4, 2
data = torch.randn((n, c, h, w), dtype=torch.float32, device='cuda')
conv = nn.Conv2d(c, k, filter_size).float().cuda()
ref_out = conv(data)

conv.to(memory_format=torch.channels_last)
out = conv(data)

assertEqual(out, ref_out)
