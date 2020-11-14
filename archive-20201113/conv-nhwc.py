import torch
import torch.nn as nn

dtype = torch.double

def assertEqual(x, y, **kwargs):
    print((x.cpu() - y.cpu()).abs().max())

def helper(n, c, h, w, out_channels, kernel_size, groups):
    input = torch.randn((n, c, h, w), dtype=dtype, device='cuda')
    conv = nn.Conv2d(c, out_channels, kernel_size, groups=groups).to(dtype).cuda()

    out = conv(input)
    ref_out = conv.cpu()(input.cpu())

    assertEqual(out, ref_out, exact_dtype=False)

helper(2, 8, 4, 4, out_channels=4, kernel_size=3, groups=1)
