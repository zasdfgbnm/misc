import torch
from torch import nn

device = 'cuda'

# c = torch.nn.Conv2d(1,1,1, bias=False).cuda()
# i = torch.randn(1, 1, 1, 1).cuda()
# o = c(i)
# print('expect:', (c.weight * i).item(), 'get:', o.item())
# o_cpu = c.cpu()(i.cpu())
# print('error:', (o.cpu() - o_cpu).abs().max())
# print('cuda vs cpu:', torch.stack([o.cpu().flatten(), o_cpu.flatten()]).t())

# dtype = torch.float32
# input = torch.randn((1, 16, 1, 1), dtype=dtype, device="cuda", requires_grad=True)
# weight = torch.randn((8, 16, 3, 3), dtype=dtype, device="cuda", requires_grad=True)
# weight = weight.to(memory_format=torch.channels_last)
# input = input.to(memory_format=torch.channels_last)
# o = torch.conv2d(input, weight, None, (2, 1), (1, 1), (1, 1), 1)
# o.sum().backward()

dtype = torch.half
m = nn.Conv2d(4, 4, kernel_size=3, groups=2).to(device, dtype)
i = torch.randn(2, 4, 6, 6, device=device, dtype=dtype, requires_grad=True)
output = m(i)