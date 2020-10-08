import torch

c = torch.nn.Conv2d(1,1,1, bias=False).cuda()
i = torch.randn(1, 1, 1, 1).cuda()
o = c(i)
print((c.weight * i).item(), o.item())
o_cpu = c.cpu()(i.cpu())
print((o.cpu() - o_cpu).abs().max())
print(torch.stack([o.cpu().flatten(), o_cpu.flatten()]).t())

# dtype = torch.float32
# input = torch.randn((1, 16, 1, 1), dtype=dtype, device="cuda", requires_grad=True)
# weight = torch.randn((8, 16, 3, 3), dtype=dtype, device="cuda", requires_grad=True)
# weight = weight.to(memory_format=torch.channels_last)
# o = torch.conv2d(input, weight, None, (2, 1), (1, 1), (1, 1), 1)
# o.sum().backward()