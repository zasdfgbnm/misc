import torch
import math

print("CUDA name: {}".format(torch.cuda.get_device_name(0)))

a = torch.rand(65537, 22, 64).cuda().half()
b = torch.rand(65537, 64, 22).cuda().half()
c = torch.full((65537, 22, 22), math.nan, dtype=torch.half, device='cuda')
cpu_result = torch.matmul(a.cpu().float(), b.cpu().float()).cuda().half()
torch.matmul(a, b, out=c)
print((c - cpu_result).abs().max().item())
