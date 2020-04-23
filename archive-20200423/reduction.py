import torch
c = 7
layer = torch.nn.Conv2d(3, 1, 3).cuda()
inp = torch.rand(100, 3, 100, 1000).cuda()
out = layer(inp)
res = out.unfold(3, c, 1)
res = res.contiguous()
print(res.shape)
res = torch.zeros((10, 9721600, 2), device='cuda')[:,1:,1:]
print(res.shape, res.stride())
res_sum = res.sum()
torch.cuda.synchronize()

