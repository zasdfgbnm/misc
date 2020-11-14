import torch
c = 7
layer = torch.nn.Conv2d(3, 1, 3).cuda()
inp = torch.rand(100, 3, 100, 1000).cuda()
out = layer(inp)
res = out.unfold(3, c, 1)
res_sum = res.sum()
torch.cuda.synchronize()

