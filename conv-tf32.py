import torch

batch = 32
ins = 1024
outs = 1024
kh = 1
kw = 1
ih = 32
iw= 32
a = torch.randn(batch,ins,ih,iw,requires_grad=True).cuda()
b = torch.randn(batch,ins,ih,iw,requires_grad=True).cuda()
a.retain_grad()
b.retain_grad()
torch.cuda.synchronize()
conv1 = torch.nn.Conv2d(ins, outs, kernel_size=kw).cuda()
conv2 = torch.nn.Conv2d(ins, outs, kernel_size=kw).cuda()
torch.backends.cudnn.allow_tf32 = False
b = conv1(a)
torch.backends.cudnn.allow_tf32 = True
c = conv2(b)
torch.backends.cudnn.allow_tf32 = False
torch.sum(c).backward()
torch.cuda.synchronize()
