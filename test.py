import torch
print(torch.__version__)
x = torch.randn(6, 28, 108, 108, dtype=torch.half, device='cuda', requires_grad=True)\
    .to(memory_format=torch.channels_last)
net = torch.nn.ConvTranspose2d(28, 28, kernel_size=1, groups=1).cuda().half()\
    .to(memory_format=torch.channels_last)
with torch.backends.cudnn.flags(enabled=False, deterministic=True, benchmark=True, allow_tf32=True):
    out = net(x)
    out.backward(torch.randn_like(out))
torch.cuda.synchronize()
