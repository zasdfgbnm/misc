import torch
a = torch.zeros(10000, 10, device='cuda')
b = a[0:500,:]
d = a[0:500,:]
print(d.is_contiguous())
with torch.autograd.profiler.record_function("copy-bug"):
    b.copy_(d)
