import torch
r = 2. 
c = 2. + 2j
rt = torch.randn(5, dtype=torch.float32)
ct = torch.randn(5, dtype=torch.complex64)
o = torch.randn(5, dtype=torch.complex64)

torch.pow(r, rt, out=o)
torch.pow(r, ct, out=o)
torch.pow(c, rt, out=o)
torch.pow(c, ct, out=o)
torch.pow(rt, r, out=o)
torch.pow(rt, c, out=o)
torch.pow(rt, rt, out=o)
torch.pow(rt, ct, out=o)
torch.pow(ct, r, out=o)
torch.pow(ct, c, out=o)
torch.pow(ct, rt, out=o)
torch.pow(ct, ct, out=o)
