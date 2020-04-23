import torch
a = torch.randn(5, dtype=torch.complex64).cuda()
torch.cuda.synchronize()
b = a.to(torch.complex128)
torch.cuda.synchronize()
# a = torch.zeros(5, dtype=torch.complex64, device='cuda')
# b = torch.zeros(5, dtype=torch.float, device='cuda')
# a + b
# torch.cuda.synchronize()
