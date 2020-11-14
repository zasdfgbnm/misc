import torch

a = torch.tensor((complex(1, 1),), device='cuda:0', dtype=torch.complex128)
b = torch.tensor((complex(1, 1 + 1e-8),), device='cuda:0', dtype=torch.complex128)
atol = 1e-08
rtol = 1e-05
close = a == b
allowed_error = atol + (rtol * b).abs()
# torch.isclose(a, b, equal_nan = False, atol = 1e-08, rtol = 1e-05)
torch.cuda.synchronize()
