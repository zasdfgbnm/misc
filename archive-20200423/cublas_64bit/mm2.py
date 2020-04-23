import torch as torch

N = 16909322
K = 127

U = torch.zeros((N * K, 1), dtype=torch.half, device="cuda:0")
Ut = torch.zeros((1, N * K), dtype=torch.half, device="cuda:0")
v = torch.mm(Ut, U)

torch.cuda.synchronize()
