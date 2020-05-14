import torch

def f(x):
    return x.min(0)

x = torch.zeros(5,5, device='cuda')
print(torch.jit.trace(f, x).graph)
