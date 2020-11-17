import torch

def f(x):
    return x.view(torch.int32)

s = torch.jit.script(f)
print(s.graph)
print(type(s.graph))

x = torch.randn(5, 5, 5)
s(x)
