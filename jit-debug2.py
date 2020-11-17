import torch

def f(x):
    return x.view("fsdafas")

s = torch.jit.script(f)
print(s.graph)

x = torch.randn(5, 5, 5)
s(x)
