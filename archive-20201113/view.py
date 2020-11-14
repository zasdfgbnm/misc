import torch

def foo(x):
    return x * 2

x = torch.rand(3, 4)
traced_foo = torch.jit.trace(foo, x)