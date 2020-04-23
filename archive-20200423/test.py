import torch
torch.jit.optimized_execution(False)

def f(x):
    return x + x + x * x

f = torch.jit.script(f)
input = torch.zeros(5)
f(input)
print(f.graph_for(input))
