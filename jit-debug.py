import torch

def g(x: torch.dtype):
    return torch.zeros(6, dtype=x)

s = torch.jit.script(g)
print(s.graph)
print(g(torch.float32))
print(s(torch.float32))

torch.jit.save(s, 'save.zip')
