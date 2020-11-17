import torch

def fn(x):
    print(torch.jit.is_tracing())
    return x.view(x.shape[1] * 2, x.size(0), 2)

x = torch.randn(5, 2, 4, requires_grad=False)
y = torch.randn(4, 8, 4, requires_grad=False)

# Check that it behaves as expected
traced_fn = torch.jit.trace(fn, x)
print(traced_fn.graph)
traced_fn(x)
traced_fn(y)
