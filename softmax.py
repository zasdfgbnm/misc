import torch
x = torch.randn((1100000000, 2), device="cuda:0", dtype=torch.float16, requires_grad=True)
y = torch.nn.functional.log_softmax(x, dim=-1, dtype=torch.float32)
y.backward(y)
torch.cuda.synchronize()