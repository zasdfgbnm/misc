import torch
device = 'cuda'
m = torch.nn.PReLU().cuda().half()
input_ = torch.ones((1024, 1024, 1024, 2), dtype=torch.half, device=device)
output = m(input_)
output.backward(input_)
