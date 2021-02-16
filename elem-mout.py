import torch
device = "cuda"
x = torch.ones(3,4, device=device, requires_grad=True)
scale_curr = torch.ones(3, device=device, requires_grad=True)
zero_point = torch.zeros(3, device=device, requires_grad=True)
axis = 0
quant_min = 0
quant_max = 15
Y_prime = torch._fake_quantize_learnable_per_channel_affine(
                x, scale_curr, zero_point, axis, quant_min, quant_max)
dout = torch.rand(x.shape, dtype=torch.float, device=device)
Y_prime.backward(dout)
torch.cuda.synchronize()
print(x, Y_prime, x.grad, dout, sep='\n')
