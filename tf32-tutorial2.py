import torch

dtype = torch.float
device = torch.device("cuda:0") # Uncomment this to run on GPU
torch.backends.cuda.matmul.allow_tf32 = False

N, D_in, D_out = 64, 100, 10

x = torch.randn(N, D_in, device=device, dtype=dtype)
w_target = torch.randn(D_in, D_out, device=device, dtype=dtype, requires_grad=False)
y = x.clamp(min=0).mm(w_target)

torch.backends.cuda.matmul.allow_tf32 = True

w = torch.randn(D_in, D_out, device=device, dtype=dtype, requires_grad=True)

learning_rate = 5e-4
for t in range(100000):
    y_pred = x.clamp(min=0).mm(w)
    loss = (y_pred - y).pow(2).sum()

    if t % 100 == 99:
        print(t, loss.item())
        # learning_rate /= 1.03

    loss.backward()
    with torch.no_grad():
        w -= learning_rate * w.grad
        w.grad.zero_()