import torch
import math
from torch.utils.tensorboard import SummaryWriter


dtype = torch.float
device = torch.device("cuda:0") # Uncomment this to run on GPU

writer = SummaryWriter()

N, D_in, H, D_out = 64, 1000, 100, 10

x = torch.randn(N, D_in, device=device, dtype=dtype)
y = torch.randn(N, D_out, device=device, dtype=dtype)
w1 = torch.randn(D_in, H, device=device, dtype=dtype, requires_grad=True)
w2 = torch.randn(H, D_out, device=device, dtype=dtype, requires_grad=True)

def write_matrix(name, tensor, global_step=None):
    writer.add_histogram(f'hist-{name}', tensor, global_step)
    writer.add_image(f'img-{name}', tensor, global_step, dataformats='HW')

write_matrix('x', x)
write_matrix('y', y)

learning_rate = 1e-6

best = math.inf
best_t = 0

def forward_backward(write):
    if write:
        write_matrix(f'w1', w1, t)
        write_matrix(f'w2', w2, t)

    def run():
        w1.grad = None
        w2.grad = None
        out1 = x.mm(w1)
        out2 = out1.clamp(min=0)
        y_pred = out2.mm(w2)
        loss = (y_pred - y).pow(2).sum()
        loss.backward()

        if write:
            tf32_tag = 'tf32' if torch.backends.cuda.matmul.allow_tf32 else 'fp32'
            write_matrix(f'out1-{tf32_tag}', out1, t)
            write_matrix(f'out2-{tf32_tag}', out2, t)
            write_matrix(f'w1.grad-{tf32_tag}', w1.grad, t)
            write_matrix(f'w2.grad-{tf32_tag}', w2.grad, t)
            writer.add_scalar(f'loss-{tf32_tag}', loss, t)

        return out1, out2, y_pred, loss

    torch.backends.cuda.matmul.allow_tf32 = True
    out1_tf32, out2_tf32, y_pred_tf32, loss_tf32 = run()
    torch.backends.cuda.matmul.allow_tf32 = False
    out1, out2, y_pred, loss = run()
    writer.add_scalar(f'out1-diff-std', (out1_tf32 - out1).std(), t)
    writer.add_scalar(f'out2-diff-std', (out2_tf32 - out2).std(), t)
    writer.add_scalar(f'y_pred-diff-std', (y_pred_tf32 - y_pred).std(), t)
    return loss

for t in range(-300, 100000000):

    loss = forward_backward(t > 0)

    if loss.item() < 1e-3:
        break

    if t % 10 == 9:
        print(t, loss.item())

    with torch.no_grad():
        w1 -= learning_rate * w1.grad
        w2 -= learning_rate * w2.grad

