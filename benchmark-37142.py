import torch
import torch.nn as nn


class Perm(nn.Module):
    def __init__(self):
        super().__init__()
        self.s = nn.Parameter(torch.tensor([1.0]))

    def forward(self, x):
        # = torch.nn.functional.adaptive_avg_pool2d(x, output_size=(1, 1))
        y = x.mean(dim=(2, 3), keepdim=True)
        y = y.permute(0, 2, 3, 1)  # channels to last dim
        y = y*self.s  # without this, there is next to no difference
        # channels back to second dim
        y = y.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)
        y = x * y
        return y


device = torch.device('cuda:0')  # can be changed to 'cpu'
net = Perm().to(device)
opt = torch.optim.Adam(params=net.parameters(), lr=0.001)
x = torch.zeros(10, 64, 256, 256, requires_grad=False, device=device)


def train(device=device, iters=10, x=x):
    torch.cuda.synchronize()
    for k in range(iters):
        y = net(x).sum()
        opt.zero_grad()
        y.backward()
        opt.step()
    torch.cuda.synchronize()


if __name__ == '__main__':
    import timeit
    torch.cuda.synchronize()
    print(timeit.timeit("train()", setup="from __main__ import train", number=10))

