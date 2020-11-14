import torch
torch.backends.cuda.matmul.allow_tf32 = False

amp = True

scaler = torch.cuda.amp.GradScaler()

dtype = torch.float
device = torch.device("cuda:0") # Uncomment this to run on GPU
N, D_in, H, D_out = 64, 1000, 100, 10

x = torch.randn(N, D_in, device=device, dtype=dtype)
y = torch.randn(N, D_out, device=device, dtype=dtype)



class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w1 = torch.nn.Parameter(torch.randn(D_in, H, device=device, dtype=dtype, requires_grad=True))
        self.w2 = torch.nn.Parameter(torch.randn(H, D_out, device=device, dtype=dtype, requires_grad=True))

    def forward(self, x):
        return x.mm(self.w1).clamp(min=0).mm(self.w2)


learning_rate = 1e-6

m = M()
m_truth = M()
with torch.no_grad():
    y = m_truth(x)

optimizer = torch.optim.SGD(m.parameters(), learning_rate)

for t in range(10000):
    optimizer.zero_grad()
    with torch.cuda.amp.autocast(amp):
        y_pred = m(x)
        loss = (y_pred - y).pow(2).sum()
    if t % 100 == 99:
        print(t, loss.item())
    scaler.scale(loss).backward()
    with torch.no_grad():
        scaler.step(optimizer)
        scaler.update()
