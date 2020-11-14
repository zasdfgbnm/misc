import torch


def torch_memory(device):
    print(f'{torch.cuda.memory_allocated(device)/1024/1024:.2f} MB USED')
    print(f'{torch.cuda.max_memory_allocated(device)/1024/1024:.2f} MB USED MAX')
    print('')


# Options that should affect which algorithm is chosen
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device(0)

conv0 = torch.nn.Conv3d(64, 64, 3, padding=1).to(device)

x = torch.randn((1, 64, 24, 512, 512), dtype=torch.float32, device=device)
print('Memory after input array is created:')
torch_memory(device)

with torch.no_grad():
    y = conv0(x)
    print('Memory after convolution is computed:')
    torch_memory(device)
