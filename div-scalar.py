import torch

with torch.autograd.profiler.emit_nvtx():
    for i in range(100):
        c = torch.rand(6, 16, 512, 512, dtype=torch.float16, device='cuda')
        c / 8.0