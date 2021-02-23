import torch
x = torch.randn((64, 400, 120, 1024), device="cuda:0", dtype=torch.float16)
torch.cuda.profiler.start()
y = torch.nn.functional.log_softmax(x, dim=-1)
torch.cuda.profiler.stop()
torch.cuda.synchronize()