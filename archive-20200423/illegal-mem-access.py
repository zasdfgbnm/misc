import torch
model = torch.hub.load('pytorch/vision:v0.5.0', 'squeezenet1_0', pretrained=True).to(memory_format=torch.channels_last,dtype=torch.half,device='cuda')
inp = torch.rand(10,3,256,256, device='cuda',dtype=torch.float16).contiguous(memory_format=torch.channels_last)
out = model(inp)
out.sum().backward()

