import torch
import torchvision

m = torchvision.models.resnet.resnet50().cuda()
i = torch.randn(64, 3, 224, 224, device='cuda')
o = torch.optim.Adam(m.parameters())
m(i).sum().backward()
o.step()
