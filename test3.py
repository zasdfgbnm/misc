import torch
import torchvision.models as models
torch.backends.cudnn.benchmark=False
torch.backends.cudnn.deterministic=True

resnet = models.resnet18().half().cuda()
t = torch.ones([1, 3, 10210, 8641], dtype=torch.float16, device="cuda")
output = resnet(t)
