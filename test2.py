import torch
import torch.optim as optim
import torchvision.models as models
from apex import amp

torch.backends.cudnn.benchmark = True

model = models.resnet18().cuda()
t = torch.ones([1, 3, 2465, 4001]).cuda(non_blocking=True)
optimizer = optim.Adam(params=model.parameters(), lr=1e-3, weight_decay=0.1)

model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

output = model(t)

