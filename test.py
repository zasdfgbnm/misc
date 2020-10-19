import torch
import torchvision

torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(num_classes=50, pretrained_backbone=False)
model.eval()
input_shape = (3, 300, 300)
x = torch.rand(input_shape)
out1 = model([x])[0]

model.to('cuda')
x = x.to(device='cuda')
out2 = model([x])[0]

boxes1 = out1['boxes'].flatten()
boxes2 = out2['boxes'].flatten().cpu()
print((boxes1 - boxes2).abs().max())
print(torch.stack([boxes1, boxes2], dim=1))

labels1 = out1['labels'].flatten()
labels2 = out2['labels'].flatten().cpu()
print((labels1 - labels2).abs().max())
print(torch.stack([labels1, labels2], dim=1))

scores1 = out1['scores'].flatten()
scores2 = out2['scores'].flatten().cpu()
print((scores1 - scores2).abs().max())
print(torch.stack([scores1, scores2], dim=1))

