import torch
import torchvision
from skimage import io

img = io.imread('300x300-horse.jpg')

transformations = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
t_img = transformations(img)

# torch.backends.cuda.matmul.allow_tf32 = False
# torch.backends.cudnn.allow_tf32 = False

torch.manual_seed(0)

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(num_classes=50, pretrained_backbone=False)
model.eval()
# input_shape = (3, 300, 300)
# x = torch.rand(input_shape)
x = t_img
out1 = model([x])[0]

model.to('cuda')
x = x.to(device='cuda')
with torch.cuda.amp.autocast():
    out2 = model([x])[0]

boxes1 = out1['boxes'].flatten()
boxes2 = out2['boxes'].flatten().cpu()
print((boxes1 - boxes2).abs().max())
# print(torch.stack([boxes1, boxes2], dim=1))

labels1 = out1['labels'].flatten()
labels2 = out2['labels'].flatten().cpu()
print((labels1 - labels2).abs().max())
# print(torch.stack([labels1, labels2], dim=1))

scores1 = out1['scores'].flatten()
scores2 = out2['scores'].flatten().cpu()
print((scores1 - scores2).abs().max())
# print(torch.stack([scores1, scores2], dim=1))
