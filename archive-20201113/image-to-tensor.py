from skimage import io, transform
import torchvision
import torch

img = io.imread('300x300.jpg')

transformations = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
t_img = transformations(img)
torch.set_printoptions(threshold=1000000000000)
print(t_img)