import torch
s = torch.jit.load('/home/gaoxiang/pytorch-tf32/build_test_custom_build/MobileNetV2.pt')
i = torch.ones(1, 3, 224, 224)
s(i)
