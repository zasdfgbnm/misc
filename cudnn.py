# pytest -v test/test_nn.py::TestNN::test_Conv3d_circular_stride2_pad2

import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.allow_tf32 = True
data = torch.randn([2, 2, 6, 8, 10], dtype=torch.float, device='cuda', requires_grad=True)
net = torch.nn.Conv3d(2, 3, kernel_size=[3, 3, 3], padding=[0, 0, 0], stride=[2, 2, 2], dilation=[1, 1, 1], groups=1)
net = net.cuda().float()
out = net(data)
out.backward(torch.randn_like(out))
torch.cuda.synchronize()

print("done")