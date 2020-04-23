import torch
c = torch.nn.Conv2d(1, 1, 1).cuda()
i = torch.zeros(10, 1, 10, 10).cuda()
c(i)