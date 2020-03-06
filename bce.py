import torch

a = torch.tensor([0.9706, 0.0532])
b = torch.tensor([1.0, 0.0])
print(a)
print(b)
print(torch.nn.functional.binary_cross_entropy(a, b))
print(torch.nn.functional.binary_cross_entropy(a.cuda(), b.cuda()))

