import torch

# This case works well
x = torch.tensor([1,2,3,4,5,2,1,-1,-4,12,4]).cuda()
torch.topk(x,k=1,largest=False)
torch.cuda.synchronize()
print("case 1 works!")

# This case fails
x = torch.Tensor([1,2,3,4,5,2,1,-1,-4,12,4]).cuda()
torch.topk(x,k=1,largest=False)
torch.cuda.synchronize()
print("case 2 works!")
