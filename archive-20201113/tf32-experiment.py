import torch
import numpy

def view_dtype(tensor, target_type):
    return torch.from_numpy(tensor.cpu().numpy().view(target_type)).to(tensor.device)