# python -m torch.distributed.launch --nproc_per_node=2 ddp_torch_distributed_launch.py

import torch
from torch import nn
import torch.distributed as dist
import torch.nn.functional as F
import os
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", default=0, type=int)
args = parser.parse_args()

# Control flow that is the same across ranks.
batch = 20
dim = 10
world_size = 2

def assertEqual(t1, t2):
    print("{} {}".format(t1, t2), flush=True)
    # assert torch.allclose(t1, t2)

torch.distributed.init_process_group(backend='nccl',
                                     init_method='env://')

rank = args.local_rank
world_size = dist.get_world_size()
torch.cuda.set_device(rank)

class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.lin1 = nn.Linear(10, 10, bias=False)
        self.lin2 = nn.Linear(10, 10, bias=False)

    def forward(self, x):
        # Second layer is used dependent on input x.
        use_second_layer = torch.equal(
            x, torch.ones(batch, dim, device=x.device)
        )
        if use_second_layer:
            return self.lin2(F.relu(self.lin1(x)))
        else:
            return F.relu(self.lin1(x))