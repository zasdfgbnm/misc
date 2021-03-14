import torch
from torch import nn
import torch.distributed as dist
import torch.nn.functional as F
from torch.multiprocessing import Process
import os

# Control flow that is the same across ranks.
batch = 20
dim = 10
world_size = 2


def assertEqual(t1, t2):
    print(t1, t2)
    assert torch.allclose(t1, t2)


def init_process(rank, fn):
    """ Initialize the distributed environment. """
    torch.cuda.profiler.start()
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group('nccl', rank=rank, world_size=world_size)
    fn(rank)


def fn(rank):
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

    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    model = torch.nn.parallel.DistributedDataParallel(
        ToyModel().cuda(rank),
        device_ids=[rank],
        find_unused_parameters=True,
    )
    random_input = torch.randn(batch, dim, device=rank)
    ones_input = torch.ones(batch, dim, device=rank)
    for i in range(6):
        if i % 2 == 0:
            out = model(random_input)
        else:
            out = model(ones_input)
        loss = out.sum()
        loss.backward()
        # On even iterations, 2nd param goes unused, on odd iterations,
        # it is used.
        local_used_maps = model.reducer._get_local_used_maps()
        if i % 2 == 0:
            expected = torch.tensor([world_size, 0], device=rank, dtype=torch.int32)
        else:
            expected = torch.tensor([world_size, world_size], device=rank, dtype=torch.int32)

        # Validate parameter usage.
        variable_usage_tensor = local_used_maps[0]
        torch.cuda.synchronize()
        assertEqual(variable_usage_tensor, expected)

if __name__ == "__main__":
    processes = []
    for rank in range(world_size):
        p = Process(target=init_process, args=(rank, fn))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
