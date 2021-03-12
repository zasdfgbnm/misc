import torch
from torch._C import dtype
import torch.distributed as dist
from torch.multiprocessing import Process
import os

world_size = 2


def assertEqual(t1, t2):
    print(t1, t2)
    assert torch.allclose(t1, t2)


def init_process(rank, fn):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group('nccl', rank=rank, world_size=world_size)
    fn(rank)

def useless_computation(rank):
    for _ in range(2):
        t1 = torch.randn(1024, 1024, device=f'cuda:{rank}')
        t2 = torch.randn(1024, 1024, device=f'cuda:{rank}')
        t1 @ t2

def fn(rank):
    cpu_tensor = torch.ones(2, dtype=torch.int32)
    gpu_tensor = torch.zeros(2, dtype=torch.int32, device=f'cuda:{rank}')
    useless_computation(rank)
    gpu_tensor.copy_(cpu_tensor, True)
    dist.all_reduce(gpu_tensor)
    print(gpu_tensor)

if __name__ == "__main__":
    processes = []
    for rank in range(world_size):
        p = Process(target=init_process, args=(rank, fn))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
