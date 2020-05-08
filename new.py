import torch
import inspect

class Model(torch.nn.Module):

    def __new__(cls, *args, from_cache=True, **kwargs):
        if from_cache:
            return torch.load('cache.pt')
        return super().__new__(cls)

    def __init__(self, *args, from_cache=True, **kwargs):
        super().__init__()
        print('slow initialization')
        size1, size2 = args
        self.layer1 = torch.nn.Linear(size1, size2)

    def forward(self, x):
        return self.layer1(x)


def build_cache():
    print('entering build_cache')
    m = Model(from_cache=False)
    torch.save(m, 'cache.pt')
    print('leaving build_cache')

build_cache()
# Model(10, 10)