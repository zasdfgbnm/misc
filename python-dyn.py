import torch


def f()->int:
    return 1


class M(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: int):
        return x + f()


m = M()
s = torch.jit.script(m)
print(m(1))
print(s(1))

f = lambda: 2
print(m(1))
print(s(1))
