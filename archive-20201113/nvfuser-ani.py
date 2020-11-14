import torch
import torchani

device = torch.device('cuda')
coordinates = torch.tensor([[[3., 3., 4.], [1.0, 2.0, 1.0]]],
                            dtype=torch.double,
                            requires_grad=True,
                            device=device)
atomic_numbers = torch.tensor([[1, 6]], dtype=torch.long, device=device)
with torch.jit.fuser('fuser2'):
    model_jit = torch.jit.script(torchani.models.ANI1x(model_index=0, periodic_table_index=True)).to(device).double()
    for j in range(4):
        energy = model_jit((atomic_numbers, coordinates)).energies * torchani.units.HARTREE_TO_KCALMOL
        _ = -torch.autograd.grad(energy.sum(), coordinates)[0][0]
