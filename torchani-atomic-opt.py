import ase
import ase.optimize
import torch
import torchani
import math
from ase.visualize import view
from ase.io.trajectory import Trajectory
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet
from ase import units

# torch.backends.cuda.matmul.allow_tf32 = False

device = torch.device('cuda')
model = torchani.models.ANI1x(periodic_table_index=True).to(device)

atoms = ase.io.read('aspirin.sdf')
atoms.calc = model.ase()

traj = Trajectory('example.traj', 'w', atoms)

MaxwellBoltzmannDistribution(atoms, 300 * units.kB)
dyn = VelocityVerlet(atoms, 1 * units.fs)  # 5 fs time step.
dyn.attach(traj.write, interval=1)
dyn.run(1000)

opt = ase.optimize.BFGS(atoms)
opt.attach(traj.write, interval=1)
opt.run(fmax=1e-3)
traj.close()
