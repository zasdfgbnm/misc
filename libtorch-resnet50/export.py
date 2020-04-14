import torch
import torchvision

# An instance of your model.
model = torchvision.models.resnet50()

# An example input you would normally provide to your model's forward() method.
example = torch.rand(1, 3, 224, 224)

# Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
s = torch.jit.script(model)

s.save("resnet50_libtorch.pt")
