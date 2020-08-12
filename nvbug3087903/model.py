import torch
import torchvision


class FasterRCNNWrapper(torch.nn.Module):

    def __init__(self):
        super().__init__()

        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    def forward(self, x):
        losses, detections = self.model([x])
        return detections[0]['boxes'], detections[0]['labels'], detections[0]['scores']


model = FasterRCNNWrapper()

script = torch.jit.script(model)
script.save('model.pt')