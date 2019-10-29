import torch.nn as nn


class EmptyLayer(nn.Module):
    def forward(self, *input):
        pass

    def __init__(self):
        super(EmptyLayer, self).__init__()


class DetectionLayer(nn.Module):
    def forward(self, *input):
        pass

    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors

