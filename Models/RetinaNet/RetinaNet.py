import math

from torch.autograd import Variable
import torch
import torch.nn as nn
from Models.RetinaNet.ResNetFPN import resnet50_FPN, resnet101_FPN


class RetinaNet(nn.Module):
    num_anchors = 9

    def __init__(self, num_classes=1):
        super(RetinaNet, self).__init__()
        self.num_classes = num_classes
        self.backbone = resnet50_FPN(self.num_classes)
        self.loc_head = self._make_head(self.num_anchors * 4,
                                        cls=False)
        self.cls_head = self._make_head(self.num_anchors * self.num_classes,
                                        cls=True)
        self.freeze_bn()

    def forward(self, x):
        features = self.backbone(x)
        loc_preds = []
        cls_preds = []
        for f in features:
            loc_pred = self.loc_head(f)
            cls_pred = self.cls_head(f)
            loc_pred = loc_pred.permute(0, 2, 3, 1).\
                contiguous().view(x.size(0), -1, 4)
            cls_pred = cls_pred.permute(0, 2, 3, 1).\
                contiguous().view(x.size(0), -1, self.num_classes)
            loc_preds.append(loc_pred)
            cls_preds.append(cls_pred)
        return torch.cat(loc_preds, 1), torch.cat(cls_preds, 1)

    def _make_head(self, out_planes, feature_size=256, cls=False):
        layers = []
        for _ in range(4):
            layers.append(nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(feature_size, out_planes, kernel_size=3, stride=1, padding=1))
        layers[-1].weight.data.fill_(0)
        if cls:
            prior = 0.01
            layers[-1].bias.data.fill_(-math.log((1.0 - prior) / prior))
        else:
            layers[-1].bias.data.fill_(0)
        return nn.Sequential(*layers)

    def freeze_bn(self):
        '''Freeze BatchNorm layers.'''
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()

