from __future__ import division

import numpy as np
import torch
import torch.nn as nn


class ResNet(nn.Module):
    def __int__(self, width, height, depth, classes, stages=None, filters=None):
        super(ResNet, self).__init__()
        if filters is None:
            filters = [64, 128, 256, 512]
        if stages is None:
            stages = [3, 4, 6]
        self.stages = stages
        self.filters = filters


class resModule(nn.Module):
    def __int__(self, channels, red=False, bn_eps=2e-5, bn_mom=0.9):
        super(resModule, self).__init__()
        print("START")
        self.bn_eps = bn_eps
        self.bn_mom = bn_mom
        self.red = red
        self.kernel_size = [(1, 1), (3, 3)]
        modules = nn.Sequential(
            nn.BatchNorm2d(channels,
                           eps=bn_eps,
                           momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels,
                      int(channels / 4),
                      self.kernel_size[0],
                      bias=False),
            nn.BatchNorm2d(int(channels / 4),
                           eps=bn_eps,
                           momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(channels / 4),
                      int(channels / 4),
                      self.kernel_size[1],
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(int(channels / 4),
                           eps=bn_eps,
                           momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(channels / 4),
                      channels,
                      self.kernel_size[0],
                      bias=False)
        )

    def forward(self, data):
        shortcut = data
        x = self.modules(data)
        return x.add(shortcut)


