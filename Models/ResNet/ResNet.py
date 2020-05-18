from __future__ import division

import numpy as np
import torch
import torch.nn as nn


class ResNet(nn.Module):
    def __int__(self):
        print("START")

    def build_resModule(self, data, out_chanels, in_chanels, kernel_size):
        bn_eps = 2e-5
        shortcut = data
        modules = nn.Sequential(
            nn.BatchNorm2d(out_chanels,
                           eps=bn_eps),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_chanels,
                      int(out_chanels/4),
                      kernel_size[0],
                      bias=False),
            nn.BatchNorm2d(int(out_chanels/4),
                           eps=bn_eps),
            nn.ReLU(inplace=True),
            nn.Conv2d()
        )