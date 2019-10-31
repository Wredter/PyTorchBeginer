from __future__ import division
from Models.YOLO.config.parser import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


class YOLODeepNet(nn.Module):
    def __init__(self,cfgfile):
        super(YOLODeepNet,self).__init__()
        self.blocks = parse_cfg(cfgfile)
        self.net_info, self.module_list = create_modules(self.blocks)

    def forward(self, x, CUDA):
        return
