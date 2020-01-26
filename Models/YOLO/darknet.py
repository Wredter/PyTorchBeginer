from __future__ import division
from Models.YOLO.config.parser import *
from Models.YOLO.utility import *
import torch
import torch.nn as nn
import cv2
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


def get_test_input():
    img = cv2.imread("dog-cycle-car.png")
    img = cv2.resize(img, (416, 416))          #Resize to the input dimension
    img_ = img[:, :, ::-1].transpose((2, 0, 1))  # BGR -> RGB | H X W C -> C X H X W
    img_ = img_[np.newaxis, :, :, :]/255.0       #Add a channel at 0 (for batch) | Normalise
    img_ = torch.from_numpy(img_).float()     #Convert to float
    img_ = Variable(img_)                     # Convert to Variable
    return img_


class YOLODarkNet(nn.Module):
    def __init__(self, cfgfile):
        super(YOLODarkNet, self).__init__()
        self.blocks = parse_cfg(cfgfile)
        self.net_info, self.module_list = create_modules(self.blocks)

    def forward(self, x, CUDA):
        modules = self.blocks[1:]
        # print(self.blocks[1:])
        inp_dim = 0
        num_classes = 0
        outputs = {}
        write = 0
        for i, module in enumerate(modules):
            module_type = (module["type"])
            if module_type == "convolutional" or module_type == "upsample":
                # To pytorch umie samemu
                x = self.module_list[i](x)
            elif module_type == "route":
                # Tego nie umie
                layers = module["layers"]
                layers = [int(a) for a in layers]

                if (layers[0]) > 0:
                    layers[0] = layers[0] - i

                if len(layers) == 1:
                    x = outputs[i + (layers[0])]
                else:
                    if(layers[1]>0):
                        layers[1] = layers[1] - i
                    map1 = outputs[i + layers[0]]
                    map2 = outputs[i + layers[1]]

                    x = torch.cat((map1, map2), 1)
            elif module_type == "shortcut":
                from_ = int(module["from"])
                x = outputs[i-1] + outputs[i+from_]
            elif module_type == 'yolo':
                anchors = self.module_list[i][0].anchors
                # Get the input dimensions
                inp_dim = int(self.net_info["height"])

                # Get the number of classes
                num_classes = int(module["classes"])

                # Transform
                x = x.data
                if CUDA:
                    x = x.to('cuda')
                x = predict_transform(x, inp_dim, anchors, num_classes, CUDA)
                if not write:  # if no collector has been intialised.
                    detections = x
                    write = 1
                else:
                    detections = torch.cat((detections, x), 1)

            outputs[i] = x

        dumy_target = torch.cuda.FloatTensor([[0.5, 0.5, 0.1, 0.1, 0]])
        write_results(detections, 0.5, num_classes, dumy_target, inp_dim)

        return detections
