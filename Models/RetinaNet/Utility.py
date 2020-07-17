import itertools
from math import sqrt

import numpy as np

from Models.SSD.Utility import show_areas, decode
from Models.Utility.NMS import NMS
import torch


def nms_prep(img, targets, network_output_loc, network_output_cls, db, conf_threshold=0.5, epoch=-1):
    network_output_loc = decode(network_output_loc, db)
    network_output = torch.cat((network_output_loc, network_output_cls), 1)
    nms_output = NMS(network_output[:, :4], network_output[:, 4])
    targets_loc = targets[:, :4]
    if nms_output.shape[0] != 0:
        if epoch >= 0:
            show_areas(img, targets_loc, nms_output[:, :4], None, f"Detections epoch: {epoch}")
        else:
            show_areas(img, targets_loc, nms_output[:, :4], None, "Detections")


class AnchorBoxes(object):
    def __init__(self, fig_size, feat_size, steps, scales, aspect_ratios, variance,
                 scale_xy=0.1, scale_wh=0.2):

        self.feat_size = feat_size
        self.fig_size = fig_size
        self.variance = variance
        self.scale_xy_ = scale_xy
        self.scale_wh_ = scale_wh

        # According to https://github.com/weiliu89/caffe
        # Calculation method slightly different from paper
        self.steps = steps
        self.scales = scales

        fk = fig_size/np.array(steps)
        self.aspect_ratios = aspect_ratios

        self.default_boxes = []
        # size of feature and number of feature
        for idx, sfeat in enumerate(self.feat_size):

            sk = 1/sfeat
            sk2 = sk/2
            all_sizes = [(sk, sk), (sk, sk2), (sk2, sk)]

            size = len(all_sizes)
            for idy in range(size):
                w, h = all_sizes[idy]
                w1 = (w * 1.26)
                w2 = (w * 1.5874)
                h1 = (h * 1.26)
                h2 = (h * 1.5874)
                all_sizes.append((w1, h1))
                all_sizes.append((w2, h2))
            for w, h in all_sizes:
                for i, j in itertools.product(range(sfeat), repeat=2):
                    cx, cy = (j+0.5)/fk[idx], (i+0.5)/fk[idx]
                    self.default_boxes.append((cx, cy, w, h))

        self.dboxes = torch.tensor(self.default_boxes)
        self.dboxes.clamp_(min=0, max=1)
        # For IoU calculation
        self.dboxes_ltrb = self.dboxes.clone()
        self.dboxes_ltrb[:, 0] = self.dboxes[:, 0] - 0.5 * self.dboxes[:, 2]
        self.dboxes_ltrb[:, 1] = self.dboxes[:, 1] - 0.5 * self.dboxes[:, 3]
        self.dboxes_ltrb[:, 2] = self.dboxes[:, 0] + 0.5 * self.dboxes[:, 2]
        self.dboxes_ltrb[:, 3] = self.dboxes[:, 1] + 0.5 * self.dboxes[:, 3]

    @property
    def scale_xy(self):
        return self.scale_xy_

    @property
    def scale_wh(self):
        return self.scale_wh_

    def __call__(self, order="ltrb"):
        if order == "ltrb": return self.dboxes_ltrb
        if order == "xywh": return self.dboxes


def retinabox300():
    figsize = 300
    feat_size = [38, 19, 10, 5, 3]
    steps = [8, 16, 32, 64, 100]
    variance = [0.1, 0.2]
    # use the scales here: https://github.com/amdegroot/ssd.pytorch/blob/master/data/config.py
    scales = [21, 45, 99, 153, 207, 261]
    aspect_ratios = [[2], [2], [2], [2], [2], [2]]
    dboxes = AnchorBoxes(figsize, feat_size, steps, scales, aspect_ratios, variance)
    return dboxes


def retinabox600():
    figsize = 600
    feat_size = [38, 19, 10, 5, 3]
    steps = [8, 15, 30, 60, 100]
    variance = [0.1, 0.2]
    # use the scales here: https://github.com/amdegroot/ssd.pytorch/blob/master/data/config.py
    scales = [21, 45, 99, 153, 207, 261]
    aspect_ratios = [[2], [2], [2], [2], [2], [2]]
    dboxes = AnchorBoxes(figsize, feat_size, steps, scales, aspect_ratios, variance)
    return dboxes