import math
import numpy as np
from itertools import product
from torch.autograd import Variable
from Models.YOLO.utility import bbox_iou

import torch


class BoundingBox:
    def __init__(self):
        self.num_classes = 1
        self.image_size = 300
        self.num_priors = [1, 2, 2, 2, 1, 1]
        self.feature_maps = [38, 19, 10, 5, 3, 1]
        self.min_size = [6, 12, 28, 50, 90, 280]
        self.max_size = [10, 20, 35, 70, 120, 320]

    def generatebbox(self):
        bbox = []
        for i, feature_map in enumerate(self.feature_maps):
            stride = self.image_size / feature_map
            for x, y in product(range(feature_map), repeat=2):
                # Center of each grid cell
                center_x = (x + 0.5) / stride
                center_y = (y + 0.5) / stride
                width_lenght = []
                width_lenght.append(self.min_size[i] / self.image_size)
                width_lenght.append(math.sqrt(width_lenght[0] * (self.max_size[i]/self.image_size)))
                # Aspect ratios
                bbox += [center_x, center_y, width_lenght[0], width_lenght[0]]
                bbox += [center_x, center_y, width_lenght[1], width_lenght[1]]
                for ar in range(self.num_priors[i]):
                    bbox += [center_x, center_y, 2 * width_lenght[ar], width_lenght[ar]]
                    bbox += [center_x, center_y, width_lenght[ar], 2 * width_lenght[ar]]
        output = torch.tensor(bbox).view(-1, 4)
        output.clamp(min=0, max=1)
        return output


def compare_prediction_with_bbox(predictions, bboxes, grand_truth_bb, grand_truth_cls, iou_tres):
    bbox_ious = torch.zeros((int(grand_truth_bb.size()[0]), int(bboxes.size()[0])))
    for batch in range(grand_truth_bb.size(0)):
        bbox_ious[batch] = (bbox_iou(grand_truth_bb[batch, ...], bboxes, x1y1x2y2=False))
    # (Bipartite Matching)
    bbox_ious = Variable(bbox_ious.to("cuda")) if predictions.is_cuda else Variable(bbox_ious)
    # [1,num_objects] best prior for each ground truth
    best_prior_overlap, best_prior_idx = bbox_ious.max(1, keepdim=True)
    # [1,num_priors] best ground truth for each bbox
    best_truth_overlap, best_truth_idx = bbox_ious.max(0, keepdim=True)
    best_truth_idx.squeeze_(0)
    best_truth_overlap.squeeze_(0)
    best_prior_idx.squeeze_(1)
    best_prior_overlap.squeeze_(1)
    best_truth_overlap.index_fill_(0, best_prior_idx, 2)  # ensure best prior
    for j in range(best_prior_idx.size(0)):
        best_truth_idx[best_prior_idx[j]] = j
    matches = grand_truth_bb[best_truth_idx]  # Shape: [num_priors,4]
    conf = grand_truth_cls[best_truth_idx] + 1  # Shape: [num_priors]
    conf[best_truth_overlap < iou_tres] = 0  # label as background
    loc = encode(matches, priors, variances)
    loc_t[idx] = loc  # [num_priors,4] encoded offsets to learn
    conf_t[idx] = conf  # [num_priors] top class label for each prior

    return bbox_ious
