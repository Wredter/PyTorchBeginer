import math
import numpy as np
import matplotlib.pyplot as ptl
import matplotlib.patches as patches
from torch.autograd import Variable
import torch


def compare_prediction_with_bbox(bboxes, grand_truth_bb, grand_truth_cls, iou_tres, variance):
    conf = torch.zeros((int(grand_truth_bb.size()[0]), int(bboxes.size()[0])), dtype=torch.int64)
    matches = torch.zeros((int(grand_truth_bb.size()[0]), int(bboxes.size()[0]), 4))
    for batch in range(grand_truth_bb.size(0)):
        temp_bbox_ious = jaccard(point_form(grand_truth_bb[batch]),
                                 bboxes)
        temp_bbox_ious = Variable(temp_bbox_ious.to("cuda")) if grand_truth_bb.is_cuda else Variable(temp_bbox_ious)
        # (Bipartite Matching)
        # [1,num_objects] best prior for each ground truth
        best_prior_overlap, best_prior_idx = temp_bbox_ious.max(1, keepdim=True)
        # [1,num_priors] best ground truth for each bbox
        best_truth_overlap, best_truth_idx = temp_bbox_ious.max(0, keepdim=True)
        best_truth_idx.squeeze_(0)
        best_truth_overlap.squeeze_(0)
        best_prior_idx.squeeze_(1)
        best_prior_overlap.squeeze_(1)
        best_truth_overlap.index_fill_(0, best_prior_idx, 2)  # ensure best prior
        for j in range(best_prior_idx.size(0)):
            best_truth_idx[best_prior_idx[j]] = j
        temp_matches = grand_truth_bb[batch][best_truth_idx]  # Shape: [num_priors,4]
        temp_conf = grand_truth_cls[batch][best_truth_idx]  # Shape: [num_priors]
        temp_conf[best_truth_overlap < iou_tres] = 0  # label as background
        temp_matches = encode(temp_matches, bboxes)
        matches[batch] = temp_matches
#        torch.set_printoptions(profile='full')
        conf[batch] = temp_conf
#        torch.set_printoptions(profile='default')
    return matches, conf


def intersect(box_a, box_b):
    """ We resize both tensors to [A,B,2] without new malloc:
    [A,2] -> [A,1,2] -> [A,B,2]
    [B,2] -> [1,B,2] -> [A,B,2]
    Then we compute the area of intersect between box_a and box_b.
    Args:
      box_a: (tensor) bounding boxes, Shape: [A,4].
      box_b: (tensor) bounding boxes, Shape: [B,4].
    Return:
      (tensor) intersection area, Shape: [A,B].
    """
    A = box_a.size(0)
    B = box_b.size(0)
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                       box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                       box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, :, 0] * inter[:, :, 1]


def jaccard(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.  Here we operate on
    ground truth boxes and default boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]
        box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_priors,4]
    Return:
        jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    area_b = ((box_b[:, 2]-box_b[:, 0]) *
              (box_b[:, 3]-box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]


def point_form(boxes):
    """ Convert prior_boxes to (xmin, ymin, xmax, ymax)
    representation for comparison to point form ground truth data.
    Args:
        boxes: (tensor) center-size default boxes from priorbox layers.
    Return:
        boxes: (tensor) Converted xmin, ymin, xmax, ymax form of boxes.
    """
    return torch.cat((boxes[:, :2] - boxes[:, 2:]/2,     # xmin, ymin
                     boxes[:, :2] + boxes[:, 2:]/2), 1)  # xmax, ymax


def encode(gt_box, default_box):
    """Encode the variances from the priorbox layers into the ground truth boxes
    we have matched (based on jaccard overlap) with the prior boxes.
    Args:
        gt_box: (tensor) Coords of ground truth for each prior in point-form
            Shape: [num_priors, 4].
        default_box: (tensor) Prior boxes in center-offset form
            Shape: [num_priors,4].
    Return:
        encoded boxes (tensor), Shape: [num_priors, 4]
    """

    # dist b/t match center and prior's center
    g_cxcy = gt_box[:, :2] - default_box[:, :2]
    # encode variance
    g_cxcy /= default_box[:, 2:]
    # match wh / prior wh
    g_wh = gt_box[:, 2:] / default_box[:, 2:]
    g_wh = torch.log(g_wh)
    # return target for smooth_l1_loss
    return torch.cat([g_cxcy, g_wh], 1)  # [num_priors,4]


# Adapted from https://github.com/Hakuyume/chainer-ssd
def decode(loc, priors):
    """Decode locations from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        loc (tensor): location predictions for loc layers,
            Shape: [num_priors,4]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors,4].
    Return:
        decoded bounding box predictions
    """

    boxes = torch.cat((
        priors[:, :2] + loc[:, :2] * priors[:, 2:],
        priors[:, 2:] * torch.exp(loc[:, 2:])), 1)
    return boxes


def show_areas(image, boxes, classes):
    """Show image with marked areas"""
    width = image.shape[1]
    height = image.shape[2]
    boxes = lbwh_form(boxes)

    fig, ax = ptl.subplots(1)
    ax.imshow(image.cpu()[0])

    for i in range(boxes.size()[0]):
        rect = boxes[i] * width
        rect = patches.Rectangle((rect[0], rect[1]), rect[2], rect[3], linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

    ptl.show()


def lbwh_form(boxes):
    """ Convert prior_boxes to (xmin, ymin, xmax, ymax)
    representation for comparison to point form ground truth data.
    Args:
        boxes: (tensor) center-size default boxes from priorbox layers.
    Return:
        boxes: (tensor) Converted xmin, ymin, xmax, ymax form of boxes.
    """
    return torch.cat((boxes[:, :2] - boxes[:, 2:]/2,     # xmin, ymin
                     boxes[:, 2:]), 1)  # xmax, ymax