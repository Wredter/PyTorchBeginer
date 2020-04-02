import numpy as np
import matplotlib.pyplot as ptl
import matplotlib.patches as patches
from torch.autograd import Variable
from Models.Utility.NMS import NMS
import torch

from Models.Utility.Utility import jaccard, point_form


def compare_prediction_with_bbox(bboxes, grand_truth_bb, grand_truth_cls, iou_tres, variance):
    ###############       TEST         #####################################
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
        #torch.set_printoptions(profile='full')
        conf[batch] = temp_conf
        #torch.set_printoptions(profile='default')
    return matches, conf


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
    factors = [0.1, 0.2]
    # dist b/t match center and prior's center
    g_cxcy = gt_box[:, :2] - default_box[:, :2]
    # encode variance
    g_cxcy /= (factors[0] * default_box[:, 2:])
    # match wh / prior wh
    g_wh = gt_box[:, 2:] / default_box[:, 2:]
    g_wh = torch.log(g_wh) / factors[1]
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
    factors = [0.2, 0.1]
    boxes = torch.cat((
        priors[:, :2] + loc[:, :2] * factors[0] * priors[:, 2:],
        priors[:, 2:] * torch.exp(loc[:, 2:] * factors[1])), 1)
    return boxes


def show_areas(image, targets, predictions, classes):
    """Show image with marked areas"""
    if len(image.shape) > 2:
        width = image.shape[1]
        height = image.shape[2]
    else:
        width = image.shape[1]
        height = image.shape[0]
    predictions = lbwh_form(predictions)
    targets = lbwh_form(targets)

    fig, ax = ptl.subplots(1)
    if type(image) is np.ndarray:
        ax.imshow(image)
    else:
        ax.imshow(image.cpu()[0])

    for i in range(predictions.size()[0]):
        rect = [0, 0, 0, 0]
        rect[0] = predictions[i][0] * width
        rect[1] = predictions[i][1] * height
        rect[2] = predictions[i][2] * width
        rect[3] = predictions[i][3] * height
        rect = patches.Rectangle((rect[0], rect[1]),
                                 rect[2],
                                 rect[3],
                                 linewidth=1,
                                 edgecolor='orange',
                                 facecolor='none')
        rect.set_label("prediction")
        ax.add_patch(rect)
    for i in range(targets.size()[0]):
        rect = [0, 0, 0, 0]
        rect[0] = targets[i][0] * width
        rect[1] = targets[i][1] * height
        rect[2] = targets[i][2] * width
        rect[3] = targets[i][3] * height
        rect = patches.Rectangle((rect[0], rect[1]),
                                 rect[2],
                                 rect[3],
                                 linewidth=1,
                                 edgecolor='r',
                                 facecolor='none')
        rect.set_label("Target")
        ax.add_patch(rect)
    ax.legend()
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


def final_detection(locations, scores, default_boxes, threshold=0.5, top_detections=200):
    batch_size = locations.shape[0]
    cls_num = scores.shape[2]
    all_batch_detect = locations.new()
    for x in range(batch_size):
        f_loc = decode(locations[x], default_boxes)
        for cls in range(1, cls_num):
            f_scr = scores[x, :, cls]
            detect = NMS(f_loc, f_scr, threshold, top_detections)
            detect = torch.cat((detect, cls_num), -1)
            all_batch_detect = torch.cat((all_batch_detect, detect), 0)
        if len(all_batch_detect.shape) == 2:
            all_batch_detect.unsqueeze_(0)
    return all_batch_detect
