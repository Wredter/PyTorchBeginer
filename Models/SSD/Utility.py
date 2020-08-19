import numpy as np
import matplotlib.pyplot as ptl
import matplotlib.patches as patches
from torch.autograd import Variable
from Models.Utility.NMS import NMS
from Models.Utility.Utility import jaccard, point_form, box_form
import torch


# Not suited for multiclass problems
def compare_trgets_with_bbox(bboxes, grand_truth_bb, grand_truth_cls, iou_tres, show_maches_=False, img=None):
    IoUs = jaccard(bboxes, point_form(grand_truth_bb))
    IoUs_mask = IoUs.ge(iou_tres)
    if show_maches_:
        boxes = box_form(bboxes)
        boxes = boxes * IoUs_mask.expand_as(boxes).float()
        boxes = boxes[boxes.abs().sum(dim=1) != 0]
        show_areas(img, grand_truth_bb, boxes, None, "Maches")
    encoded_boxes = encode(grand_truth_bb, box_form(bboxes))
    mached_cls = grand_truth_cls.expand(encoded_boxes.shape[0], 1) * IoUs_mask.float()
    return encoded_boxes, mached_cls, IoUs_mask


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


def show_areas(image, targets, predictions, classes, plot_title=None):
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
    #ax.legend()
    if plot_title:
        ax.set_title(plot_title)
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


def final_detection(locations, scores, default_boxes, class_to_show, threshold=0.5, top_detections=50, encoding="encoded"):
    batch_size = locations.shape[0]
    cls_num = scores.shape[2]
    if class_to_show > cls_num:
        print("Grrrrrrr, zła liczba")
        return
    all_batch_detect = []
    for x in range(batch_size):
        if encoding == "encoded":
            f_loc = decode(locations[x], default_boxes)
        elif encoding == "delta":
            f_loc = default_boxes - locations[x]
        elif encoding == "d_delta":
            f_loc = default_boxes - decode(locations[x], default_boxes)
        else:
            f_loc = locations[x]
        f_scr = scores[x, :, class_to_show]
        detect = NMS(f_loc, f_scr, threshold, top_detections)
        all_batch_detect.append(detect)
    return all_batch_detect


def generate_plots(ploc, plabel, db, targets, imgs, batch_size, encoding, class_to_show):
    final = final_detection(ploc, plabel, db, class_to_show, top_detections=50)
    targets_loc = targets[..., :4]
    if "raw" in encoding:
        raw = final_detection(ploc, plabel, db, class_to_show, encoding="None")
    if "delta" in encoding:
        delta = final_detection(ploc, plabel, db, class_to_show, encoding="delta")
    if "d_delta" in encoding:
        decoded_delta = final_detection(ploc, plabel, db, class_to_show, encoding="d_delta")
    for x in range(batch_size):
        print(f'Target: {targets[x].tolist()}')
        print(f'Decoded: {final[x].tolist()}')
        if final[x].sum() == 0:
            continue
        show_areas(imgs[x], targets_loc[x], final[x][:, :4], 0, plot_title="Decoded")
        if "raw" in encoding:
            print(f'Raw: {raw[x].tolist()}')
            show_areas(imgs[x], targets_loc[x], raw[x][:, :4], 0, plot_title="Raw")
        if "delta" in encoding:
            print(f'Delta {delta[x].tolist()}')
            show_areas(imgs[x], targets_loc[x], delta[x][:, :4], 0, plot_title="delta")
        if "d_delta" in encoding:
            print(f'Decoded Delta {decoded_delta[x].tolist()}')
            show_areas(imgs[x], targets_loc[x], decoded_delta[x][:, :4], 0, plot_title="Decoded Delta")
    return 0


