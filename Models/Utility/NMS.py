import numpy as np
import torch
from Models.Utility.Utility import *


def NMS(predictions, scores, threshold=0.5, top_detections=50):
    """
    :param predictions: (Tensor) the locations preds for the img, Shape: [num_priors, 4]
    :param scores: (Tensor) the scores for single class in the img, Shape: [num_priors]
    :param threshold: (float) IoU threshold
    :param top_detections:
    :return: Best non-overlaping detections
    """
    keep = predictions.new()
    scores = scores.sigmoid()
    pred_score = torch.cat((predictions, scores.unsqueeze(-1)), 1)
    counter = 0
    obj_mask = scores.ge(threshold)
    pred_score = pred_score * obj_mask.unsqueeze(1).expand_as(pred_score).float()
    pred_score = pred_score[pred_score.abs().sum(dim=1) != 0]
    if pred_score.shape[0] < top_detections:
        top_detections = pred_score.shape[0]
    v, i = pred_score.sort(dim=0, descending=True)
    pred_score = pred_score[i[:top_detections, 4]]
    pred_score = nms_point_form(pred_score)
    pred_score.clamp(min=1e-16)
    while pred_score.numel() > 0:
        val, idx = pred_score.max(dim=0)
        keep = torch.cat((keep, pred_score[idx[4]].unsqueeze_(0)), 0)
        if len(keep.shape) < 2:
            keep.unsqueeze_(0)
        deb1 = pred_score[:, :4]
        deb2 = keep[counter, :4].unsqueeze(0)
        IoU = jaccard(deb1, deb2)
        IoU_mask = IoU.le(threshold)
        IoU_mask = IoU_mask.expand_as(pred_score)
        pred_score = pred_score * IoU_mask.float()
        pred_score = torch.cat((pred_score[:idx[4], :], pred_score[idx[4]+1:, :]), 0)
        pred_score = pred_score[pred_score.abs().sum(dim=1) != 0]

        counter += 1
    if keep.shape[0] == 0:
        print("NMS unexpeced")
        return keep
    return nms_box_form(keep)


def nms_point_form(boxes):
    scores = boxes[:, 4]
    boxes = boxes[:, :4]
    boxes = point_form(boxes)
    return torch.cat((boxes, scores.unsqueeze_(1)), 1)


def nms_box_form(boxes):
    scores = boxes[:, 4]
    boxes = boxes[:, :4]
    boxes = box_form(boxes)
    return torch.cat((boxes, scores.unsqueeze_(1)), 1)
