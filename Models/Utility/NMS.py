import numpy as np
import torch
from Models.Utility.Utility import jaccard


def NMS(predictions, scores, threshold=0.5, top_detections=200):
    """
    :param predictions: (Tensor) the locations preds for the img, Shape: [num_priors, 4]
    :param scores: (Tensor) the scores for single class in the img, Shape: [num_priors]
    :param threshold: (float) IoU threshold
    :param top_detections:
    :return: Best non-overlaping detections
    """
    x1 = predictions[:, 0]
    y1 = predictions[:, 1]
    x2 = predictions[:, 2]
    y2 = predictions[:, 3]
    areas = torch.mul(x2 - x1, y2 - y1)
    keep = predictions.new()
    pred_score = torch.cat((predictions, scores.unsqueeze(-1)), 1)
    counter = 0
    while pred_score.numel() > 0:
        val, idx = pred_score.max(dim=0)
        keep = torch.cat((keep, pred_score[idx[4]]), 0)
        if len(keep.shape) < 2:
            keep.unsqueeze_(0)
        deb1 = pred_score[:, :4]
        deb2 = keep[counter, :4].unsqueeze(0)
        IoU = jaccard(deb1, deb2)
        IoU_mask = IoU.le(threshold).expand_as(pred_score)
        pred_score = pred_score[IoU_mask]
        counter += 1

    return keep
