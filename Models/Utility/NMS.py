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
    keep = predictions.new()
    pred_score = torch.cat((predictions, scores.unsqueeze(-1)), 1)
    counter = 0
    v, i = pred_score.sort(descending=True)
    pred_score = pred_score[i[:top_detections, 4], :]

    while pred_score.numel() > 0:
        val, idx = pred_score.max(dim=0)
        keep = torch.cat((keep, pred_score[idx[4]].unsqueeze_(0)), 0)
        if len(keep.shape) < 2:
            keep.unsqueeze_(0)
        deb1 = pred_score[:, :4]
        deb2 = keep[counter, :4].unsqueeze(0)
        IoU = jaccard(deb1, deb2)
        IoU_mask = IoU.le(threshold).expand_as(pred_score)
        pred_score = pred_score * IoU_mask.float()
        pred_score = pred_score[pred_score.abs().sum(dim=1) != 0]
        counter += 1

    return keep
