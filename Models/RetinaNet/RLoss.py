
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from Models.SSD.Utility import compare_trgets_with_bbox


class RLoss(nn.Module):
    def __init__(self, num_classes=1):
        super(RLoss, self).__init__()
        self.num_classes = num_classes
        self.alpha = 0.25
        self.gamma = 2
        self.ce_loss = torch.nn.\
            BCEWithLogitsLoss(reduction="none")
        self.sf_loss = torch.nn.\
            SmoothL1Loss(reduction="sum")

    def forward(self, loc_preds, cls_preds, loc_targets,
                cls_targets, target_mask):
        cls_num = cls_preds.size(-1)
        pos_num = target_mask.long().sum().item()

        mask = target_mask.expand_as(loc_preds)       # [N,#anchors,4]
        masked_loc_preds = loc_preds[mask].view(-1, 4)      # [#pos,4]
        masked_loc_targets = loc_targets[mask].view(-1, 4)  # [#pos,4]
        loc_loss = self.sf_loss(masked_loc_preds,
                                masked_loc_targets)

        cls_preds = cls_preds.view(-1, cls_num)
        cls_targets = cls_targets.view(-1, cls_num)
        log_pt = self.ce_loss(cls_preds, cls_targets)
        pt = torch.exp(-log_pt)
        f_loss = self.alpha * ((1 - pt)**self.gamma) * log_pt
        f_loss = torch.clamp(f_loss, min=1e-16)
        f_loss = f_loss.sum()

        total_loss = (loc_loss + f_loss) / pos_num
        print(f"loc_loss: {loc_loss}, cls_loss: {f_loss}, total_loss: {total_loss}, target_num: {pos_num}")
        if pos_num == 0:
            return 0
        return total_loss

