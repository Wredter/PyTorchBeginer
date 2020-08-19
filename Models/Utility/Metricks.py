import torch

from Models.Utility.Utility import jaccard, point_form, box_form

class Metrics ():
    def __init__(self):
        self.precisions = []
        self.recalls = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.not_recognized = 0

    def mAP(self, net_output_loc, net_output_cls, gtb, det_num):
        gtb = gtb.to(self.device)
        if net_output_loc is not None:
            net_output_loc = net_output_loc.to(self.device)
            net_output_cls = net_output_cls.to(self.device)
            t1 = point_form(gtb)
            t2 = point_form(net_output_loc)
            ious = jaccard(t1, t2).squeeze(0)
            mask_output_cls = net_output_cls.ge(0.45)
            for i in range(det_num):
                tp = 0
                fp = 0
                fn = 0
                cls = mask_output_cls[i].item()
                iou = ious[i].item()
                if iou > 0.5 and cls:
                    tp += 1
                if iou > 0.5 and not cls:
                    fp += 1
                if iou < 0.5:
                    fn += 1
                if tp == 0 and fp == 0:
                    self.precisions.append(0.0)
                else:
                    self.precisions.append(tp / (tp + fp))
                self.recalls.append(tp / (tp + fn))
        else:
            self.not_recognized += 1
