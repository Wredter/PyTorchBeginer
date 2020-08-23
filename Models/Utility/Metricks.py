import torch

from Models.Utility.Utility import jaccard, point_form, box_form
from Models.SSD.Utility import decode


class Metrics:
    def __init__(self):
        self.precisions = []
        self.recalls = []
        self.accuracys = []
        self.false_alarm_ratios = []
        self.miss_rates = []
        self.TNRs = []
        self.balanced_accs = []
        self.map = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.not_recognized = 0
        self.recognised = 0

    def stastic_prep_YOLO(self, net_output_loc, net_output_cls, gtb, img, conf_thres=0.48, jacc_thres=0.4):
        net_output_loc = net_output_loc / img.shape[1]
        net_output_loc = net_output_loc.to(self.device)
        target_bb = gtb[2:].unsqueeze(0).to(self.device)
        gtobj_mask = match_bb_with_gtb(net_output_loc, target_bb, jacc_thres)
        pobj_mask = net_output_cls.ge(conf_thres)
        tp = 0
        fp = 0
        fn = 0
        tn = 0
        for pred, gt in zip(pobj_mask, gtobj_mask):
            if pred and gt:
                tp += 1
            if pred and not gt:
                fp += 1
            if not pred and gt:
                fn += 1
            if not pred and not gt:
                tn += 1
        if tp + fp > 0 and fn > 0:
            self.precisions.append(tp / (tp + fp))
            self.recalls.append(tp / (tp + fn))
            self.accuracys.append((tp + tn)/(tp + tn + fp + fn))
            self.false_alarm_ratios.append(fp / (fp + tn))
            self.miss_rates.append(fn / (tp + fn))
            self.TNRs.append(tn/(tn + fp))
            self.balanced_accs.append(((tn/(tn + fp)) + (tp / (tp + fn))) / 2)
            self.recognised += 1
        else:
            self.not_recognized += 1

    def statistic_prep(self, net_output_loc, net_output_cls, target_loc, target_cls, dbox, conf_thres=0.48, jacc_thres=0.25):
        net_output_loc = net_output_loc.to(self.device)
        net_output_loc = decode(net_output_loc, dbox)
        target_loc = target_loc.to(self.device)
        gtobj_mask = jaccard(point_form(net_output_loc), point_form(target_loc))
        gtobj_mask = gtobj_mask.ge(jacc_thres)
        pobj_mask = net_output_cls.ge(conf_thres)
        tp = 0
        fp = 0
        fn = 0
        tn = 0
        for pred, gt in zip(pobj_mask, gtobj_mask):
            if pred and gt:
                tp += 1
            if pred and not gt:
                fp += 1
            if not pred and gt:
                fn += 1
            if not pred and not gt:
                tn += 1
        if tp + fp > 0 and fn > 0:
            self.precisions.append(tp / (tp + fp))
            self.recalls.append(tp / (tp + fn))
            self.accuracys.append((tp + tn) / (tp + tn + fp + fn))
            self.false_alarm_ratios.append(fp / (fp + tn))
            self.miss_rates.append(fn / (tp + fn))
            self.TNRs.append(tn / (tn + fp))
            self.balanced_accs.append(((tn / (tn + fp)) + (tp / (tp + fn))) / 2)
            self.recognised += 1
        else:
            self.not_recognized += 1

    def mAP(self):
        precision = torch.FloatTensor(self.precisions)
        recall = torch.FloatTensor(self.recalls)
        list = []
        pr = torch.cat((recall.unsqueeze(0), precision.unsqueeze(0)), 0)
        _, i = recall.sort(0)
        for idx in i:
            pr_part = pr[:, idx]
            list.append(pr_part)
        max_prec = 0
        min_recall = 0
        rec = []
        map = 0
        for idx2, val in enumerate(list):
            if val[1] >= max_prec:
                max_prec = val[1]
                min_recall = val[0]
            if val[1] < max_prec:
                rec.append([min_recall, max_prec])
                max_prec = val[1]
                min_recall = val[0]
            if idx2 == (len(list) - 1):
                rec.append([min_recall, max_prec])
        for i, box in enumerate(rec):
            if map == 0:
                map = box[0] * box[1]
            else:
                if box[1] < rec[i-1][1]:
                    map += (box[0].item() - rec[i-1][0])*box[1]
                else:
                    map = box[0] * box[1]
        self.map = map

    def calc_net_stat(self):
        if(len(self.precisions) != 0):
            self.mAP()
            avg_prec = sum(self.precisions) / len(self.precisions)
            avg_recal = sum(self.recalls) / len(self.recalls)
            avg_acc = sum(self.accuracys) / len(self.accuracys)
            avg_bacc = sum(self.balanced_accs) / len(self.balanced_accs)
            avg_far = sum(self.false_alarm_ratios) / len(self.false_alarm_ratios)
            avg_mr = sum(self.miss_rates) / len(self.miss_rates)
            recognition_ratio = self.recognised / (self.recognised + self.not_recognized)

            print(f"precision   : {avg_prec} \n"
                  f"recall      : {avg_recal} \n"
                  f"acuracy     : {avg_acc} \n"
                  f"balans acc  : {avg_bacc}\n"
                  f"false alarm : {avg_far}\n"
                  f"miss rate   : {avg_mr}\n"
                  f"AP          : {self.map}\n"
                  f"% rozpo     : {recognition_ratio}")
            return avg_prec, avg_recal, avg_acc, avg_bacc, avg_far, avg_mr, self.map, recognition_ratio
        else:
            print("no detections")


def match_bb_with_gtb(output, target, conf_thres):
    IoUs = jaccard(point_form(output), point_form(target))
    IoUs_mask = IoUs.ge(conf_thres)
    return IoUs_mask
