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
        self.ap = 0
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

    def statistic_prep(self, net_output_loc, net_output_cls, target_loc, target_cls, dbox, conf_thres=0.5, jacc_thres=0.5):
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

    def AP(self):
        precision = torch.FloatTensor(self.precisions)
        recall = torch.FloatTensor(self.recalls)
        recall_values = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        interpolated_prec = []
        for recal_i, recall_v in enumerate(recall_values):
            if recall_v < 1:
                ge_mask = recall.ge(recall_v)
                lt_mask = recall.lt(recall_values[recal_i + 1])
                temp_prec = precision * ge_mask.float()
                temp_prec = temp_prec * lt_mask.float()
                prec = max(temp_prec)
                interpolated_prec.append(prec)
            else:
                ge_mask = recall.ge(recall_v)
                temp_prec = precision * ge_mask.float()
                prec = max(temp_prec)
                interpolated_prec.append(prec)
        self.ap = sum(interpolated_prec) / len(interpolated_prec)

    def calc_net_stat(self):
        if(len(self.precisions) != 0):
            self.AP()
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
                  f"AP          : {self.ap}\n"
                  f"% rozpo     : {recognition_ratio}")
            return avg_prec, avg_recal, avg_acc, avg_bacc, avg_far, avg_mr, self.ap, recognition_ratio
        else:
            print("no detections")


def match_bb_with_gtb(output, target, conf_thres):
    IoUs = jaccard(point_form(output), point_form(target))
    IoUs_mask = IoUs.ge(conf_thres)
    return IoUs_mask
