from Models.SSD.Utility import show_areas, decode
from Models.Utility.NMS import NMS
import torch


def nms_prep(img, targets, network_output_loc, network_output_cls, db, conf_threshold=0.5):
    network_output_loc = decode(network_output_loc, db)
    network_output = torch.cat((network_output_loc, network_output_cls), 1)
    obj_mask = network_output[:, 4].ge(conf_threshold)
    obj_mask = obj_mask.unsqueeze(1).expand_as(network_output)
    network_output = network_output * obj_mask.float()
    network_output = network_output[network_output.abs().sum(dim=1) != 0]
    if network_output.shape[0] == 0:
        return
    nms_output = NMS(network_output[:, :4], network_output[:, 4])
    targets_loc = targets[:, :4]
    show_areas(img, targets_loc, nms_output[:, :4], None, "Detections")