import os
import torch


def prep_paths():
    train = os.getcwd()
    train += "\\Data\\preped_data_mass_train.csv"
    test = os.getcwd()
    test += "\\Data\\preped_data_mass_test.csv"
    dummy_test = os.getcwd()
    dummy_test += "\\Data\\Dumy_test.csv"
    x = os.getcwd()
    x += "/Models/YOLO/config/yolov3.cfg"
    class_names = ["patologia"]
    return train, test, dummy_test, class_names, x


def jaccard(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.  Here we operate on
    ground truth boxes and default boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]
        box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_priors,4]
    Return:
        jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    area_b = ((box_b[:, 2]-box_b[:, 0]) *
              (box_b[:, 3]-box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]


def intersect(box_a, box_b):
    """ We resize both tensors to [A,B,2] without new malloc:
    [A,2] -> [A,1,2] -> [A,B,2]
    [B,2] -> [1,B,2] -> [A,B,2]
    Then we compute the area of intersect between box_a and box_b.
    Args:
      box_a: (tensor) bounding boxes, Shape: [A,4].
      box_b: (tensor) bounding boxes, Shape: [B,4].
    Return:
      (tensor) intersection area, Shape: [A,B].
    """
    A = box_a.size(0)
    B = box_b.size(0)
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                       box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                       box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=1e-16)
    return inter[:, :, 0] * inter[:, :, 1]


def point_form(boxes):
    """ Convert boxes to (xmin, ymin, xmax, ymax)
    Args:
        boxes: (tensor) center-size boxes
    Return:
        boxes: (tensor) Converted xmin, ymin, xmax, ymax boxes.
    """
    return torch.cat((boxes[:, :2] - boxes[:, 2:]/2,     # xmin, ymin
                     boxes[:, :2] + boxes[:, 2:]/2), 1)  # xmax, ymax


def box_form(boxes):
    """ Convert boxes to (x_center, y_center, width, height)
    Args:
        boxes: (tensor) Converted xmin, ymin, xmax, ymax boxes.
    Return:
        boxes: (tensor) center-size boxes
    """
    return torch.cat(((boxes[:, :2] + boxes[:, 2:])/2,     # xmin, ymin
                     boxes[:, 2:] - boxes[:, :2]), 1)  # xmax, ymax



def list_avg(my_list):
    return sum(my_list) / len(my_list)
