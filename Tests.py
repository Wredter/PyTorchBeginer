import torch
import torch.nn as nn
from Models.SSD.DefaultsBox import *
from Models.SSD.Utility import show_areas
from Models.Utility.Utility import point_form, box_form
"""
t = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
t2 = torch.gather(t, 0, torch.tensor([[[0, 0], [0, 0]], [[0, 1], [0, 0]]]))
print(t2)
"""

box = retinabox300()
db = box(order="xywh")
image = torch.zeros((3, 300, 300), dtype=torch.long)
target = torch.zeros(1, 4)
show_areas(image, target, db, None, "Test")
print("boxes")

