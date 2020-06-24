import torch
import torch.nn as nn

from Models.Utility.Utility import point_form, box_form
"""
t = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
t2 = torch.gather(t, 0, torch.tensor([[[0, 0], [0, 0]], [[0, 1], [0, 0]]]))
print(t2)
"""

box = torch.tensor([[2, 2, 2, 2], [5.5, 4, 3, 6], [1, 1, 2, 2], [2, 3, 4, 2]])
point = point_form(box)
box2 = box_form(point)
print(point)
