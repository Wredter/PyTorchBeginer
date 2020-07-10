from Models.RetinaNet.Utility import retinabox300
from Models.SSD.DefaultsBox import dboxes300
from Models.SSD.DefaultsBox import *
from Models.SSD.Utility import show_areas
from Models.Utility.Utility import point_form, box_form

"""
t = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
t2 = torch.gather(t, 0, torch.tensor([[[0, 0], [0, 0]], [[0, 1], [0, 0]]]))
print(t2)
"""

box = dboxes300()
db = box(order="ltrb")
db = box_form(db)
db = db.view(-1, 37, 4)
image = torch.zeros((3, 300, 300), dtype=torch.long)
target = torch.zeros(1, 4)
shape = db.shape[0]
for i in range(db.shape[0]):
    print(f"Test {i} : {db[i]}")
    show_areas(image, target, db[i], None, f"Test {i}")
print("boxes")

