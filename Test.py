import torch
from Models.YOLO.config.parser import parse_cfg
import os

x = os.getcwd()
x += "\\Models\\YOLO\\config\\yolov3.cfg"
print(x)

y = parse_cfg(x)
print(y)


