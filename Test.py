import torch
from Models.YOLO.config.parser import parse_cfg, create_modules
from Models.YOLO.deepnet import *
import os

x = os.getcwd()
x += "\\Models\\YOLO\\config\\yolov3.cfg"
print(x)
YOLO = YOLODeepNet(x)
#YOLO.forward(0, True)

