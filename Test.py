import torch
from Models.YOLO.config.parser import parse_cfg, create_modules
from Models.YOLO.deepnet import *
import os
from Models.Utility.DataLoader import test_dicom_reader

x = os.getcwd()
x += "\\Models\\YOLO\\config\\yolov3.cfg"
print(x)

test_dicom_reader()
print("skończyłem")
#YOLO = YOLODeepNet(x)
#YOLO.forward(0, True)

