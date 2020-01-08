import torch
from Models.YOLO.config.parser import parse_cfg, create_modules
from Models.YOLO.darknet import *
import os
from Models.Utility.ResourceProvider import *
from Models.Utility.DataLoader import test_dicom_reader
import matplotlib.pyplot as plt

x = os.getcwd()
x += "\\Models\\YOLO\\config\\yolov3.cfg"
model = YOLODarkNet(x)
input = get_test_input()
prediction = model(input, torch.cuda.is_available())
print("COO KURWA")
print(prediction)


