import torch
from Models.YOLO.config.parser import parse_cfg, create_modules
from Models.YOLO.darknet import *
import os
from Models.Utility.ResourceProvider import *
from Models.Utility.DataLoader import test_dicom_reader
import matplotlib.pyplot as plt

x = os.getcwd()
x += "/Models/YOLO/config/yolov3.cfg"
model = Darknet(x)
wejscie = get_test_input()
dumy_target = torch.FloatTensor([[0, 0, 0.5, 0.5, 0.1, 0.1], [0, 0, 0.25, 0.25, 0.01, 0.01]])
if torch.cuda.is_available():
    dumy_target.cuda()
prediction, loss = model(wejscie, dumy_target)
print(prediction)
print(loss)





