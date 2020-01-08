import torch
from Models.YOLO.config.parser import parse_cfg, create_modules
from Models.YOLO.darknet import *
import os
from Models.Utility.ResourceProvider import *
from Models.Utility.DataLoader import test_dicom_reader
import matplotlib.pyplot as plt

#x = os.getcwd()
#x += "\\Models\\YOLO\\config\\yolov3.cfg"
#print(x)
y = os.getcwd()
y += "\\Data\\only_CC_set.csv"
test = ResourceProvider(y, "D:\\DataSet\\CBIS-DDSM\\", "D:\\DataSet\\ROI\\CBIS-DDSM\\")
test.read()
for x, y in test.columns.items():
  print(x, y)
img, roi = test.read_dicom_file(0)
plt.imshow(img)
plt.show()
plt.imshow(roi)
plt.show()
#test_dicom_reader()
print("skończyłem")
#YOLO = YOLODeepNet(x)
#YOLO.forward(0, True)

