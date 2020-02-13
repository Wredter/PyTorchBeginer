import torch
from Models.YOLO.config.parser import parse_cfg, create_modules
from Models.YOLO.darknet import *
import os
from Models.Utility.ResourceProvider import *
from Models.Utility.DataLoader import test_dicom_reader
import matplotlib.pyplot as plt

# mass_case_description_test_set.csv
# only_MLO_set.csv
# only_CC_set.csv
y = os.getcwd()
x = y + "\\Data\\only_MLO_set.csv"
y += "\\Data\\only_CC_set.csv"
CC = ResourceProvider(y, "D:\\DataSet\\CBIS-DDSM\\", "D:\\DataSet\\ROI\\CBIS-DDSM\\")
CC.read()
MLO = ResourceProvider(x, "D:\\DataSet\\CBIS-DDSM\\", "D:\\DataSet\\ROI\\CBIS-DDSM\\")
MLO.read()

i_CC, r_CC, i_p_CC, r_p_CC = CC.read_dicom_file(1, True)
i_MLO, r_MLO, i_p_MLO, r_p_MLO = MLO.read_dicom_file(1, True)
plt.imshow(i_CC)
plt.show()
plt.imshow(r_CC)
plt.show()
plt.imshow(i_MLO)
plt.show()
plt.imshow(r_MLO)
plt.show()
print("skończyłem")