import torch
from Models.YOLO.config.parser import parse_cfg, create_modules
from Models.YOLO.darknet import *
import os
import pydicom as py
from Models.Utility.ResourceProvider import *
from Models.Utility.DataLoader import test_dicom_reader
import matplotlib.pyplot as plt

# mass_case_description_test_set.csv
# only_MLO_set.csv
# only_CC_set.csv
y = os.getcwd()
x = "D:\\DataSet\\CBIS-DDSM\\Mass-Test_P_01787_LEFT_CC\\1.3.6.1.4.1.9590.100.1.2.374375446912980469729505435053598798252\\1.3.6.1.4.1.9590.100.1.2.246616253112617539506581039831772141703\\000000.dcm"
y = "D:\\DataSet\\ROI\\CBIS-DDSM\\Mass-Test_P_01787_LEFT_CC_1\\1.3.6.1.4.1.9590.100.1.2.137178661810518177826975776450613138254\\1.3.6.1.4.1.9590.100.1.2.131421612911608368538869006151649317483\\000001.dcm"


ds = py.dcmread(x)
ds2 = py.dcmread(y)

plt.imshow(ds.pixel_array)
plt.show()
plt.imshow(ds2.pixel_array)
plt.show()