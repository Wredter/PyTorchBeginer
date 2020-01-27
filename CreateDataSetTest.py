from __future__ import print_function, division
import os
from Models.Utility.ResourceProvider import *
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

y = os.getcwd()
y += "\\Data\\mass_case_description_test_set.csv"
test = ResourceProvider(y, "D:\\DataSet\\CBIS-DDSM\\", "D:\\DataSet\\ROI\\CBIS-DDSM\\")
data = test.prepare_data()
print(data)
