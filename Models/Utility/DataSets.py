from __future__ import print_function, division
import pandas as pd
import pydicom as pyd
from skimage import io, transform
import os
from Models.Utility.ResourceProvider import *
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

class ImgDataset(Dataset):
    def __int__(self, csv_file, transform=None):
        self.img_data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.img_data)

    def __getitem__(self, item):
        if torch.is_tensor(item):
            item = item.tolist()

        image = pyd.dcmread(self.img_data[item][0]).pixel_array
        target = self.img_data[item][1]