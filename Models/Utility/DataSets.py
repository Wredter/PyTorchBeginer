from __future__ import print_function, division

import cv2
import pandas as pd
import pydicom as pyd
import torch.nn.functional as F
from skimage import io, transform
import os
from Models.Utility.ResourceProvider import *
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


class ImgDataset(Dataset):
    def __init__(self, csv_file, img_size=416, normalized_labels=True,mod=None):
        self.img_data = pd.read_csv(csv_file, header=None)
        self.img_size = img_size
        self.normalized_labels = normalized_labels
        self.batch_count = 0
        self.mod = mod

    def __len__(self):
        return len(self.img_data)

    def __getitem__(self, item):
        if torch.is_tensor(item):
            item = item.tolist()

        image = pyd.dcmread(self.img_data.iloc[item][0]).pixel_array
        image = image[::8, ::8]
        image = image.astype(np.float32)
        image = torch.from_numpy(image)
        if len(image.shape) != 3:
            image = image.unsqueeze(0)
            h = image.shape[1]
            w = image.shape[2]
            image = image.expand((3, h, w))

        temp, h, w = image.shape
        h_factor, w_factor = (h, w) if self.normalized_labels else (1, 1)
        # Pad to square resolution
        img, pad = pad_to_square(image, 0)
        temp, padded_h, padded_w = img.shape
        image, _ = pad_to_square(image, 0)
        boxes = self.img_data.iloc[item][1]
        boxes = boxes.replace("[", "").replace("]", "").replace("'", "").replace(" ","")
        boxes = boxes.split(",")
        boxes = np.asarray(boxes)
        boxes = boxes.astype(np.float)
        boxes = torch.from_numpy(boxes.reshape(-1, 5))
        x1 = w_factor * (boxes[:, 1] - boxes[:, 3] / 2)
        y1 = h_factor * (boxes[:, 2] - boxes[:, 4] / 2)
        x2 = w_factor * (boxes[:, 1] + boxes[:, 3] / 2)
        y2 = h_factor * (boxes[:, 2] + boxes[:, 4] / 2)
        # Adjust for added padding
        x1 += pad[0]
        y1 += pad[2]
        x2 += pad[1]
        y2 += pad[3]
        # Returns (x, y, w, h)
        boxes[:, 1] = ((x1 + x2) / 2) / padded_w
        boxes[:, 2] = ((y1 + y2) / 2) / padded_h
        boxes[:, 3] *= w_factor / padded_w
        boxes[:, 4] *= h_factor / padded_h

        targets = torch.zeros((len(boxes), 6))
        targets[:, 1:] = boxes
        return image, targets

    def collate_fn(self, batch):
        imgs, targets = list(zip(*batch))
        # Remove empty placeholder targets
        targets = [boxes for boxes in targets if boxes is not None]
        # Add sample index to targets
        for i, boxes in enumerate(targets):
            boxes[:, 0] = i
        targets = torch.cat(targets, 0)
        # Resize images to input shape
        imgs = torch.stack([resize(img, self.img_size) for img in imgs])
        self.batch_count += 1
        return imgs, targets

def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size).squeeze(0)
    return image


def pad_to_square(img, pad_value):
    c, h, w = img.shape
    dim_diff = np.abs(h - w)
    # (upper / left) padding and (lower / right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
    # Add padding
    img = F.pad(img, pad, "constant", value=pad_value)

    return img, pad


class SSDDataset(Dataset):
    def __init__(self, csv_file, img_size=300, normalized_labels=True, amplyfied=False):
        self.img_data = pd.read_csv(csv_file, header=None)
        self.img_size = img_size
        self.normalized_labels = normalized_labels
        self.batch_count = 0
        self.amplyfied = amplyfied

    def __len__(self):
        return len(self.img_data)

    def __getitem__(self, item):
        if torch.is_tensor(item):
            item = item.tolist()

        image = pyd.dcmread(self.img_data.iloc[item][0]).pixel_array
        boxes = self.prep_target(item)
        if self.amplyfied:
            image, boxes = self.amplyfy(image, boxes)
            image = image[::4, ::4]
        else:
            image = image[::8, ::8]
        image = image.astype(np.float32)
        image = torch.from_numpy(image)
        if len(image.shape) != 3:
            image = image.unsqueeze(0)
            h = image.shape[1]
            w = image.shape[2]
            image = image.expand((3, h, w))

        temp, h, w = image.shape
        h_factor, w_factor = (h, w) if self.normalized_labels else (1, 1)
        # Pad to square resolution
        img, pad = pad_to_square(image, 0)
        temp, padded_h, padded_w = img.shape
        image, _ = pad_to_square(image, 0)
        classes = boxes[:, 0] * 1
        x1 = w_factor * (boxes[:, 1] - boxes[:, 3] / 2)
        y1 = h_factor * (boxes[:, 2] - boxes[:, 4] / 2)
        x2 = w_factor * (boxes[:, 1] + boxes[:, 3] / 2)
        y2 = h_factor * (boxes[:, 2] + boxes[:, 4] / 2)
        # Adjust for added padding
        x1 += pad[0]
        y1 += pad[2]
        x2 += pad[0]
        y2 += pad[2]
        # Returns (x, y, w, h)
        boxes[:, 0] = ((x1 + x2) / 2) / padded_w
        boxes[:, 1] = ((y1 + y2) / 2) / padded_h
        boxes[:, 2] = (boxes[:, 3]*w_factor) / padded_w
        boxes[:, 3] = (boxes[:, 4]*h_factor) / padded_h
        boxes[:, 4] = classes
        targets = torch.zeros((1, len(boxes), 5))
        targets[0] = boxes
        return image, targets

    def collate_fn(self, batch):
        imgs, targets = list(zip(*batch))
        # Remove empty placeholder targets
        targets = [boxes for boxes in targets if boxes is not None]
        # Add sample index to targets
        targets = torch.cat(targets, 0)
        # Resize images to input shape
        imgs = torch.stack([resize(img, self.img_size) for img in imgs])
        self.batch_count += 1
        return imgs, targets

    def amplyfy(self, img, boxes):
        width = img.shape[0]
        height = img.shape[1]
        x = boxes[:, 1]
        y = boxes[:, 2]
        x_abs = x * width
        y_abs = y * height
        w_abs = boxes[:, 3] * width
        h_abs = boxes[:, 4] * height

        if 0.6 > x > 0.4:
            cut_x = int(width / 3)
            cut_y = int(height / 3)
            if y > 0.6:
                img = img[cut_x:cut_x * 2:1, cut_y * 2::1]
                boxes[:, 1] = (x_abs - cut_x) / cut_x
                boxes[:, 2] = (y_abs - cut_y * 2) / cut_y
                boxes[:, 3] = (w_abs - cut_x) / cut_x
                boxes[:, 4] = (h_abs - cut_y * 2) / cut_y
            elif y < 0.4:
                img = img[cut_x:cut_x * 2:1, :cut_y:1]
                boxes[:, 1] = (x_abs - cut_x) / cut_x
                boxes[:, 2] = (y_abs - 0) / cut_y
                boxes[:, 3] = (w_abs - cut_x) / cut_x
                boxes[:, 4] = (h_abs - 0) / cut_y
            else:
                img = img[cut_x:cut_x * 2:1, cut_y:cut_y * 2:1]
                boxes[:, 1] = (x_abs - cut_x) / cut_x
                boxes[:, 2] = (y_abs - cut_y) / cut_y
                boxes[:, 3] = (w_abs - cut_x) / cut_x
                boxes[:, 4] = (h_abs - cut_y) / cut_y
        elif 0.6 > y > 0.4:
            cut_x = int(width / 3)
            cut_y = int(height / 3)
            if x > 0.6:
                img = img[cut_x * 2::1, cut_y:cut_y * 2:1]
                boxes[:, 1] = (x_abs - cut_x * 2) / cut_x
                boxes[:, 2] = (y_abs - cut_y) / cut_y
                boxes[:, 3] = (w_abs - cut_x * 2) / cut_x
                boxes[:, 4] = (h_abs - cut_y) / cut_y
            elif x < 0.4:
                img = img[:cut_x:1, cut_y:cut_y * 2:1]
                boxes[:, 1] = (x_abs - 0) / cut_x
                boxes[:, 2] = (y_abs - cut_y) / cut_y
                boxes[:, 3] = (w_abs - 0) / cut_x
                boxes[:, 4] = (h_abs - cut_y) / cut_y
            else:
                img = img[cut_x:cut_x * 2:1, cut_y:cut_y * 2:1]
                boxes[:, 1] = (x_abs - cut_x) / cut_x
                boxes[:, 2] = (y_abs - cut_y) / cut_y
                boxes[:, 3] = (w_abs - cut_x) / cut_x
                boxes[:, 4] = (h_abs - cut_y) / cut_y
        elif x > 0.5:
            cut_x = int(width / 2)
            cut_y = int(height / 2)
            if y > 0.5:
                img = img[cut_x::1, cut_y::1]
                boxes[:, 1] = (x_abs - cut_x) / cut_x
                boxes[:, 2] = (y_abs - cut_y) / cut_y
                boxes[:, 3] = (w_abs - cut_x) / cut_x
                boxes[:, 4] = (h_abs - cut_y) / cut_y
            else:
                img = img[cut_x::1, :cut_y:1]
                boxes[:, 1] = (x_abs - cut_x) / cut_x
                boxes[:, 2] = (y_abs - 0) / cut_y
                boxes[:, 3] = (w_abs - cut_x) / cut_x
                boxes[:, 4] = (h_abs - 0) / cut_y
        else:
            cut_x = int(width / 2)
            cut_y = int(height / 2)
            if y > 0.5:
                img = img[:cut_x:1, cut_y::1]
                boxes[:, 1] = (x_abs - 0) / cut_x
                boxes[:, 2] = (y_abs - cut_y) / cut_y
                boxes[:, 3] = (w_abs - 0) / cut_x
                boxes[:, 4] = (h_abs - cut_y) / cut_y
            else:
                img = img[:cut_x:1, :cut_y:1]
            boxes[:, 1] = (x_abs - 0) / cut_x
            boxes[:, 2] = (y_abs - 0) / cut_y
            boxes[:, 3] = (w_abs - 0) / cut_x
            boxes[:, 4] = (h_abs - 0) / cut_y

        return img, boxes

    def prep_target(self, item):
        boxes = self.img_data.iloc[item][1]
        boxes = boxes.replace("[", "").replace("]", "").replace("'", "").replace(" ", "")
        boxes = boxes.split(",")
        boxes = np.asarray(boxes)
        boxes = boxes.astype(np.float)
        boxes = torch.from_numpy(boxes.reshape(-1, 5))
        return boxes
