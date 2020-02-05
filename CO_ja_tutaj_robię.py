from __future__ import division

import os

import torch
import tqdm
from torch.autograd import Variable
from torch.utils.data import DataLoader
from Models.Utility.DataSets import ImgDataset
from Models.SSD.SSD import *

if __name__ == "__main__":
    train = os.getcwd()
    train += "\\Data\\preped_data_mass_train.csv"
    test = os.getcwd()
    test += "\\Data\\preped_data_mass_test.csv"
    x = os.getcwd()
    x += "/Models/YOLO/config/yolov3.cfg"
    # test = ResourceProvider(y, "D:\\DataSet\\CBIS-DDSM\\", "D:\\DataSet\\ROI\\CBIS-DDSM\\")
    class_names = ["patologia"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    evaluation_interval = 10
    epochs = 100
    img_size = 300
    model = SSD300().to(device)
    ds = ImgDataset(csv_file=train, img_size=img_size)

    dataloader = torch.utils.data.DataLoader(
        ds,
        batch_size=8,
        shuffle=True,
        num_workers=1,
        pin_memory=True,
        collate_fn=ds.collate_fn,
    )

    optimizer = torch.optim.Adam(model.parameters())
    dataloader = torch.utils.data.DataLoader(
            ds,
            batch_size=8,
            shuffle=True,
            num_workers=1,
            pin_memory=True,
            collate_fn=ds.collate_fn,
        )

    for batch_i, (imgs, targets) in enumerate(dataloader):

        imgs = Variable(imgs.to(device))
        targets = Variable(targets.to(device), requires_grad=False)

        pos, cls = model(imgs)
        print(pos)
        print(cls)
