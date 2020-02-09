from __future__ import division

import os

import torch
import tqdm
from torch.autograd import Variable
from torch.utils.data import DataLoader
from Models.Utility.DataSets import ImgDataset
from Models.RetinaNet.RetinaNet import *
from Models.RetinaNet.Loss import *


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
    batch_size = 1
    img_size = 800
    model = RetinaNet().to(device)
    ds = ImgDataset(csv_file=train, img_size=img_size)

    dataloader = torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=1,
        pin_memory=True,
        collate_fn=ds.collate_fn,
    )

    optimizer = torch.optim.Adam(model.parameters())
    dataloader = torch.utils.data.DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=1,
            pin_memory=True,
            collate_fn=ds.collate_fn,
        )
    model.train()
    criterion = FocalLoss()
    for epoch in range(epochs):
        total_loss = 0
        for batch_i, (imgs, targets) in enumerate(dataloader):

            imgs = Variable(imgs.to(device))
            targets = Variable(targets.to(device), requires_grad=False)
            loc_targets = targets[:, 2:]
            cls_targets = targets[:, :1]

            optimizer.zero_grad()

            pos, cls = model(imgs)
            loc_loss, cls_loss = criterion(pos, loc_targets, cls, cls_targets)
            loss = loc_loss + cls_loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print("Epoch: " + str(epoch) + " Total loss: " + str(total_loss))



