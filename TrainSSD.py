from __future__ import division

import os

import torch
import tqdm
from torch.autograd import Variable
from torch.utils.data import DataLoader
from Models.Utility.DataSets import SSDDataset
from Models.SSD.SSD import *
import torch.optim as optim
import logging as log


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
    learning_rate = 0.001
    momentum = 0.9
    weight_decay = 0.0005
    model = build_ssd(img_size)
    model.to(device)
    ds = SSDDataset(csv_file=train, img_size=img_size)
    dataloader = torch.utils.data.DataLoader(
        ds,
        batch_size=8,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        collate_fn=ds.collate_fn,
    )

    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum,
                          weight_decay=weight_decay)
    criterion = MultiBoxLoss(cfg['num_classes'], 0.5, True, 0, True, 3, 0.5,
                             False, torch.cuda.is_available())

    model.train()
    iteration = 0
    loc_loss = 0
    conf_loss = 0
    epoch = 0

    for epoch in range(0, epochs):
        for batch_i, (imgs, targets) in enumerate(dataloader):
            imgs = Variable(imgs.to(device))
            targets = Variable(targets.to(device), requires_grad=False)

            loc_loss = 0
            conf_loss = 0
            epoch += 1
            out = model(imgs)

            optimizer.zero_grad()

            l_loss, c_loss = criterion(out, targets)
            loss = l_loss + c_loss
            loss.backward()
            optimizer.step()
            loc_loss += l_loss.data[0]
            conf_loss += c_loss.data[0]
            log.info("batch: f'{batch_i} loss: f'{loss}")
    z = os.getcwd()
    z += "\\Models\\SSD\\TrainedModel\\SSD.pth"
    torch.save(model.state_dict(), z)
