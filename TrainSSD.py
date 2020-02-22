from __future__ import division

import os

import torch
import tqdm
import logging
from torch.autograd import Variable
from torch.utils.data import DataLoader
from Models.SSD.DefaultsBox import dboxes300
from Models.SSD.Utility import *
from Models.Utility.DataSets import SSDDataset
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
    batch_size = 8

    model = SSD300().to(device)

    # na przyszłość nie robić tak jak zrobiłem to głupie i działa tylko dla konkretnego przypadku
    ds = SSDDataset(csv_file=train, img_size=img_size)

    dataloader = torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=1,
        pin_memory=True,
        collate_fn=ds.collate_fn,
    )

    optimizer = torch.optim.Adam(model.parameters())

    start_epoch = 0
    iteration = 0
    t_bbox = dboxes300()
    loss_func = Loss(t_bbox)
    for epoch in range(start_epoch, epochs):
        loss = 0
        for batch_i, (imgs, targets) in enumerate(dataloader):

            imgs = Variable(imgs.to(device))
            targets = Variable(targets.to(device), requires_grad=False)
            # locations of targets
            targets_loc = targets[..., :4]
            # classes 0 for first
            targets_c = targets[:, :, -1]
            ploc, plabel = model(imgs)

            loc_t, conf_t = compare_prediction_with_bbox(ploc, t_bbox(order="ltrb").to(device), targets_loc, targets_c, 0.5)

            loc_t = Variable(loc_t, requires_grad=False)
            conf_t = Variable(conf_t, requires_grad=False)

            transpose_ploc = ploc.transpose(1, 2).contiguous()
            transpose_loc_t = loc_t.transpose(1, 2).contiguous()

            loss = loss_func(ploc, plabel, loc_t, conf_t)
            logging.info("Epoch: f'{epoch} Total loss : f'{loss}")
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()


