from __future__ import division

import os
import matplotlib.pyplot as ptl
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
    dummy_test = os.getcwd()
    dummy_test += "\\Data\\Dumy_test.csv"
    x = os.getcwd()
    x += "/Models/YOLO/config/yolov3.cfg"
    # test = ResourceProvider(y, "D:\\DataSet\\CBIS-DDSM\\", "D:\\DataSet\\ROI\\CBIS-DDSM\\")
    class_names = ["patologia"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    evaluation_interval = 10
    num_classes = 2
    epochs = 100
    img_size = 300
    batch_size = 8
    loslist = []

    model = SSD300(num_classes).to(device)
    model = model.train(False)

    # na przyszłość nie robić tak jak zrobiłem to głupie i działa tylko dla konkretnego przypadku
    dumy_ds = SSDDataset(csv_file=dummy_test, img_size=img_size, amplyfied=True)
    dummy_loader = torch.utils.data.DataLoader(
        dumy_ds,
        batch_size=2,
        shuffle=True,
        num_workers=1,
        pin_memory=True,
        collate_fn=dumy_ds.collate_fn,
    )

    optimizer = torch.optim.Adam(model.parameters(), 0.001)
    model = model.train(True)
    start_epoch = 0
    iteration = 0
    t_bbox = dboxes300()
    loss_func = Loss(t_bbox, num_classes).to(device)
    # Test input
    for batch_i, (imgs, targets) in enumerate(dummy_loader):
        imgs = Variable(imgs.to(device))
        targets = Variable(targets.to(device), requires_grad=False)
        # locations of targets
        targets_loc = targets[..., :4]
        # classes 0 for first
        targets_c = targets[..., -1]
        ploc, plabel = model(imgs)
        plabel = plabel[:, :, 1]
        db = t_bbox(order="xywh").to(device)
        _, idx = plabel.max(1, keepdim=True)
        for x in range(2):
            print(f'Target: {targets_loc[x].tolist()} \n'
                  f'Predicted: {(ploc[x, idx[x]] + db[idx[x]]).tolist()} \n'
                  f'Decoded: {(decode(ploc[x, idx[x]], db[idx[x]])).tolist()}')
            temp = torch.cat((targets_loc[x], decode(ploc[x, idx[x]], db[idx[x]])), 0)
            temp = torch.cat((temp, ploc[x, idx[x]] + db[idx[x]]), 0)
            show_areas(imgs[x], temp, 0)
    # Training
    ds = SSDDataset(csv_file=train, img_size=img_size)

    dataloader = torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        collate_fn=ds.collate_fn,
    )
    for epoch in range(start_epoch, epochs):

        for batch_i, (imgs, targets) in enumerate(dataloader):
            imgs = Variable(imgs.to(device))
            targets = Variable(targets.to(device), requires_grad=False)
            # locations of targets
            targets_loc = targets[..., :4]
            # classes 0 for first
            targets_c = targets[..., -1]
            ploc, plabel = model(imgs)

            loc_t, conf_t = compare_prediction_with_bbox(t_bbox(order='ltrb').to(device),
                                                         targets_loc,
                                                         targets_c,
                                                         0.5,
                                                         t_bbox.variance)
            loc_t = Variable(loc_t.to(device), requires_grad=False)
            conf_t = Variable(conf_t.to(device), requires_grad=False)

            loss = loss_func(ploc, plabel, loc_t, conf_t)
            loss.backward()
            loslist.append(loss.item())
            if batch_i % 2:
                optimizer.step()
                optimizer.zero_grad()

        if epoch % 5 == 0:
            print("Epoch: " + str(epoch) + " Total loss : " + str(loslist[-1])
                  )
    ptl.plot(loslist)
    ptl.ylabel("loss")

    ptl.show()

    for batch_i, (imgs, targets) in enumerate(dummy_loader):
        imgs = Variable(imgs.to(device))
        targets = Variable(targets.to(device), requires_grad=False)
        # locations of targets
        targets_loc = targets[..., :4]
        # classes 0 for first
        targets_c = targets[..., -1]
        ploc, plabel = model(imgs)
        plabel = plabel[:, :, 1]
        _, idx = plabel.max(1, keepdim=True)
        db = t_bbox(order="xywh").to(device)
        for x in range(2):
            print(f'Target: {targets_loc[x].tolist()} \n'
                  f'Predicted: {(ploc[x, idx[x]] + db[idx[x]]).tolist()} \n'
                  f'Decoded: {(decode(ploc[x, idx[x]], db[idx[x]])).tolist()}')
            temp = torch.cat((targets_loc[x], decode(ploc[x, idx[x]], db[idx[x]])), 0)
            temp = torch.cat((temp, ploc[x, idx[x]] + db[idx[x]]), 0)
            show_areas(imgs[x], temp, 0)

