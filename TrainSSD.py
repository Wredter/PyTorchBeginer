from __future__ import division

import os
from torch.utils.data import DataLoader
from Models.SSD.DefaultsBox import dboxes300
from Models.SSD.Utility import *
from Models.Utility.Utility import list_avg
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
    epochs = 300
    img_size = 300
    batch_size = 8
    loslist = []

    model = SSD300(num_classes).to(device)
    model = model.train(False)

    # na przyszłość nie robić tak jak zrobiłem to głupie i działa tylko dla konkretnego przypadku
    dumy_ds = SSDDataset(csv_file=dummy_test, img_size=img_size, mod="dac")
    dummy_loader = torch.utils.data.DataLoader(
        dumy_ds,
        batch_size=2,
        shuffle=True,
        num_workers=1,
        pin_memory=True,
        collate_fn=dumy_ds.collate_fn,
    )

    optimizer = torch.optim.Adam(model.parameters(), 0.0008)

    start_epoch = 0
    iteration = 0
    t_bbox = dboxes300()
    loss_func = Loss(t_bbox, 0.5, num_classes).to(device)
    # Test input
    for batch_i, (imgs, targets) in enumerate(dummy_loader):
        imgs = Variable(imgs.to(device))
        targets = Variable(targets.to(device), requires_grad=False)
        # locations of targets
        targets_loc = targets[..., :4]
        # classes 0 for first
        targets_c = targets[..., -1]
        ploc, plabel = model(imgs)
        db = t_bbox(order="xywh").to(device)
        final = final_detection(ploc, plabel, db)
        raw = final_detection(ploc, plabel, db, encoding=None)
        delta = final_detection(ploc, plabel, db, encoding="delta")
        decoded_delta = final_detection(ploc, plabel, db, encoding="d_delta")
        for x in range(dummy_loader.batch_size):
            print(f'Target: {targets[x].tolist()} \n'
                  f'Decoded: {final[x].tolist()} \n'
                  f'Raw: {raw[x].tolist()} \n'
                  f'Delta {delta[x].tolist()} \n'
                  f'Decoded Delta {decoded_delta[x].tolist()}')
            show_areas(imgs[x], targets_loc[x], final[x][:, :4], 0, plot_title="Decoded")
            show_areas(imgs[x], targets_loc[x], raw[x][:, :4], 0, plot_title="Raw")
            show_areas(imgs[x], targets_loc[x], delta[x][:, :4], 0, plot_title="delta")
            show_areas(imgs[x], targets_loc[x], decoded_delta[x][:, :4], 0, plot_title="Decoded Delta")
    # Training
    ds = SSDDataset(csv_file=dummy_test, img_size=img_size, mod="dac")
    model = model.train(True)
    dataloader = torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        collate_fn=ds.collate_fn,
    )
    for epoch in range(start_epoch, epochs):
        epoch_err = []
        for batch_i, (imgs, targets) in enumerate(dataloader):
            imgs = Variable(imgs.to(device))
            targets = Variable(targets.to(device), requires_grad=False)
            # locations of targets
            targets_loc = targets[..., :4]
            # classes 0 for first
            targets_c = targets[..., -1]
            ploc, plabel = model(imgs)

            loss = loss_func(ploc, plabel, targets_loc, targets_c)

            if loss != 0:
                epoch_err.append(loss.item())
                loss.backward()
            else:
                print("Skipped loss")
            #if batch_i % 2:
            optimizer.step()
            optimizer.zero_grad()
        loslist.append(list_avg(epoch_err))
        if epoch % 5 == 0:
            print("Epoch: " + str(epoch) + " Total loss : " + str(loslist[-1])
                  )
            final = final_detection(ploc, plabel, db)
            raw = final_detection(ploc, plabel, db, encoding=None)
            delta = final_detection(ploc, plabel, db, encoding="delta")
            decoded_delta = final_detection(ploc, plabel, db, encoding="d_delta")
#            for x in range(1):
#                print(f'Target: {targets[x].tolist()} \n'
#                      f'Decoded: {final[x].tolist()} \n'
#                      f'Raw: {raw[x].tolist()} \n'
#                      f'Delta {delta[x].tolist()} \n'
#                      f'Decoded Delta {decoded_delta[x].tolist()}')
#                show_areas(imgs[x], targets_loc[x], final[x][:, :4], 0, plot_title="Decoded")
#                show_areas(imgs[x], targets_loc[x], raw[x][:, :4], 0, plot_title="Raw")
#                show_areas(imgs[x], targets_loc[x], delta[x][:, :4], 0, plot_title="delta")
#                show_areas(imgs[x], targets_loc[x], decoded_delta[x][:, :4], 0, plot_title="Decoded Delta")
    ptl.plot(loslist)
    ptl.ylabel("loss")

#    ptl.show()

    for batch_i, (imgs, targets) in enumerate(dummy_loader):
        imgs = Variable(imgs.to(device))
        targets = Variable(targets.to(device), requires_grad=False)
        # locations of targets
        targets_loc = targets[..., :4]
        # classes 0 for first
        targets_c = targets[..., -1]
        ploc, plabel = model(imgs)
        _, idx = plabel.max(1, keepdim=True)
        final = final_detection(ploc, plabel, db)
        raw = final_detection(ploc, plabel, db, encoding=None)
        delta = final_detection(ploc, plabel, db, encoding="delta")
        decoded_delta = final_detection(ploc, plabel, db, encoding="d_delta")
        for x in range(dummy_loader.batch_size):
            print(f'Target: {targets[x].tolist()} \n'
                  f'Decoded: {final[x].tolist()} \n'
                  f'Raw: {raw[x].tolist()} \n'
                  f'Delta {delta[x].tolist()} \n'
                  f'Decoded Delta {decoded_delta[x].tolist()}')
            show_areas(imgs[x], targets_loc[x], final[x][:, :4], 0, plot_title="Decoded")
            show_areas(imgs[x], targets_loc[x], raw[x][:, :4], 0, plot_title="Raw")
            show_areas(imgs[x], targets_loc[x], delta[x][:, :4], 0, plot_title="delta")
            show_areas(imgs[x], targets_loc[x], decoded_delta[x][:, :4], 0, plot_title="Decoded Delta")
        print("Skończyłem")

