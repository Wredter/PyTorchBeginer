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

    start_epoch = 0
    iteration = 0

    for epoch in range(start_epoch, epochs):
        start_epoch_time = time.time()
        scheduler.step()
        iteration = train_loop_func(ssd300, loss_func, epoch, optimizer, train_loader, val_dataloader, encoder,
                                    iteration,
                                    logger, args, mean, std)
        end_epoch_time = time.time() - start_epoch_time
        total_time += end_epoch_time

        if args.local_rank == 0:
            logger.update_epoch_time(epoch, end_epoch_time)

        if epoch in args.evaluation:
            acc = evaluate(ssd300, val_dataloader, cocoGt, encoder, inv_map, args)

            if args.local_rank == 0:
                logger.update_epoch(epoch, acc)

        if args.save and args.local_rank == 0:
            print("saving model...")
            obj = {'epoch': epoch + 1,
                   'iteration': iteration,
                   'optimizer': optimizer.state_dict(),
                   'scheduler': scheduler.state_dict(),
                   'label_map': val_dataset.label_info}
            if args.distributed:
                obj['model'] = ssd300.module.state_dict()
            else:
                obj['model'] = ssd300.state_dict()
            torch.save(obj, './models/epoch_{}.pt'.format(epoch))
        train_loader.reset()
    print('total training time: {}'.format(total_time))