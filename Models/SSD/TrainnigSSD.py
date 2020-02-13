from __future__ import print_function, division

import torch
from terminaltables import AsciiTable
from torch.autograd import Variable
from torch.utils.data import DataLoader

from Models.Utility.DataSets import ImgDataset
from Models.Utility.ResourceProvider import *


def train_loop(model, loss_func, epoch, optim, train_dataloader, val_dataloader, encoder, iteration, logger, args, mean, std):
#     for nbatch, (img, _, img_size, bbox, label) in enumerate(train_dataloader):
    for  batch_i, (imgs, targets) in enumerate(train_dataloader):

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        imgs = Variable(imgs.to(device))
        targets = Variable(targets.to(device), requires_grad=False)

        N = img.shape[0]
        if bbox_offsets[-1].item() == 0:
            print("No labels in batch")
            continue
        bbox, label = C.box_encoder(N, bbox, bbox_offsets, label, encoder.dboxes.cuda(), 0.5)
        # output is ([N*8732, 4], [N*8732], need [N, 8732, 4], [N, 8732] respectively
        M = bbox.shape[0] // N
        bbox = bbox.view(N, M, 4)
        label = label.view(N, M)

        ploc, plabel = model(imgs)
        ploc, plabel = ploc.float(), plabel.float()

        trans_bbox = bbox.transpose(1, 2).contiguous().cuda()

        if not args.no_cuda:
            label = label.cuda()
        gloc = Variable(trans_bbox, requires_grad=False)
        glabel = Variable(label, requires_grad=False)

        loss = loss_func(ploc, plabel, gloc, glabel)

        if args.local_rank == 0:
            logger.update_iter(epoch, iteration, loss.item())

        if args.amp:
            with amp.scale_loss(loss, optim) as scale_loss:
                scale_loss.backward()
        else:
            loss.backward()

        if args.warmup is not None:
            warmup(optim, args.warmup, iteration, args.learning_rate)

        optim.step()
        optim.zero_grad()
        iteration += 1

    return iteration