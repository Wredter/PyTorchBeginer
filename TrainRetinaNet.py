import torch
import matplotlib.pyplot as ptl
from torch.autograd import Variable
from Models.RetinaNet.RetinaNet import RetinaNet
from Models.RetinaNet.Utility import nms_prep, retinabox300
from Models.SSD.Utility import compare_trgets_with_bbox
from Models.Utility.DataSets import SSDDataset
from Models.Utility.Utility import prep_paths, list_avg
from Models.RetinaNet.Loss import RLoss


if __name__ == "__main__":
    train, test, dummy_test, class_names = prep_paths()
    torch.autograd.set_detect_anomaly(True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoding = []
    loslist = []
    num_classes = 1
    epochs = 150
    img_size = 300
    batch_size = 4

    model = RetinaNet(num_classes).to(device)
    loss_func = RLoss(num_classes)
    model = model.train(False)

    # na przyszłość nie robić tak jak zrobiłem to głupie i działa tylko dla konkretnego przypadku
    dumy_ds = SSDDataset(csv_file=dummy_test, img_size=img_size)
    dummy_loader = torch.utils.data.DataLoader(
        dumy_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=1,
        pin_memory=True,
        collate_fn=dumy_ds.collate_fn,
    )

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.95)

    start_epoch = 0
    iteration = 0
    t_bbox = retinabox300()
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
        for batch_j in range(batch_size):
            nms_prep(imgs[batch_j], targets[batch_j], ploc[batch_j], plabel[batch_j], db)
    # Training
    optimizer.zero_grad()
    ds = SSDDataset(csv_file=train, img_size=img_size)
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
        print(f"--------------------- Epoch {epoch}/{epochs} ---------------------")
        epoch_err = []
        for batch_i, (imgs, targets) in enumerate(dataloader):
            optimizer.zero_grad()
            imgs = Variable(imgs.to(device))
            targets = Variable(targets.to(device), requires_grad=False)
            # locations of targets
            targets_loc = targets[..., :4]
            # classes 0 for first
            targets_c = targets[..., -1]
            ploc, plabel = model(imgs)
            for batch_j in range(ploc.shape[0]):
                mached_loc, mached_label, mask = compare_trgets_with_bbox(t_bbox(order="ltrb").to(device),
                                                                          targets_loc[batch_j],
                                                                          targets_c[batch_j],
                                                                          0.4)
                if batch_j == 0:
                    m_pos = mached_loc.unsqueeze(0)
                    m_cls = mached_label.unsqueeze(0)
                    m_iou = mask.unsqueeze(0)
                    pos_num1 = m_iou.long().sum().item()
                else:
                    m_pos = torch.cat((m_pos, mached_loc.unsqueeze(0)), dim=0)
                    m_cls = torch.cat((m_cls, mached_label.unsqueeze(0)), dim=0)
                    m_iou = torch.cat((m_iou, mask.unsqueeze(0)), dim=0)
                    pos_num1 = m_iou.long().sum().item()

            m_pos = Variable(m_pos.to(ploc.device), requires_grad=False)
            m_cls = Variable(m_cls.to(ploc.device, dtype=torch.float32), requires_grad=False)
            m_iou = Variable(m_iou.to(ploc.device), requires_grad=False)
            pos_num = m_iou.long().sum().item()
            if pos_num == 0:
                print("Jeeeeez popraw boxy")
                break
            loss = loss_func(ploc, plabel, m_pos, m_cls, m_iou)
            loss.backward()
            if batch_i == 1:
                for batch_j in range(batch_size):
                    nms_prep(imgs[batch_j], targets[batch_j], ploc[batch_j], plabel[batch_j], db, epoch=epoch)
            epoch_err.append(loss.item())
            # if batch_i % 2:
            optimizer.step()

        scheduler.step()

        loslist.append(list_avg(epoch_err))
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

        for batch_j in range(batch_size):
            nms_prep(imgs[batch_j], targets[batch_j], ploc[batch_j], plabel[batch_j], db)

    print("Skończyłem")

