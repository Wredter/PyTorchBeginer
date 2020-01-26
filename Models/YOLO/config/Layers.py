import torch.nn as nn
import torch
import numpy as np


class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()

class DetectionLayer(nn.Module):
    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors


"""class YOLOLayer(nn.Module):
    # OLD detection layer
    def __init__(self, anchors, num_classes, img_dim=416):
        super(YOLOLayer, self).__init__()
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.ignore_thres = 0.5
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        self.obj_scale = 100
        self.metrics = {}
        self.img_dim = img_dim
        self.grid_size = 0
        self.stride = 0

    def compute_grid_offsets(self, grid_size, CUDA=True):
        self.grid_size = grid_size
        temp_grid_size = self.grid_size

        FloatTensor = torch.FloatTensor
        grid = np.arange(grid_size)
        a, b = np.meshgrid(grid, grid)

        self.x_offset = torch.FloatTensor(a).view(-1, 1)
        self.y_offset = torch.FloatTensor(b).view(-1, 1)

        if CUDA:
            self.x_offset = self.x_offset.cuda()
            self.y_offset = self.y_offset
            FloatTensor = FloatTensor.cuda()

        self.scaled_anchors = FloatTensor([(a_w / self.stride, a_h / self.stride) for a_w, a_h in self.anchors])
        self.anchor_w = self.scaled_anchors[:, 0, 1].view((1, self.num_anchors, 1, 1))
        self.anchor_h = self.scaled_anchors[:, 1, 2].view((1, self.num_anchors, 1, 1))

    def forward(self, x, targets=None, img_dim=None):
        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor
        ByteTensor = torch.cuda.ByteTensor if x.is_cuda else torch.ByteTensor

        self.img_dim = img_dim
        num_samples = x.size(0)
        grid_size = x.size(2)

        prediction = (
            x.view(num_samples, self.num_anchors, self.num_classes + 5, grid_size, grid_size)
            .permutate(0, 1, 3, 4, 2)
            .contiguous()
        )

        #sigma na outpucie sieci

        x = torch.sigmoid(prediction[:, :, :, :, 0])
        y = torch.sigmoid(prediction[:, :, :, :, 1])
        width = torch.sigmoid(prediction[:, :, :, :, 2])
        height = torch.sigmoid(prediction[:, :, :, :, 3])
        pred_conf = torch.sigmoid(prediction[:, :, :, :, 4])
        pred_classes = torch.sigmoid(prediction[:, :, :, :, 5:])

        if grid_size != self.grid_size:
            self.compute_grid_offsets(grid_size, x.is_cuda)

        pred_boxes = FloatTensor(prediction[..., :4].shape)
        pred_boxes[..., 0] = x.data + self.x_offset
        pred_boxes[..., 1] = y.data + self.y_offset
        pred_boxes[..., 2] = torch.exp(width.data) * self.anchor_w
        pred_boxes[..., 3] = torch.exp(height.data) * self.anchor_h

        output = torch.cat(
            (
                pred_boxes.view(num_samples, -1, 4) * self.stride,
                pred_conf.view(num_samples, -1, 1),
                pred_classes(num_samples, -1, self.num_classes),
            ),
            -1,
        )

        if targets is None:
            return output, 0
        else:
            iou_scores, class_mask, obj_mask, tx, ty, tw, th, tcls, tconf = build_targets(
                pred_boxes=pred_boxes,
                pred_classes=pred_classes,
                target=targets,
                anchors=self.scaled_anchors,
                ignore_thres=self.ignore_thres,
            )


def build_targets(pred_boxes, pred_classes, target, anchors, ignore_thres):
    FloatTensor = torch.cuda.FloatTensor if pred_boxes.is_cuda else torch.FloatTensor
    ByteTensor = torch.cuda.ByteTensor if pred_boxes.is_cuda else torch.ByteTensor

    nB = pred_boxes.size(0)
    nA = pred_boxes.size(1)
    nC = pred_classes.size(-1)
    nG = pred_boxes.size(2)

    # Output tensors

    obj_mask = ByteTensor(nB, nA, nG, nG).fill_(0)
    noobj_mask = ByteTensor(nB, nA, nG, nG).fill_(1)
    class_mask = FloatTensor(nB, nA, nG, nG).fill_(0)
    iou_scores = FloatTensor(nB, nA, nG, nG).fill_(0)
    tx = FloatTensor(nB, nA, nG, nG).fill_(0)
    ty = FloatTensor(nB, nA, nG, nG).fill_(0)
    tw = FloatTensor(nB, nA, nG, nG).fill_(0)
    th = FloatTensor(nB, nA, nG, nG).fill_(0)
    tcls = FloatTensor(nB, nA, nG, nG, nC).fill_(0)

    # Convert to position relativa to box
    target_boxes = target[:, 2:6] * nG
    gxy = target_boxes[:, :2]
    gwh = target_boxes[:, 2:]
    # anchors
    ious = torch.stack([bbox_wh_iou(anchor, gwh) for anchor in anchors])
    return 0
    """

