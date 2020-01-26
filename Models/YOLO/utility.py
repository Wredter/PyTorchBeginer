import torch
import numpy as np


def predict_transform(prediction, inp_dim, anchors, num_classes, CUDA = True):
    batch_size = prediction.size(0)
    stride = inp_dim // prediction.size(2)
    grid_size = inp_dim // stride
    bbox_attrs = 5 + num_classes
    num_anchors = len(anchors)

    prediction = prediction.view(batch_size, bbox_attrs * num_anchors, grid_size * grid_size)
    prediction = prediction.transpose(1, 2).contiguous()
    prediction = prediction.view(batch_size, grid_size * grid_size * num_anchors, bbox_attrs)
    anchors = [(a[0] / stride, a[1] / stride) for a in anchors]

    # Sigmoid the  centre_X, centre_Y. and object confidencce
    prediction[:, :, 0] = torch.sigmoid(prediction[:, :, 0])
    prediction[:, :, 1] = torch.sigmoid(prediction[:, :, 1])
    prediction[:, :, 4] = torch.sigmoid(prediction[:, :, 4])

    # Add the center offsets
    grid = np.arange(grid_size)
    a, b = np.meshgrid(grid, grid)

    x_offset = torch.FloatTensor(a).view(-1, 1)
    y_offset = torch.FloatTensor(b).view(-1, 1)

    if CUDA:
        x_offset = x_offset.cuda()
        y_offset = y_offset.cuda()

    x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1, num_anchors).view(-1, 2).unsqueeze(0)

    prediction[:, :, :2] += x_y_offset

    # log space transform height and the width
    anchors = torch.FloatTensor(anchors)

    if CUDA:
        anchors = anchors.cuda()

    anchors = anchors.repeat(grid_size * grid_size, 1).unsqueeze(0)
    prediction[:, :, 2:4] = torch.exp(prediction[:, :, 2:4]) * anchors

    prediction[:, :, 5: 5 + num_classes] = torch.sigmoid((prediction[:, :, 5: 5 + num_classes]))

    prediction[:, :, :4] *= stride

    return prediction


def write_results(prediction, confidence, num_classes, target, imp_dim, nms_conf=0.4,):
    conf_mask = (prediction[:, :, 4] > confidence).float().unsqueeze(2)
    # prediction = prediction[:, :, 4:] * conf_mask

    boxes = prediction.new(prediction.shape)
    boxes[:, :, 0] = (prediction[:, :, 0] - prediction[:, :, 2] / 2)
    boxes[:, :, 1] = (prediction[:, :, 1] - prediction[:, :, 3] / 2)
    boxes[:, :, 2] = (prediction[:, :, 0] + prediction[:, :, 2] / 2)
    boxes[:, :, 3] = (prediction[:, :, 1] + prediction[:, :, 3] / 2)
    boxes[..., 4:] = prediction[..., 4:]

    t_boxes = target.new(target.shape)
    t_boxes[:, 0] = (target[:, 0] - target[:, 2] / 2) * imp_dim
    t_boxes[:, 1] = (target[:, 1] - target[:, 3] / 2) * imp_dim
    t_boxes[:, 2] = (target[:, 0] + target[:, 2] / 2) * imp_dim
    t_boxes[:, 3] = (target[:, 1] + target[:, 3] / 2) * imp_dim
    t_boxes[:, 4:] = target[:, 4:]

    write = False

    samples = prediction.size(0)

    for sample in range(samples):
        img_predictions = prediction[sample]
    return 0
