# Copyright (c) 2018-2019, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from torch.autograd import Variable
import torch
import torch.nn as nn
from torchvision.models.resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from Models.SSD.Utility import compare_trgets_with_bbox


class ResNet(nn.Module):
    def __init__(self, backbone='resnet50'):
        super(ResNet, self).__init__()
        if backbone == 'resnet50':
            backbone = resnet50()
            self.out_channels = [1024, 512, 512, 256, 256, 256]
        elif backbone == 'resnet101':
            backbone = resnet101()
            self.out_channels = [1024, 512, 512, 256, 256, 256]

        self.feature_extractor = nn.Sequential(*list(backbone.children())[:7])

        conv4_block1 = self.feature_extractor[-1][0]

        conv4_block1.conv1.stride = (1, 1)
        conv4_block1.conv2.stride = (1, 1)
        conv4_block1.downsample[0].stride = (1, 1)

    def forward(self, x):
        x = self.feature_extractor(x)
        return x


class SSD300(nn.Module):
    def __init__(self, num_classes, backbone=ResNet('resnet50')):
        super(SSD300, self).__init__()
        self.feature_extractor = backbone
        self.label_num = num_classes# 1 class without bacground
        self._build_additional_features(self.feature_extractor.out_channels)
        self.num_defaults = [4, 6, 6, 6, 4, 4]
        self.loc = []
        self.conf = []

        for nd, oc in zip(self.num_defaults, self.feature_extractor.out_channels):
            self.loc.append(
                    nn.Conv2d(oc, nd * 4, kernel_size=3, padding=1))
            self.conf.append(
                    nn.Conv2d(oc, nd * self.label_num, kernel_size=3, padding=1))

        self.loc = nn.ModuleList(self.loc)
        self.conf = nn.ModuleList(self.conf)

    def _build_additional_features(self, input_size):
        self.additional_blocks = []
        for i, (input_size, output_size, channels) in enumerate(zip(input_size[:-1], input_size[1:], [256, 256, 128, 128, 128])):
            if i < 3:
                layer = nn.Sequential(
                    nn.Conv2d(input_size, channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(channels, output_size, kernel_size=3, padding=1, stride=2, bias=False),
                    nn.BatchNorm2d(output_size),
                    nn.ReLU(inplace=True),
                )
            else:
                layer = nn.Sequential(
                    nn.Conv2d(input_size, channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(channels, output_size, kernel_size=3, bias=False),
                    nn.BatchNorm2d(output_size),
                    nn.ReLU(inplace=True),
                )

            self.additional_blocks.append(layer)

        self.additional_blocks = nn.ModuleList(self.additional_blocks)

    # Shape the classifier to the view of bboxes
    def bbox_view(self, src, loc, conf):
        ret = []
        for s, l, c in zip(src, loc, conf):
            ret.append((l(s).view(s.size(0), -1, 4), c(s).view(s.size(0), -1, self.label_num)))

        locs, confs = list(zip(*ret))
        locs, confs = torch.cat(locs, 1).contiguous(), torch.cat(confs, 1).contiguous()
        return locs, confs

    def forward(self, x):
        x = self.feature_extractor(x)

        detection_feed = [x]
        for l in self.additional_blocks:
            x = l(x)
            detection_feed.append(x)

        # Feature Map 38x38x4, 19x19x6, 10x10x6, 5x5x6, 3x3x4, 1x1x4
        locs, confs = self.bbox_view(detection_feed, self.loc, self.conf)

        # For SSD 300, shall return nbatch x 8732 x {nlabels, nlocs} results
        return locs, confs


# TO DO
class Loss(nn.Module):
    """
        Implements the loss as the sum of the followings:
        1. Confidence Loss: All labels, with hard negative mining
        2. Localization Loss: Only on positive labels
        Suppose input dboxes has the shape 8732x4
    """
    def __init__(self, dboxes, match_threshold, num_classes):
        super(Loss, self).__init__()
        self.num_classes = num_classes
        self.sl1_loss = nn.SmoothL1Loss(reduction='none')
        self.dboxes = dboxes
        self.match_threshold = match_threshold
        self.con_loss = nn.CrossEntropyLoss(reduction='none')

    def forward(self, ploc, plabel, gtloc, gtlabel):
        m_pos = None
        m_cls = None
        for batch_i in range(ploc.shape[0]):
            mached_loc, mached_label, mask = compare_trgets_with_bbox(self.dboxes(order='ltrb').to(ploc.device),
                                                    gtloc[batch_i],
                                                    gtlabel[batch_i],
                                                    0.5)
            if batch_i == 0:
                m_pos = mached_loc
                m_cls = mached_label
                m_iou = mask
            else:
                m_pos = torch.cat((m_pos, mached_loc), dim=0)
                m_cls = torch.cat((m_cls, mached_label), dim=0)
                m_iou = torch.cat((m_iou, mask), dim=0)

        m_pos = Variable(m_pos.to(ploc.device), requires_grad=False)
        m_cls = Variable(m_cls.to(ploc.device, dtype=torch.long), requires_grad=False)
        m_iou = Variable(m_iou.to(ploc.device), requires_grad=False)

        num = m_pos.size(0)
        #torch.set_printoptions(profile='full')
        pos_num = m_iou.sum(dim=0)

        # location loss
        loc_p = ploc.view(-1, 4)
        loss_l = self.sl1_loss(loc_p, m_pos).sum(1).unsqueeze(1)
        loss_l = (m_iou.float()*loss_l).sum()
        #torch.set_printoptions(profile='default')

        batch_conf = plabel.view(-1, self.num_classes)

        loss_c = self.con_loss(batch_conf, m_cls.squeeze()).contiguous()

        # postive mask will never selected
        con_neg = loss_c.clone()
        m_neg_iou = ~m_iou
        con_neg = (m_neg_iou.squeeze().float() * con_neg)
        neg_num = pos_num * 3
        _, con_idx = con_neg.sort(dim=0, descending=True)
        con_neg = con_neg[con_idx[:neg_num]]

        # print(con.shape, mask.shape, neg_mask.shape)
        closs = (loss_c * (m_iou.squeeze().float())).sum()
        closs = closs + con_neg.sum()
        # avoid no object detected
        total_loss = loss_l + closs
        if pos_num.item() != 0:
            ret = (total_loss / pos_num)
            return ret
        else:
            return 0
