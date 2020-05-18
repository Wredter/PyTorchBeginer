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
from Models.SSD.Utility import compare_prediction_with_bbox


class ResNet(nn.Module):
    def __init__(self, backbone='resnet50', backbone_path=None):
        super(ResNet, self).__init__()
        if backbone == 'resnet18':
            backbone = resnet18(pretrained=not backbone_path)
            self.out_channels = [256, 512, 512, 256, 256, 128]
        elif backbone == 'resnet34':
            backbone = resnet34(pretrained=not backbone_path)
            self.out_channels = [256, 512, 512, 256, 256, 256]
        elif backbone == 'resnet50':
            backbone = resnet50(pretrained=not backbone_path)
            self.out_channels = [1024, 512, 512, 256, 256, 256]
        elif backbone == 'resnet101':
            backbone = resnet101(pretrained=not backbone_path)
            self.out_channels = [1024, 512, 512, 256, 256, 256]
        else:  # backbone == 'resnet152':
            backbone = resnet152(pretrained=not backbone_path)
            self.out_channels = [1024, 512, 512, 256, 256, 256]
        if backbone_path:
            backbone.load_state_dict(torch.load(backbone_path))

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
        self.label_num = num_classes# 1 class + background
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

    def forward(self, ploc, plabel, gloc, glabel):
        gloc, glabel = compare_prediction_with_bbox(self.dboxes(order='ltrb').to(ploc.device),
                                                     gloc,
                                                     glabel,
                                                     0.4)
        gloc = Variable(gloc.to(ploc.device), requires_grad=False)
        glabel = Variable(glabel.to(ploc.device), requires_grad=False)


        num = ploc.size(0)
        #torch.set_printoptions(profile='full')

        mask = glabel.view(-1) > 0
        pos_num = mask.sum(dim=0)

        # location loss
        loc_p = ploc.view(-1, 4)
        loc_t = gloc.view(-1, 4)
        loss_l = self.sl1_loss(loc_p, loc_t).sum(1)
        loss_l = (mask.float()*loss_l).sum()
        #torch.set_printoptions(profile='default')
        batch_conf = plabel.view(-1, self.num_classes)
        batch_gt = glabel.view(-1)

        loss_c = self.con_loss(batch_conf, batch_gt)

        # postive mask will never selected
        con_neg = loss_c.clone()
        con_neg[mask] = 0
        _, con_idx = con_neg.sort(dim=0, descending=True)
        _, con_rank = con_idx.sort(dim=0)

        # number of negative three times positive
        neg_num = torch.clamp(3 * pos_num, max=mask.size(0)).unsqueeze(-1)
        neg_mask = con_rank < neg_num

        # print(con.shape, mask.shape, neg_mask.shape)
        closs = (loss_c * (mask.float() + neg_mask.float())).sum(dim=0)

        # avoid no object detected
        total_loss = loss_l + closs
        if pos_num.item() != 0:
            ret = (total_loss / pos_num)
            return ret
        else:
            return 0
