# -*- coding: utf-8 -*-
# Written by yq_yao
# 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn.init as init
import os
from layers.yolo_layer import YoloLayer


class YoloLoss(nn.Module):
    def __init__(self, input_wh, num_classes, ignore_thresh, anchors, anchors_mask, use_gpu=True):
        super(YoloLoss, self).__init__()
        self.input_wh = input_wh
        self.num_classes = num_classes
        self.ignore_thresh = ignore_thresh
        self.use_gpu = use_gpu
        self.anchors = anchors
        self.anchors_mask = anchors_mask
        self.yolo_layer1 = YoloLayer(input_wh, num_classes, ignore_thresh, anchors, anchors_mask[0])
        self.yolo_layer2 = YoloLayer(input_wh, num_classes, ignore_thresh, anchors, anchors_mask[1])
        self.yolo_layer3 = YoloLayer(input_wh, num_classes, ignore_thresh, anchors, anchors_mask[2])      

    def forward(self, inputs, targets, input_wh, debug):
        self.input_wh = input_wh
        x, y, z = inputs
        batch_size = x.size(0)
        loc_loss1, conf_loss1, cls_loss1 = self.yolo_layer1(x, targets, self.input_wh, debug)
        loc_loss2, conf_loss2, cls_loss2 = self.yolo_layer2(y, targets, self.input_wh, debug)
        loc_loss3, conf_loss3, cls_loss3 = self.yolo_layer3(z, targets, self.input_wh, debug)
        loc_loss = loc_loss1 + loc_loss2 + loc_loss3
        conf_loss = conf_loss1 + conf_loss2 + conf_loss3
        cls_loss = cls_loss1 + cls_loss2 + cls_loss3
        loss = loc_loss + conf_loss + cls_loss
        return loss