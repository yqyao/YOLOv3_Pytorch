# -*- coding: utf-8 -*-
# Written by yq_yao
# 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import math
from .weight_mseloss import WeightMseLoss
from utils.box_utils import targets_match_single, permute_sigmoid, decode

class YoloLayer(nn.Module):

    def __init__(self, input_wh, num_classes, ignore_thresh, anchors, anchors_mask,use_gpu=True):
        super(YoloLayer, self).__init__()
        self.num_classes = num_classes
        self.ignore_thresh = ignore_thresh
        self.use_gpu = use_gpu
        self.anchors = anchors
        self.anchors_mask = anchors_mask
        self.input_wh = input_wh
        self.mse_loss = nn.MSELoss(size_average=False)
        self.bce_loss = nn.BCELoss(size_average=False)
        self.weight_mseloss = WeightMseLoss(size_average=False)

    def forward(self, x, targets, input_wh, debug=False):
        self.input_wh = input_wh
        batch_size = x.size(0)
        # feature map size w, h, this produce wxh cells to predict
        grid_wh = (x.size(3), x.size(2))
        x, stride = permute_sigmoid(x, input_wh, 3, self.num_classes)
        pred = x
        num_pred = pred.size(1)

        decode_pred = decode(pred.new_tensor(pred).detach(), self.input_wh, self.anchors[self.anchors_mask[0]: self.anchors_mask[-1]+1], self.num_classes, stride)

        # prediction targets x,y,w,h,objectness, class
        pred_t = torch.Tensor(batch_size, num_pred, 6).cuda()
        # xywh scale, scale = 2 - truth.w * truth.h (if truth is normlized to 1)
        scale_t = torch.FloatTensor(batch_size, num_pred).cuda()
        # foreground targets mask
        fore_mask_t = torch.ByteTensor(batch_size, num_pred).cuda()

        # background targets mask, we only calculate the objectness pred loss 
        back_mask_t = torch.ByteTensor(batch_size, num_pred).cuda()

        for idx in range(batch_size):
            # match our targets
            targets_match_single(self.input_wh, self.ignore_thresh, targets[idx], decode_pred[idx][:, :4], self.anchors, self.anchors_mask, pred_t, scale_t, fore_mask_t, back_mask_t, grid_wh, idx)

        cls_t = pred_t[..., 5][fore_mask_t].long().view(-1, 1)
        cls_pred = pred[..., 5:]
        conf_t = pred_t[..., 4]
        if cls_t.size(0) == 0:
            print("grid_wh {} no matching anchors".format(grid_wh))
            back_conf_t = conf_t[back_mask_t].view(-1, 1)
            back_conf_pred = pred[..., 4][back_mask_t].view(-1, 1)
            back_num = back_conf_pred.size(0)
            no_obj = back_conf_pred.sum().item() / back_num
            back_conf_loss = self.bce_loss(back_conf_pred, back_conf_t)
            if debug:
                print("grid_wh", grid_wh, "loc_loss", 0, "conf_loss", round(back_conf_loss.item(), 5), "cls_loss", 0, "Obj", 0, "no_obj", round(no_obj, 5))
            return torch.zeros(1), back_conf_loss, torch.zeros(1)

        scale_factor = scale_t[fore_mask_t].view(-1, 1)
        scale_factor = scale_factor.expand((scale_factor.size(0), 2))

        # cls loss
        cls_fore_mask_t = fore_mask_t.new_tensor(fore_mask_t).view(batch_size, num_pred, 1).expand_as(cls_pred)
        cls_pred = cls_pred[cls_fore_mask_t].view(-1, self.num_classes)
        class_mask = cls_pred.data.new(cls_t.size(0), self.num_classes).fill_(0)
        class_mask.scatter_(1, cls_t, 1.)
        cls_loss = self.bce_loss(cls_pred, class_mask)
        ave_cls = (class_mask * cls_pred).sum().item() / cls_pred.size(0)
        
        # conf loss
        fore_conf_t = conf_t[fore_mask_t].view(-1, 1)
        back_conf_t = conf_t[back_mask_t].view(-1, 1)
        fore_conf_pred = pred[..., 4][fore_mask_t].view(-1, 1)
        back_conf_pred = pred[..., 4][back_mask_t].view(-1, 1)
        fore_num = fore_conf_pred.size(0)
        back_num = back_conf_pred.size(0)
        Obj = fore_conf_pred.sum().item() / fore_num
        no_obj = back_conf_pred.sum().item() / back_num

        fore_conf_loss = self.bce_loss(fore_conf_pred, fore_conf_t)
        back_conf_loss = self.bce_loss(back_conf_pred, back_conf_t)
        conf_loss = fore_conf_loss + back_conf_loss  

        # loc loss
        loc_pred = pred[..., :4]
        loc_t = pred_t[..., :4]
        fore_mask_t = fore_mask_t.view(batch_size, num_pred, 1).expand_as(loc_pred)
        loc_t = loc_t[fore_mask_t].view(-1, 4)
        loc_pred = loc_pred[fore_mask_t].view(-1, 4)

        xy_t, wh_t = loc_t[:, :2], loc_t[:, 2:]
        xy_pred, wh_pred = loc_pred[:, :2], loc_pred[:, 2:]
        # xy_loss = F.binary_cross_entropy(xy_pred, xy_t, scale_factor, size_average=False)

        xy_loss = self.weight_mseloss(xy_pred, xy_t, scale_factor) / 2
        wh_loss = self.weight_mseloss(wh_pred, wh_t, scale_factor) / 2

        loc_loss = xy_loss + wh_loss        

        loc_loss /= batch_size
        conf_loss /= batch_size
        cls_loss /= batch_size

        if debug:
            print("grid_wh", grid_wh, "xy_loss", round(xy_loss.item(), 5), "wh_loss", round(wh_loss.item(), 5), "cls_loss", round(cls_loss.item(), 5), "ave_cls", round(ave_cls, 5), "Obj", round(Obj, 5), "no_obj", round(no_obj, 5), "fore_conf_loss", round(fore_conf_loss.item(), 5),
                "back_conf_loss", round(back_conf_loss.item(), 5))

        return loc_loss, conf_loss, cls_loss









 







