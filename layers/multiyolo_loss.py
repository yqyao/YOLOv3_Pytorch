# -*- coding: utf-8 -*-
# Written by yq_yao
# 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from .weight_mseloss import WeightMseLoss
from utils.box_utils import targets_match_all, permute_sigmoid, decode

class MultiYoloLoss(nn.Module):

    def __init__(self, input_wh, num_classes, ignore_thresh, anchors, anchors_mask, use_gpu=True):
        super(MultiYoloLoss, self).__init__()
        self.num_classes = num_classes
        self.ignore_thresh = ignore_thresh
        self.use_gpu = use_gpu
        self.anchors = anchors
        self.mse_loss = nn.MSELoss(size_average=False)
        self.bce_loss = nn.BCELoss(size_average=False)
        self.weight_mseloss = WeightMseLoss(size_average=False)
        self.input_wh = input_wh
        self.anchors_mask = anchors_mask

    def forward(self, x, targets, input_wh, debug=False):
        self.input_wh = input_wh
        l_data, m_data, h_data = x
        l_grid_wh = (l_data.size(3), l_data.size(2))
        m_grid_wh = (m_data.size(3), m_data.size(2))
        h_grid_wh = (h_data.size(3), h_data.size(2))
        feature_dim = (l_grid_wh, m_grid_wh, h_grid_wh)
        batch_size = l_data.size(0)
        pred_l, stride_l = permute_sigmoid(l_data, self.input_wh, 3, self.num_classes)
        pred_m, stride_m = permute_sigmoid(m_data, self.input_wh, 3, self.num_classes)
        pred_h, stride_h = permute_sigmoid(h_data, self.input_wh, 3, self.num_classes)
        pred = torch.cat((pred_l, pred_m, pred_h), 1)

        anchors1 = self.anchors[self.anchors_mask[0][0]: self.anchors_mask[0][-1]+1]
        anchors2 = self.anchors[self.anchors_mask[1][0]: self.anchors_mask[1][-1]+1]
        anchors3 = self.anchors[self.anchors_mask[2][0]: self.anchors_mask[2][-1]+1]

        decode_l = decode(pred_l.new_tensor(pred_l).detach(), self.input_wh, anchors1, self.num_classes, stride_l)
        decode_m = decode(pred_m.new_tensor(pred_m).detach(), self.input_wh, anchors2, self.num_classes, stride_m)
        decode_h = decode(pred_h.new_tensor(pred_h).detach(), self.input_wh, anchors3, self.num_classes, stride_h)
        decode_pred = torch.cat((decode_l, decode_m, decode_h), 1)

        num_pred = pred_l.size(1) + pred_m.size(1) + pred_h.size(1)

        # prediction targets x,y,w,h,objectness, class
        pred_t = torch.Tensor(batch_size, num_pred, 6).cuda()
        # xywh scale, scale = 2 - truth.w * truth.h (if truth is normlized to 1)
        scale_t = torch.FloatTensor(batch_size, num_pred).cuda()
        # foreground targets mask
        fore_mask_t = torch.ByteTensor(batch_size, num_pred).cuda()

        # background targets mask, we only calculate the objectness pred loss 
        back_mask_t = torch.ByteTensor(batch_size, num_pred).cuda()

        for idx in range(batch_size):
            # match all targets
            targets_match_all(self.input_wh, self.ignore_thresh, targets[idx], decode_pred[idx][:, :4], self.anchors, feature_dim, pred_t, scale_t, fore_mask_t, back_mask_t, num_pred, idx)

        scale_factor = scale_t[fore_mask_t].view(-1, 1)
        scale_factor = scale_factor.expand((scale_factor.size(0), 2))
        cls_t = pred_t[..., 5][fore_mask_t].long().view(-1, 1)
        cls_pred = pred[..., 5:]

        # cls loss
        cls_fore_mask_t = fore_mask_t.new_tensor(fore_mask_t).view(batch_size, num_pred, 1).expand_as(cls_pred)
        cls_pred = cls_pred[cls_fore_mask_t].view(-1, self.num_classes)
        class_mask = cls_pred.data.new(cls_t.size(0), self.num_classes).fill_(0)
        class_mask.scatter_(1, cls_t, 1.)
        cls_loss = self.bce_loss(cls_pred, class_mask)
        ave_cls = (class_mask * cls_pred).sum().item() / cls_pred.size(0)
        
        # conf loss
        conf_t = pred_t[..., 4]
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
            print("xy_loss", round(xy_loss.item(), 5), "wh_loss", round(wh_loss.item(), 5), "cls_loss", round(cls_loss.item(), 5), "ave_cls", round(ave_cls, 5), "Obj", round(Obj, 5), "no_obj", round(no_obj, 5), "fore_conf_loss", round(fore_conf_loss.item(), 5),
                "back_conf_loss", round(back_conf_loss.item(), 5))

        loss = loc_loss + conf_loss + cls_loss

        return loss







