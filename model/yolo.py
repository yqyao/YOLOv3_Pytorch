# -*- coding: utf-8 -*-
# Written by yq_yao
# 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn.init as init
from model.darknet53 import Darknet53
import os
from utils.box_utils import permute_sigmoid, decode
from layers.yolo_layer import YoloLayer

def xavier(param):
    init.xavier_uniform(param)

# kaiming_weights_init
def weights_init(m):
    for key in m.state_dict():
        if key.split('.')[-1] == 'weight':
            if 'conv' in key:
                init.kaiming_normal_(m.state_dict()[key], mode='fan_out')
            if 'bn' in key:
                m.state_dict()[key][...] = 1
        elif key.split('.')[-1] == 'bias':
            m.state_dict()[key][...] = 0


# def weights_init(m):
#     for key in m.state_dict():
#         if key.split('.')[-1] == 'weight':
#             if 'conv' in key:
#                 init.xavier_uniform(m.state_dict()[key])
#             if 'bn' in key:
#                 m.state_dict()[key][...] = 1
#         elif key.split('.')[-1] == 'bias':
#             m.state_dict()[key][...] = 0

class ConvBN(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size=3, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(ch_in, ch_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(ch_out, momentum=0.01)

    def forward(self, x):
        return F.leaky_relu(self.bn(self.conv(x)), negative_slope=0.1, inplace=True)

class DetectionLayer(nn.Module):
    def __init__(self, anchors, anchors_mask, input_wh, num_classes):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors
        self.input_wh = input_wh
        self.anchors_mask = anchors_mask
        self.num_classes = num_classes
    
    def forward(self, x):
        l_data, m_data, h_data = x
        l_grid_wh = (l_data.size(3), l_data.size(2))
        m_grid_wh = (m_data.size(3), m_data.size(2))
        h_grid_wh = (h_data.size(3), h_data.size(2))

        pred_l, stride_l = permute_sigmoid(l_data, self.input_wh, 3, self.num_classes)
        pred_m, stride_m = permute_sigmoid(m_data, self.input_wh, 3, self.num_classes)
        pred_h, stride_h = permute_sigmoid(h_data, self.input_wh, 3, self.num_classes)

        anchors1 = self.anchors[self.anchors_mask[0][0]: self.anchors_mask[0][-1]+1]
        anchors2 = self.anchors[self.anchors_mask[1][0]: self.anchors_mask[1][-1]+1]
        anchors3 = self.anchors[self.anchors_mask[2][0]: self.anchors_mask[2][-1]+1]
        
        decode_l = decode(pred_l.detach(), self.input_wh, anchors1, self.num_classes, stride_l)
        decode_m = decode(pred_m.detach(), self.input_wh, anchors2, self.num_classes, stride_m)
        decode_h = decode(pred_h.detach(), self.input_wh, anchors3, self.num_classes, stride_h)
        decode_pred = torch.cat((decode_l, decode_m, decode_h), 1)

        return decode_pred

def predict_conv_list1(num_classes):
    layers = list()
    layers += [ConvBN(1024, 512, kernel_size=1, stride=1, padding=0)]
    layers += [ConvBN(512, 1024, kernel_size=3, stride=1, padding=1)]
    layers += [ConvBN(1024, 512, kernel_size=1, stride=1, padding=0)]
    layers += [ConvBN(512, 1024, kernel_size=3, stride=1, padding=1)]
    layers += [ConvBN(1024, 512, kernel_size=1, stride=1, padding=0)]
    layers += [ConvBN(512, 1024, kernel_size=3, stride=1, padding=1)]
    layers += [nn.Conv2d(1024, (5 + num_classes) * 3, kernel_size=1, stride=1, padding=0)]
    return layers

def predict_conv_list2(num_classes):
    layers = list()
    layers += [ConvBN(768, 256, kernel_size=1, stride=1, padding=0)]
    layers += [ConvBN(256, 512, kernel_size=3, stride=1, padding=1)]
    layers += [ConvBN(512, 256, kernel_size=1, stride=1, padding=0)]
    layers += [ConvBN(256, 512, kernel_size=3, stride=1, padding=1)]
    layers += [ConvBN(512, 256, kernel_size=1, stride=1, padding=0)]
    layers += [ConvBN(256, 512, kernel_size=3, stride=1, padding=1)]
    layers += [nn.Conv2d(512, (5 + num_classes) * 3, kernel_size=1, stride=1, padding=0)]
    return layers

def predict_conv_list3(num_classes):
    layers = list()
    layers += [ConvBN(384, 128, kernel_size=1, stride=1, padding=0)]
    layers += [ConvBN(128, 256, kernel_size=3, stride=1, padding=1)]
    layers += [ConvBN(256, 128, kernel_size=1, stride=1, padding=0)]
    layers += [ConvBN(128, 256, kernel_size=3, stride=1, padding=1)]
    layers += [ConvBN(256, 128, kernel_size=1, stride=1, padding=0)]
    layers += [ConvBN(128, 256, kernel_size=3, stride=1, padding=1)]
    layers += [nn.Conv2d(256, (5 + num_classes) * 3, kernel_size=1, stride=1, padding=0)]
    return layers

class YOLOv3(nn.Module):
    def __init__(self, phase, num_blocks, anchors, anchors_mask, input_wh, num_classes):
        super().__init__()
        self.phase = phase
        self.extractor = Darknet53(num_blocks)
        self.predict_conv_list1 = nn.ModuleList(predict_conv_list1(num_classes))
        self.smooth_conv1 = ConvBN(512, 256, kernel_size=1, stride=1, padding=0)
        self.predict_conv_list2 = nn.ModuleList(predict_conv_list2(num_classes))
        self.smooth_conv2 = ConvBN(256, 128, kernel_size=1, stride=1, padding=0)
        self.predict_conv_list3 = nn.ModuleList(predict_conv_list3(num_classes))
        if phase == "test":
            self.detection = DetectionLayer(anchors, anchors_mask, input_wh, num_classes)

    def forward(self, x, targets=None):
        c3, c4, c5 = self.extractor(x)
        x = c5
        # predict_list1
        for i in range(5):
            x = self.predict_conv_list1[i](x)
        smt1 = self.smooth_conv1(x)
        smt1 = F.upsample(smt1, scale_factor=2, mode='nearest')

        smt1 = torch.cat((smt1, c4), 1)
        for i in range(5, 7):
            x = self.predict_conv_list1[i](x)
        out1 = x

        x = smt1
        for i in range(5):
            x = self.predict_conv_list2[i](x)
        smt2 = self.smooth_conv2(x)
        smt2 = F.upsample(smt2, scale_factor=2, mode='nearest')
        smt2 = torch.cat((smt2, c3), 1)
        for i in range(5, 7):
            x = self.predict_conv_list2[i](x)
        out2 = x
        x = smt2
        for i in range(7):
            x = self.predict_conv_list3[i](x)
        out3 = x

        if self.phase == "test":
            detections = self.detection((out1, out2, out3))
            return detections
        elif self.phase == "train":
            detections = (out1, out2, out3)
            return detections
        
    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.extractor.load_state_dict(torch.load(base_file))
            print("initing  darknet53 ......")
            self.predict_conv_list1.apply(weights_init)
            self.smooth_conv1.apply(weights_init)
            self.predict_conv_list2.apply(weights_init)
            self.smooth_conv2.apply(weights_init)
            self.predict_conv_list3.apply(weights_init)
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')

def Yolov3(phase, input_wh, anchors, anchors_mask, num_classes):
    num_blocks = [1,2,8,8,4]
    return YOLOv3(phase, num_blocks, anchors, anchors_mask, input_wh, num_classes)
