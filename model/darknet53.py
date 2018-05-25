# -*- coding: utf-8 -*-
# Written by yq_yao
# 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils.box_utils import predict_transform

class ConvBN(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size=3, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(ch_in, ch_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(ch_out, momentum=0.01, eps=1e-05, affine=True)

    def forward(self, x):
        return F.leaky_relu(self.bn(self.conv(x)), negative_slope=0.1, inplace=True)

class DarknetBlock(nn.Module):
    def __init__(self, ch_in):
        super().__init__()
        ch_hid = ch_in // 2
        self.conv1 = ConvBN(ch_in, ch_hid, kernel_size=1, stride=1, padding=0)
        self.conv2 = ConvBN(ch_hid, ch_in, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        return out + x

class DetectionLayer(nn.Module):
    def __init__(self, anchors, input_dim, num_classes):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors
        self.input_dim = input_dim
        self.num_classes = num_classes
    
    def forward(self, x, y, z):
        x = x.data
        y = y.data
        z = z.data
        prediction_x = x
        prediction_y = y
        prediction_z = z
        prediction_x = predict_transform(prediction_x, self.input_dim, self.anchors[0], self.num_classes, True)
        prediction_y = predict_transform(prediction_y, self.input_dim, self.anchors[1], self.num_classes, True)
        prediction_z = predict_transform(prediction_z, self.input_dim, self.anchors[2], self.num_classes, True)
        prediction = torch.cat((prediction_x, prediction_y, prediction_z), 1)
        return prediction

class Darknet19(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.conv = ConvBN(3, 32, kernel_size=3, stride=1, padding=1)
        self.layer1 = self._make_layer1()
        self.layer2 = self._make_layer2()
        self.layer3 = self._make_layer3()
        self.layer4 = self._make_layer4()
        self.layer5 = self._make_layer5()

    def _make_layer1(self):
        layers = [nn.MaxPool2d(kernel_size=2, stride=2),
                    ConvBN(32, 64, kernel_size=3, stride=1, padding=1)]
        return nn.Sequential(*layers)
        
    def _make_layer2(self):
        layers = [nn.MaxPool2d(kernel_size=2, stride=2),
                  ConvBN(64, 128, kernel_size=3, stride=1, padding=1),
                  ConvBN(128, 64, kernel_size=1, stride=1, padding=1),
                  ConvBN(64, 128, kernel_size=3, stride=1, padding=1)]
        return nn.Sequential(*layers)

    def _make_layer3(self):
        layers = [nn.MaxPool2d(kernel_size=2, stride=2),
                  ConvBN(128, 256, kernel_size=3, stride=1, padding=1),
                  ConvBN(256, 128, kernel_size=1, stride=1, padding=1),
                  ConvBN(128, 256, kernel_size=3, stride=1, padding=1)]
        return nn.Sequential(*layers)

    def _make_layer4(self):
        layers = [nn.MaxPool2d(kernel_size=2, stride=2),
                  ConvBN(256, 512, kernel_size=3, stride=1, padding=1),
                  ConvBN(512, 256, kernel_size=1, stride=1, padding=1),
                  ConvBN(256, 512, kernel_size=3, stride=1, padding=1),
                  ConvBN(512, 256, kernel_size=1, stride=1, padding=1),
                  ConvBN(256, 512, kernel_size=3, stride=1, padding=1)]
        return nn.Sequential(*layers)        

    def _make_layer5(self):
        layers = [nn.MaxPool2d(kernel_size=2, stride=2),
                  ConvBN(512, 1024, kernel_size=3, stride=1, padding=1),
                  ConvBN(1024, 512, kernel_size=1, stride=1, padding=1),
                  ConvBN(512, 1024, kernel_size=3, stride=1, padding=1),
                  ConvBN(1024, 512, kernel_size=1, stride=1, padding=1),
                  ConvBN(512, 1024, kernel_size=3, stride=1, padding=1)]
        return nn.Sequential(*layers) 

    def forward(self, x):
        out = self.conv(x)
        c1 = self.layer1(out)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)
        c5 = self.layer5(c4)

def predict_conv_list1():
    layers = list()
    layers += [ConvBN(1024, 512, kernel_size=1, stride=1, padding=0)]
    layers += [ConvBN(512, 1024, kernel_size=3, stride=1, padding=1)]
    layers += [ConvBN(1024, 512, kernel_size=1, stride=1, padding=0)]
    layers += [ConvBN(512, 1024, kernel_size=3, stride=1, padding=1)]
    layers += [ConvBN(1024, 512, kernel_size=1, stride=1, padding=0)]
    layers += [ConvBN(512, 1024, kernel_size=3, stride=1, padding=1)]
    layers += [nn.Conv2d(1024, 255, kernel_size=1, stride=1, padding=0)]
    return layers

def predict_conv_list2():
    layers = list()
    layers += [ConvBN(768, 256, kernel_size=1, stride=1, padding=0)]
    layers += [ConvBN(256, 512, kernel_size=3, stride=1, padding=1)]
    layers += [ConvBN(512, 256, kernel_size=1, stride=1, padding=0)]
    layers += [ConvBN(256, 512, kernel_size=3, stride=1, padding=1)]
    layers += [ConvBN(512, 256, kernel_size=1, stride=1, padding=0)]
    layers += [ConvBN(256, 512, kernel_size=3, stride=1, padding=1)]
    layers += [nn.Conv2d(512, 255, kernel_size=1, stride=1, padding=0)]
    return layers

def predict_conv_list3():
    layers = list()
    layers += [ConvBN(384, 128, kernel_size=1, stride=1, padding=0)]
    layers += [ConvBN(128, 256, kernel_size=3, stride=1, padding=1)]
    layers += [ConvBN(256, 128, kernel_size=1, stride=1, padding=0)]
    layers += [ConvBN(128, 256, kernel_size=3, stride=1, padding=1)]
    layers += [ConvBN(256, 128, kernel_size=1, stride=1, padding=0)]
    layers += [ConvBN(128, 256, kernel_size=3, stride=1, padding=1)]
    layers += [nn.Conv2d(256, 255, kernel_size=1, stride=1, padding=0)]
    return layers


class Darknet53(nn.Module):
    def __init__(self, num_blocks, anchors, input_dim, num_classes):
        super().__init__()
        self.conv = ConvBN(3, 32, kernel_size=3, stride=1, padding=1)
        self.layer1 = self._make_layer(32, num_blocks[0], stride=2)
        self.layer2 = self._make_layer(64, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(128, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(256, num_blocks[3], stride=2)
        self.layer5 = self._make_layer(512, num_blocks[4], stride=2)
        self.predict_conv_list1 = nn.ModuleList(predict_conv_list1())
        self.smooth_conv1 = ConvBN(512, 256, kernel_size=1, stride=1, padding=0)
        self.predict_conv_list2 = nn.ModuleList(predict_conv_list2())
        self.smooth_conv2 = ConvBN(256, 128, kernel_size=1, stride=1, padding=0)
        self.predict_conv_list3 = nn.ModuleList(predict_conv_list3())
        self.detection = DetectionLayer(anchors, input_dim, num_classes)

    def _make_layer(self, ch_in, num_blocks, stride=1):
        layers = [ConvBN(ch_in, ch_in*2, stride=stride, padding=1)]
        for i in range(num_blocks):
            layers.append(DarknetBlock(ch_in * 2))
        return nn.Sequential(*layers) 

    def forward(self, x):
        out = self.conv(x)
        c1 = self.layer1(out)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)
        c5 = self.layer5(c4)
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
        detections = self.detection(out1, out2, out3)
        return detections

def Yolov3(input_dim, num_classes):
    anchors = [[(116, 90), (156, 198), (373, 326)], 
                [(30, 61), (62, 45), (59, 119)], 
                [(10, 13), (16, 30), (33, 23)]]
    num_blocks = [1,2,8,8,4]
    return Darknet53(num_blocks, anchors, input_dim, num_classes)
