# -*- coding: utf-8 -*-
# Written by yq_yao
# 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

from model.darknet53 import Darknet53

def copy_weights_53(bn, conv, ptr, weights, use_bn=True):
    if use_bn:
        num_bn_biases = bn.bias.numel()
        
        #Load the weights
        bn_biases = torch.from_numpy(weights[ptr:ptr + num_bn_biases])
        ptr += num_bn_biases
        
        bn_weights = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
        ptr  += num_bn_biases
        
        bn_running_mean = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
        ptr  += num_bn_biases
        
        bn_running_var = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
        ptr  += num_bn_biases
        
        #Cast the loaded weights into dims of model weights. 
        bn_biases = bn_biases.view_as(bn.bias.data)
        bn_weights = bn_weights.view_as(bn.weight.data)
        bn_running_mean = bn_running_mean.view_as(bn.running_mean)
        bn_running_var = bn_running_var.view_as(bn.running_var)

        #Copy the data to model
        bn.bias.data.copy_(bn_biases)
        bn.weight.data.copy_(bn_weights)
        bn.running_mean.copy_(bn_running_mean)
        bn.running_var.copy_(bn_running_var)
    else:
        #Number of biases
        num_biases = conv.bias.numel()
    
        #Load the weights
        conv_biases = torch.from_numpy(weights[ptr: ptr + num_biases])
        ptr = ptr + num_biases
        
        #reshape the loaded weights according to the dims of the model weights
        conv_biases = conv_biases.view_as(conv.bias.data)
        
        #Finally copy the data
        conv.bias.data.copy_(conv_biases)
    
    #Let us load the weights for the Convolutional layers
    num_weights = conv.weight.numel()
    conv_weights = torch.from_numpy(weights[ptr:ptr+num_weights])
    ptr = ptr + num_weights

    conv_weights = conv_weights.view_as(conv.weight.data)
    conv.weight.data.copy_(conv_weights)
    return ptr


def load_weights_darknet53(weightfile, darknet53):
    fp = open(weightfile, "rb")
    #The first 5 values are header information 
    # 1. Major version number
    # 2. Minor Version Number
    # 3. Subversion number 
    # 4, 5. IMages seen 
    header = np.fromfile(fp, dtype = np.int32, count = 5)
    weights = np.fromfile(fp, dtype = np.float32)
    print(len(weights))
    ptr = 0
    first_conv = darknet53.conv
    bn = first_conv.bn
    conv = first_conv.conv
    # first conv copy
    ptr = copy_weights_53(bn, conv, ptr, weights)

    layers = [darknet53.layer1, darknet53.layer2, darknet53.layer3, darknet53.layer4, darknet53.layer5]
    for layer in layers:
        for i in range(len(layer)):
            if i == 0:
                bn = layer[i].bn
                conv = layer[i].conv
                ptr = copy_weights_53(bn, conv, ptr, weights)
            else:
                bn = layer[i].conv1.bn
                conv = layer[i].conv1.conv
                ptr = copy_weights_53(bn, conv, ptr, weights)
                bn = layer[i].conv2.bn
                conv = layer[i].conv2.conv
                ptr = copy_weights_53(bn, conv, ptr, weights)
    predict_conv_list1 = darknet53.predict_conv_list1
    smooth_conv1 = darknet53.smooth_conv1
    predict_conv_list2 = darknet53.predict_conv_list2
    smooth_conv2 = darknet53.smooth_conv2
    predict_conv_list3 = darknet53.predict_conv_list3
    for i in range(len(predict_conv_list1)):
        if i == 6:
            bn = 0
            conv = predict_conv_list1[i]
            ptr = copy_weights_53(bn, conv, ptr, weights, use_bn=False)
        else:
            bn = predict_conv_list1[i].bn
            conv = predict_conv_list1[i].conv
            ptr = copy_weights_53(bn, conv, ptr, weights)
    bn = smooth_conv1.bn
    conv = smooth_conv1.conv
    ptr = copy_weights_53(bn, conv, ptr, weights)
    for i in range(len(predict_conv_list2)):
        if i == 6:
            bn = 0
            conv = predict_conv_list2[i]
            ptr = copy_weights_53(bn, conv, ptr, weights, use_bn=False)
        else:
            bn = predict_conv_list2[i].bn
            conv = predict_conv_list2[i].conv
            ptr = copy_weights_53(bn, conv, ptr, weights)
    bn = smooth_conv2.bn
    conv = smooth_conv2.conv
    ptr = copy_weights_53(bn, conv, ptr, weights)

    for i in range(len(predict_conv_list3)):
        if i == 6:
            bn = 0
            conv = predict_conv_list3[i]
            ptr = copy_weights_53(bn, conv, ptr, weights, use_bn=False)
        else:
            bn = predict_conv_list3[i].bn
            conv = predict_conv_list3[i].conv
            ptr = copy_weights_53(bn, conv, ptr, weights)

    fp.close()               


def load_weights_darknet19(weightfile, darknet19):
    fp = open(weightfile, "rb")
    #The first 4 values are header information 
    # 1. Major version number
    # 2. Minor Version Number
    # 3. Subversion number 
    # 4. IMages seen 
    header = np.fromfile(fp, dtype = np.int32, count=4)
    weights = np.fromfile(fp, dtype = np.float32)
    ptr = 0
    first_conv = darknet19.conv
    bn = first_conv.bn
    conv = first_conv.conv
    # first conv copy
    ptr = copy_weights_53(bn, conv, ptr, weights)
    layers = [darknet19.layer1, darknet19.layer2, darknet19.layer3, darknet19.layer4, darknet19.layer5]
    for layer in layers:
        for i in range(len(layer)):
            if i == 0:
                pass
            else:
                bn = layer[i].bn
                conv = layer[i].conv
                ptr = copy_weights_53(bn, conv, ptr, weights)
    fp.close()

if __name__ == '__main__':
    anchors = [(373, 326), (156, 198), (116, 90), 
                (59, 119), (62, 45), (30, 61), 
                (33, 23), (16, 30), (10, 13)]
    num_blocks = [1,2,8,8,4]
    input_dim = 416
    num_classes = 80
    darknet53 = Darknet53(num_blocks, anchors, input_dim, num_classes)
    weightfile = "./weights/yolov3.weights"
    load_weights_darknet53(weightfile, darknet53)
    torch.save(darknet53.state_dict(), "./weights/convert_yolov3.pth")



