# -*- coding: utf-8 -*-
# Written by yq_yao

from __future__ import division
import time
import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1" 
import torch.nn as nn
import torch.backends.cudnn as cudnn
import numpy as np
import cv2 
import argparse
import os.path as osp
import math
from model.yolo import Yolov3
from utils.box_utils import draw_rects, detection_postprecess
from data.config import voc_config, coco_config
from utils.preprocess import preproc_for_test

def arg_parse():
    """
    Parse arguements to the detect module
    """
    parser = argparse.ArgumentParser(description='YOLO v3 Detection Module')
   
    parser.add_argument("--images", dest = 'images', help = 
                        "Image / Directory containing images to perform detection upon",default = "images", type = str)
    parser.add_argument("--confidence", dest = "confidence", help = "Object Confidence to filter predictions", default = 0.1)
    parser.add_argument("--nms_thresh", dest = "nms_thresh", help = "NMS Threshhold", default = 0.4)
    parser.add_argument("--input_wh", dest = "input_wh", help = "input_wh", default = (416, 416))
    parser.add_argument("--save_path", dest = "save_path", help = "coco name path", default = './output')
    parser.add_argument("--dataset", dest = "dataset", help = "VOC or COCO", default = 'VOC')
    parser.add_argument("--weights", dest = 'weights',
                        help = "weightsfile",
                        default = "./weights/convert_yolov3_coco.pth", type = str)
    parser.add_argument('--cuda', default=True, type=str,
                    help='Use cuda to train model')
    parser.add_argument('--use_pad', default=True, type=str,
                    help='Use pad to resize images')
    return parser.parse_args()


if __name__ ==  '__main__':
    args = arg_parse()
    weightsfile = args.weights
    confidence = args.confidence
    nms_thresh = args.nms_thresh
    images = args.images
    input_wh = args.input_wh
    cuda = args.cuda
    use_pad = args.use_pad
    save_path = args.save_path
    dataset = args.dataset
    if dataset[0] == "V":
        cfg = voc_config
    elif dataset[1] == "C":
        cfg = coco_config
    else:
        print("only support VOC and COCO datasets !!!")
    name_path = cfg["name_path"]
    num_classes = cfg["num_classes"]
    anchors = cfg["anchors"]

    with open(name_path, "r") as f:
        classes = [i.strip() for i in f.readlines()]
    try:
        im_list = [osp.join(osp.realpath('.'), images, img) for img in os.listdir(images)]
    except NotADirectoryError:
        im_list = []
        im_list.append(osp.join(osp.realpath('.'), images))
    except FileNotFoundError:
        print ("No file or directory with the name {}".format(images))
        exit()

    net = Yolov3("test", input_wh, anchors, cfg["anchors_mask"], num_classes)
    state_dict = torch.load(weightsfile)
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        head = k[:7]
        if head == 'module.':
            name = k[7:] # remove `module.`
        else:
            name = k
        new_state_dict[name] = v
    if cuda:
        net.cuda()
        cudnn.benchmark = True
    net.load_state_dict(new_state_dict)
    print("load weights successfully.....")
    net.eval()
    for img_path in im_list[:]:
        print(img_path)
        img = cv2.imread(img_path)
        ori_img = img.copy()
        ori_wh = (img.shape[1], img.shape[0])
        img = preproc_for_test(img, input_wh, use_pad)
        if cuda:
            img = img.cuda()
        st = time.time()
        detection = net(img)
        detect_time = time.time()
        detection = detection_postprecess(detection, confidence, num_classes, input_wh, ori_wh, use_pad=use_pad, nms_conf=nms_thresh)
        nms_time = time.time()
        draw_img = draw_rects(ori_img, detection, classes)
        draw_time = time.time()
        save_img_path = os.path.join(save_path, "output_" + img_path.split("/")[-1])
        cv2.imwrite(save_img_path, draw_img)
        final_time = time.time() - st

        print("detection time:", round(detect_time - st, 3), "nms_time:", round(nms_time - detect_time, 3), "draw_time:", round(draw_time - nms_time, 3), "final_time:", round(final_time ,3))









