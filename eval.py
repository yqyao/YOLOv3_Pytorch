# -*- coding: utf-8 -*-
# Written by yq_yao

from __future__ import division
import time
import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1" 
import torch.nn as nn
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import numpy as np
import cv2 
import argparse
import os.path as osp
import math
import pickle
from model.yolo import Yolov3
from data.voc0712 import  VOCDetection, detection_collate
from data.coco import COCODetection
from data.config import voc_config, coco_config, datasets_dict
from utils.box_utils import draw_rects, detection_postprecess
from utils.timer import Timer
from utils.preprocess import preproc_for_test

def arg_parse():
    """
    Parse arguements to the detect module
    
    """
    parser = argparse.ArgumentParser(description='YOLO v3 Detection Module')
   
    parser.add_argument('--dataset', default='VOC',
                    help='VOC ,VOC0712++ or COCO dataset')
    parser.add_argument("--nms_thresh", dest = "nms_thresh", help = "NMS Threshhold", default = 0.4)
    parser.add_argument("--input_wh", dest = "input_wh", help = "input wh", default = (416, 416))
    parser.add_argument("--weights", dest = 'weights',
                        help = "weightsfile",
                        default = "./weights/yolov3_COCO_epoches_10_0607.pth", type = str)
    parser.add_argument('--cuda', default=True, type=str,
                    help='Use cuda to train model')
    parser.add_argument('--use_pad', default=True, type=str,
                    help='Use pad to resize images')
    parser.add_argument('--retest', default=False, type=bool,
                    help='test cache results')
    parser.add_argument('--save_folder', default='./eval/',
                        help='results path')
    return parser.parse_args()

def test_net(cfg, save_folder, input_wh, net, cuda, testset,
             max_per_image=300, thresh=0.05, nms_conf=0.4):
    """Test a Fast R-CNN network on an image database."""
    num_images = len(testset)
    # all detections are collected into:
    #    all_boxes[cls][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)
    num_images = len(testset)
    num_classes = cfg["num_classes"]
    all_boxes = [[[] for _ in range(num_images)]
                 for _ in range(num_classes)]

    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    # timers
    _t = {'im_detect': Timer(), 'misc': Timer()}
    det_file = os.path.join(save_folder, 'detections.pkl')

    if args.retest:
        f = open(det_file,'rb')
        all_boxes = pickle.load(f)
        print('Evaluating detections')
        testset.evaluate_detections(all_boxes, save_folder)
        return
        
    for i in range(num_images):
        img, img_id = testset.pull_image(i)
        ori_wh = (img.shape[1], img.shape[0])
        img = preproc_for_test(img, input_wh, use_pad)
        x = img
        if cuda:
            x = x.cuda()

        _t['im_detect'].tic()
        out = net(x)      # forward pass
        detections = detection_postprecess(out, thresh, num_classes, input_wh, ori_wh, use_pad=use_pad, nms_conf=nms_conf)
        boxes, scores, cls_inds = detections[:, :4], detections[:,4], detections[:, -1]
        detect_time = _t['im_detect'].toc()
        if len(boxes) == 0:
            continue

        _t['misc'].tic()        
        for j in range(num_classes):
            inds = np.where(cls_inds == j)[0]
            if len(inds) == 0:
                all_boxes[j][i] = np.empty([0, 5], dtype=np.float32)
                continue
            c_bboxes = boxes[inds]
            c_scores = scores[inds]
            c_dets = np.hstack((c_bboxes, c_scores[:, np.newaxis])).astype(
                np.float32, copy=False)
            all_boxes[j][i] = c_dets

        if max_per_image > 0:
            image_scores = np.hstack([all_boxes[j][i][:, -1] for j in range(num_classes)])
            if len(image_scores) > max_per_image:
                image_thresh = np.sort(image_scores)[-max_per_image]
                for j in range(num_classes):
                    keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                    all_boxes[j][i] = all_boxes[j][i][keep, :]
        nms_time = _t['misc'].toc()
 
        if i % 20 == 0:
            print('im_detect: {:d}/{:d} {:.3f}s {:.3f}s'
                .format(i + 1, num_images, detect_time, nms_time))
            _t['im_detect'].clear()
            _t['misc'].clear()

    with open(det_file, 'wb') as f:
        pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)
    print('Evaluating detections')
    testset.evaluate_detections(all_boxes, save_folder)

if __name__ ==  '__main__':
    args = arg_parse()
    weightsfile = args.weights
    nms_thresh = args.nms_thresh
    input_wh = args.input_wh
    cuda = args.cuda
    use_pad = args.use_pad
    save_folder = args.save_folder
    dataset = args.dataset
    if dataset[0] == "V":
        cfg = voc_config
        test_dataset = VOCDetection(cfg["root"], datasets_dict["VOC2007"], input_wh)
    elif dataset[0] == "C":
        cfg =  coco_config
        test_dataset = COCODetection(cfg["root"], datasets_dict["COCOval"], input_wh) 
    else:
        print("only support VOC and COCO datasets !!!")

    print("load test_dataset successfully.....")

    with open(cfg["name_path"], "r") as f:
        classes = [i.strip() for i in f.readlines()]

    net = Yolov3("test", input_wh, cfg["anchors"], cfg["anchors_mask"], cfg["num_classes"])
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

    top_k = 200
    confidence = 0.01
    test_net(cfg, save_folder, input_wh, net, args.cuda, test_dataset, top_k, confidence, nms_thresh)








