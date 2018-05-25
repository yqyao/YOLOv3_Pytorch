from __future__ import division
import time
import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3" 
import torch.nn as nn
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import numpy as np
import cv2 
import argparse
import os.path as osp
import math
from model.darknet53 import Yolov3
from utils.box_utils import get_results, draw_rects, get_rects
from utils.preprocess import preproc_img
def arg_parse():
    """
    Parse arguements to the detect module
    
    """
    parser = argparse.ArgumentParser(description='YOLO v3 Detection Module')
   
    parser.add_argument("--images", dest = 'images', help = 
                        "Image / Directory containing images to perform detection upon",default = "images", type = str)
    parser.add_argument("--confidence", dest = "confidence", help = "Object Confidence to filter predictions", default = 0.5)
    parser.add_argument("--num_classes", dest = "num_classes", help = "num classes", default = 80)
    parser.add_argument("--nms_thresh", dest = "nms_thresh", help = "NMS Threshhold", default = 0.4)
    parser.add_argument("--input_dim", dest = "input_dim", help = "input dim", default = 416)
    parser.add_argument("--coco_name_path", dest = "coco_name_path", help = "coco name path", default = './model/coco.names')
    parser.add_argument("--save_path", dest = "save_path", help = "coco name path", default = './output')
    parser.add_argument("--weights", dest = 'weights',
                        help = "weightsfile",
                        default = "./weights/convert_yolov3.pth", type = str)
    parser.add_argument('--cuda', default=True, type=str,
                    help='Use cuda to train model')
    parser.add_argument('--use_pad', default=False, type=str,
                    help='Use pad to resize images')
    return parser.parse_args()


if __name__ ==  '__main__':
    args = arg_parse()
    weightsfile = args.weights
    confidence = args.confidence
    nms_thresh = args.nms_thresh
    images = args.images
    coco_name_path = args.coco_name_path
    num_classes = args.num_classes
    input_dim = args.input_dim
    cuda = args.cuda
    use_pad = args.use_pad
    save_path = args.save_path
    with open(coco_name_path, "r") as f:
        classes = [i.strip() for i in f.readlines()]
    try:
        im_list = [osp.join(osp.realpath('.'), images, img) for img in os.listdir(images)]
    except NotADirectoryError:
        im_list = []
        im_list.append(osp.join(osp.realpath('.'), images))
    except FileNotFoundError:
        print ("No file or directory with the name {}".format(images))
        exit()
    net = Yolov3(input_dim, num_classes)
    if cuda:
        net.cuda()
        cudnn.benchmark = True
    net.load_state_dict(torch.load(weightsfile))
    print("load weights successfully.....")
    net.eval()
    for img_path in im_list[:]:
        img = cv2.imread(img_path)
        img, ori_img, ori_dim = preproc_img(img, (input_dim, input_dim), use_pad)
        if cuda:
            img = img.cuda()
        st = time.time()
        detection = net(img)
        detect_time = time.time()
        detection = get_results(detection, confidence, num_classes, nms_conf=nms_thresh)
        nms_time = time.time()
        detection = get_rects(detection, input_dim, ori_dim, use_pad)
        draw_img = draw_rects(ori_img, detection, classes)
        draw_time = time.time()
        save_img_path = os.path.join(save_path, "output_" + img_path.split("/")[-1])
        cv2.imwrite(save_img_path, draw_img)
        final_time = time.time() - st

        print("detection time:", round(detect_time - st, 3), "nms_time:", round(nms_time - detect_time, 3), "draw_time:", round(draw_time - nms_time, 3), "final_time:", round(final_time ,3))









