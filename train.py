# -*- coding: utf-8 -*-
# Written by yq_yao

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import argparse
import torch.utils.data as data
from data.voc0712 import VOCDetection, detection_collate
from data.coco import COCODetection
from model.yolo import Yolov3
from data.config import voc_config, coco_config
from layers.yolo_loss import YoloLoss
from layers.multiyolo_loss import MultiYoloLoss
import numpy as np
import time
import os 
import sys


def arg_parse():
    """
    Parse arguments to the train module
    """
    parser = argparse.ArgumentParser(
        description='Yolov3 pytorch Training')
    parser.add_argument('-v', '--version', default='yolov3',
                        help='')
    parser.add_argument("--input_wh", dest = "input_wh", type=int, nargs=2, default = [416, 416])
    parser.add_argument('-d', '--dataset', default='VOC',
                        help='VOC or COCO dataset')
    parser.add_argument('-b', '--batch_size', default=64,
                        type=int, help='Batch size for training')
    parser.add_argument('--basenet', default='./weights/convert_darknet53.pth', help='pretrained base model')
    parser.add_argument('--ignore_thresh', default=0.5,
                        type=float, help='ignore_thresh')
    parser.add_argument('--subdivisions', default=4,
                        type=int, help='subdivisions for large batch_size')
    parser.add_argument('--num_workers', default=4,
                        type=int, help='Number of workers used in dataloading')
    parser.add_argument('--cuda', default=True,
                        type=bool, help='Use cuda to train model')
    parser.add_argument('--merge_yolo_loss', default=True,
                        type=bool, help='merge yolo loss')
    parser.add_argument('--lr', '--learning-rate',
                        default=1e-3, type=float, help='initial learning rate')
    parser.add_argument('--ngpu', default=2, type=int, help='gpus')

    parser.add_argument('--resume_net', default=None, 
                        help='resume net for retraining')
    parser.add_argument('--resume_epoch', default=0,
                        type=int, help='resume iter for retraining')
    parser.add_argument('-max','--max_epoch', default=200,
                        type=int, help='max epoch for retraining')
    parser.add_argument('--save_folder', default='./weights/',
                        help='Location to save checkpoint models')

    return parser.parse_args()

def adjust_learning_rate(optimizer, gamma, epoch, step_index, iteration, epoch_size):
    """Sets the learning rate 
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    if iteration < 1000:
        # warm up training
        lr = 0.001 * pow((iteration)/1000, 4)
    else:
        lr = args.lr * (gamma ** (step_index))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


if __name__ ==  '__main__':
    args = arg_parse()
    basenet = args.basenet
    save_folder = args.save_folder
    input_wh = args.input_wh
    batch_size = args.batch_size
    weight_decay = 0.0005
    gamma = 0.1
    momentum = 0.9
    cuda = args.cuda
    dataset_name = args.dataset
    subdivisions = args.subdivisions
    ignore_thresh = args.ignore_thresh
    merge_yolo_loss = args.merge_yolo_loss
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    if cuda and torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')

    # different datasets, include coco, voc0712 trainval, coco val
    datasets_version = {"VOC": [('0712', '0712_trainval')],
            "VOC0712++": [('0712', '0712_trainval_test')],
            "VOC2012" : [('2012', '2012_trainval')],
            "COCO": [('2014', 'train'), ('2014', 'valminusminival')],
            "VOC2007": [('0712', "2007_test")],
            "COCOval": [('2014', 'minival')]}

    print('Loading Dataset...')     
    if dataset_name[0] == "V":
        cfg = voc_config
        train_dataset = VOCDetection(cfg["root"], datasets_version[dataset_name], input_wh, batch_size, cfg["multiscale"], dataset_name)
    elif dataset_name[0] == "C":
        cfg = coco_config
        train_dataset = COCODetection(cfg["root"], datasets_version[dataset_name], input_wh, batch_size, cfg["multiscale"], dataset_name) 
    else:
        print('Unkown dataset!')

    # load Yolov3 net
    net = Yolov3("train", input_wh, cfg["anchors"], cfg["anchors_mask"], cfg["num_classes"])
    if args.resume_net == None:
        net.load_weights(basenet)
    else:
        state_dict = torch.load(args.resume_net)
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            head = k[:7]
            if head == 'module.':
                name = k[7:] # remove `module.`
            else:
                name = k
            new_state_dict[name] = v
        net.load_state_dict(new_state_dict)
        print('Loading resume network...')

    if args.ngpu > 1:
        net = torch.nn.DataParallel(net)

    if args.cuda:
        net.cuda()
        cudnn.benchmark = True

    optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=momentum, weight_decay=weight_decay)

    # load yolo loss
    if merge_yolo_loss:
        criterion = MultiYoloLoss(input_wh, cfg["num_classes"], ignore_thresh, cfg["anchors"], cfg["anchors_mask"])
    else:
        criterion = YoloLoss(input_wh, cfg["num_classes"], ignore_thresh, cfg["anchors"], cfg["anchors_mask"])
    net.train()
    ave_loss = -1
    epoch = 0 + args.resume_epoch
    mini_batch_size = int(batch_size / subdivisions)

    epoch_size = len(train_dataset) // (batch_size)
    max_iter = args.max_epoch * epoch_size

    stepvalues_VOC = (160 * epoch_size, 180 * epoch_size, 201 * epoch_size)
    stepvalues_COCO = (90 * epoch_size, 120 * epoch_size, 140 * epoch_size)
    stepvalues = (stepvalues_VOC, stepvalues_COCO)[args.dataset=='COCO']

    print('Training', args.version, 'on', train_dataset.name)
    step_index = 0

    if args.resume_epoch > 0:
        start_iter = args.resume_epoch * epoch_size
    else:
        start_iter = 0

    lr = args.lr

    # begin to train
    for iteration in range(start_iter, max_iter):
        if iteration % epoch_size == 0:
            batch_iterator = iter(data.DataLoader(train_dataset, 
                                                mini_batch_size,
                                                shuffle=False,
                                                num_workers=args.num_workers,
                                                collate_fn=detection_collate))
            if (epoch % 5 == 0 and epoch > 0) or (epoch % 5 == 0 and epoch > 200):
                torch.save(net.state_dict(), args.save_folder+args.version+'_'+args.dataset + '_epoches_'+
                           repr(epoch) + '.pth')
            epoch += 1

        load_t0 = time.time()
        if iteration in stepvalues:
            step_index += 1
        lr = adjust_learning_rate(optimizer, gamma, epoch, step_index, iteration, epoch_size)
        debug = False
        if iteration % 10 == 0:
            debug = True
        optimizer.zero_grad()
        loss_sum = 0
        for i in range(subdivisions):
            images, targets = next(batch_iterator)
            images.requires_grad_()
            if args.cuda:
                images = images.cuda()
                with torch.no_grad():
                    targets = [anno.cuda() for anno in targets]
            else:
                images = images
                with torch.no_grad():
                    targets = targets
            # forward
            resize_wh = images.size(3), images.size(2)
            out = net(images)
            loss = criterion(out, targets, resize_wh, debug) / subdivisions
            loss.backward()
            loss_sum += loss.item()

        if ave_loss < 0:
            ave_loss = loss_sum
        ave_loss = 0.1 * loss_sum + 0.9 * ave_loss
        optimizer.step()
        load_t1 = time.time()
        if iteration % 10 == 0:
            print('Epoch:' + repr(epoch) + ' || epochiter: ' + repr(iteration % epoch_size) + '/' + repr(epoch_size)
                  + '|| Totel iter ' +
                  repr(iteration) + ' Cur : %.4f  Ave : %.4f' % (loss_sum, ave_loss) + 
                ' iteration time: %.4f sec. ||' % (load_t1 - load_t0) + 'LR: %.5f' % (lr))

    torch.save(net.state_dict(), args.save_folder+args.version+'_'+args.dataset + "_final"+ '.pth')





