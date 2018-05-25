from __future__ import division

import torch 
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable
import numpy as np
import cv2 

def preproc_no_padding(img, insize):
    ori_im = img
    ori_dim = img.shape[1], img.shape[0]
    img = cv2.resize(img, (insize[1], insize[0])) 
    img_ =  img[:,:,::-1].transpose((2,0,1))
    img_ = img_[np.newaxis,:,:,:]/255.0
    img_ = torch.from_numpy(img_).float()
    # img_ = Variable(img_)
    return img_, ori_im, ori_dim

def letterbox_image(img, inp_dim):
    '''resize image with unchanged aspect ratio using padding'''
    img_w, img_h = img.shape[1], img.shape[0]
    w, h = inp_dim
    new_w = int(img_w * min(w/img_w, h/img_h))
    new_h = int(img_h * min(w/img_w, h/img_h))
    resized_image = cv2.resize(img, (new_w, new_h), interpolation = cv2.INTER_CUBIC)
    
    canvas = np.full((inp_dim[1], inp_dim[0], 3), 128)

    canvas[(h-new_h)//2:(h-new_h)//2 + new_h,(w-new_w)//2:(w-new_w)//2 + new_w,  :] = resized_image
    
    return canvas

def preproc_with_padding(img, insize):
    ori_im = img
    ori_dim = orig_im.shape[1], orig_im.shape[0]
    img = (letterbox_image(ori_im, (insize[1], insize[0])))
    img_ = img[:,:,::-1].transpose((2,0,1)).copy()
    img_ = torch.from_numpy(img_).float().div(255.0).unsqueeze(0)
    return img_, ori_im, ori_dim


def preproc_img(img, insize, use_pad=False):
    ori_im = img
    ori_dim = img.shape[1], img.shape[0]
    if not use_pad:
        img = cv2.resize(img, (insize[1], insize[0])) 
    else:
        img = letterbox_image(ori_im, (insize[1], insize[0]))        
    img_ = img[:,:,::-1].transpose((2,0,1)).copy()
    img_ = torch.from_numpy(img_).float().div(255.0).unsqueeze(0)
    return img_, ori_im, ori_dim
