from __future__ import division

import torch 
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable
import numpy as np
import cv2 


def letterbox_image(img, resize_wh):
    '''resize image with unchanged aspect ratio using padding'''
    img_w, img_h = img.shape[1], img.shape[0]
    w, h = resize_wh
    new_w = int(img_w * min(w/img_w, h/img_h))
    new_h = int(img_h * min(w/img_w, h/img_h))
    resized_image = cv2.resize(img, (new_w, new_h), interpolation = cv2.INTER_CUBIC)
    
    canvas = np.full((resize_wh[1], resize_wh[0], 3), 128)

    canvas[(h-new_h)//2:(h-new_h)//2 + new_h,(w-new_w)//2:(w-new_w)//2 + new_w,  :] = resized_image
    
    return canvas

def preproc_for_test(img, resize_wh, use_pad=False):
    if not use_pad:
        img = cv2.resize(img, resize_wh) 
    else:
        img = letterbox_image(img, resize_wh)        
    img_ = img[:,:,::-1].transpose((2,0,1)).copy()
    img_ = torch.from_numpy(img_).float().div(255.0).unsqueeze(0)
    return img_

