"""Data augmentation functionality. Passed as callable transformations to
Dataset classes.

The data augmentation procedures were interpreted from @weiliu89's SSD paper
http://arxiv.org/abs/1512.02325

TODO: implement data_augment for training

Ellis Brown, Max deGroot
"""

import torch
from torchvision import transforms
import cv2
import numpy as np
import random
import math


def matrix_iou(a,b):
    """
    return iou of a and b, numpy version for data augenmentation
    """
    lt = np.maximum(a[:, np.newaxis, :2], b[:, :2])
    rb = np.minimum(a[:, np.newaxis, 2:], b[:, 2:])

    area_i = np.prod(rb - lt, axis=2) * (lt < rb).all(axis=2)
    area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
    area_b = np.prod(b[:, 2:] - b[:, :2], axis=1)
    return area_i / (area_a[:, np.newaxis] + area_b - area_i)

def _crop(image, boxes, labels):
    height, width, _ = image.shape

    if len(boxes)== 0:
        return image, boxes, labels

    while True:
        mode = random.choice((
            None,
            (0.3, None),
            (0.5, None),
            (0.7, None),
            (0.9, None),
            (None, None),
        ))

        if mode is None:
            return image, boxes, labels

        min_iou, max_iou = mode
        if min_iou is None:
            min_iou = float('-inf')
        if max_iou is None:
            max_iou = float('inf')

        for _ in range(50):
            scale = random.uniform(0.3,1.)
            min_ratio = max(0.5, scale*scale)
            max_ratio = min(2, 1. / scale / scale)
            ratio = math.sqrt(random.uniform(min_ratio, max_ratio))
            w = int(scale * ratio * width)
            h = int((scale / ratio) * height)


            l = random.randrange(width - w)
            t = random.randrange(height - h)
            roi = np.array((l, t, l + w, t + h))

            iou = matrix_iou(boxes, roi[np.newaxis])
            
            if not (min_iou <= iou.min() and iou.max() <= max_iou):
                continue

            image_t = image[roi[1]:roi[3], roi[0]:roi[2]]

            centers = (boxes[:, :2] + boxes[:, 2:]) / 2
            mask = np.logical_and(roi[:2] < centers, centers < roi[2:]) \
                     .all(axis=1)
            boxes_t = boxes[mask].copy()
            labels_t = labels[mask].copy()
            if len(boxes_t) == 0:
                continue

            boxes_t[:, :2] = np.maximum(boxes_t[:, :2], roi[:2])
            boxes_t[:, :2] -= roi[:2]
            boxes_t[:, 2:] = np.minimum(boxes_t[:, 2:], roi[2:])
            boxes_t[:, 2:] -= roi[:2]

            return image_t, boxes_t,labels_t


def _distort(image):
    def _convert(image, alpha=1, beta=0):
        tmp = image.astype(float) * alpha + beta
        tmp[tmp < 0] = 0
        tmp[tmp > 255] = 255
        image[:] = tmp

    image = image.copy()

    if random.randrange(2):
        _convert(image, beta=random.uniform(-32, 32))

    if random.randrange(2):
        _convert(image, alpha=random.uniform(0.5, 1.5))

    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    if random.randrange(2):
        tmp = image[:, :, 0].astype(int) + random.randint(-18, 18)
        tmp %= 180
        image[:, :, 0] = tmp

    if random.randrange(2):
        _convert(image[:, :, 1], alpha=random.uniform(0.5, 1.5))

    image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

    return image


def _expand(image, boxes,fill, p):
    if random.random() > p:
        return image, boxes

    height, width, depth = image.shape
    for _ in range(50):
        scale = random.uniform(1,4)

        min_ratio = max(0.5, 1./scale/scale)
        max_ratio = min(2, scale*scale)
        ratio = math.sqrt(random.uniform(min_ratio, max_ratio))
        ws = scale * ratio
        hs = scale / ratio
        if ws < 1 or hs < 1:
            continue
        w = int(ws * width)
        h = int(hs * height)

        left = random.randint(0, w - width)
        top = random.randint(0, h - height)

        boxes_t = boxes.copy()
        boxes_t[:, :2] += (left, top)
        boxes_t[:, 2:] += (left, top)


        expand_image = np.empty(
            (h, w, depth),
            dtype=image.dtype)
        expand_image[:, :] = fill
        expand_image[top:top + height, left:left + width] = image
        image = expand_image

        return image, boxes_t


def _mirror(image, boxes):
    _, width, _ = image.shape
    if random.randrange(2):
        image = image[:, ::-1]
        boxes = boxes.copy()
        boxes[:, 0::2] = width - boxes[:, 2::-2]
    return image, boxes

def rand(a=0, b=1):
    return np.random.rand()*(b-a) + a

# def random_letterbox_image(img, resize_wh, boxes, jitter=0.3):
#     '''resize image with unchanged aspect ratio using padding'''
#     img_w, img_h = img.shape[1], img.shape[0]
#     w, h = resize_wh
#     new_ar = w / h * rand(1-jitter, 1+jitter)/rand(1-jitter, 1+jitter)
#     scale = rand(.25, 2)
#     if new_ar < 1:
#         nh = int(scale * h)
#         nw = int(nh * new_ar)
#     else:
#         nw = int(scale * w)
#         nh = int(nw / new_ar)
#     resized_image = cv2.resize(img, (nw, nh), interpolation = cv2.INTER_CUBIC)

#     dx = int(rand(0, w - nw))
#     dy = int(rand(0, h - nh))

#     if (w - nw) < 0:
#         cxmin = 0
#         xmin = nw - w + dx
#         xmax = nw + dx
#         cxmax = xmax - xmin
#     else:
#         cxmin = dx
#         xmin = 0
#         xmax = nw
#         cxmax = nw + dx
#     if (h - nh) < 0:
#         cymin = 0
#         ymin = nh - h + dy
#         ymax = nh + dy
#         cymax = ymax - ymin
#     else:
#         cymin = dy
#         ymin = 0
#         ymax = nh
#         cymax = nh + dy

#     resized_image = resized_image[ymin:ymax,xmin:xmax,:]

#     boxes[:, 0::2] = (boxes[:, 0::2] * nw / img_w  + dx) / w
#     boxes[:, 1::2] = (boxes[:, 1::2] * nh / img_h + dy ) / h
#     # clamp boxes
#     boxes[:, 0:2][boxes[:, 0:2]<=0] = 0
#     boxes[:, 2:][boxes[:, 2:]>=1] = 0.9999

#     canvas = np.full((resize_wh[1], resize_wh[0], 3), 128)
#     canvas[cymin:cymax, cxmin:cxmax,  :] = resized_image

#     img_ = canvas[:,:,::-1].transpose((2,0,1)).copy()
#     img_ = torch.from_numpy(img_).float().div(255.0)
#     return img_, boxes

def letterbox_image(img, resize_wh, boxes):
    '''resize image with unchanged aspect ratio using padding'''
    img_w, img_h = img.shape[1], img.shape[0]
    w, h = resize_wh
    new_w = int(img_w * min(w/img_w, h/img_h))
    new_h = int(img_h * min(w/img_w, h/img_h))

    if len(boxes) > 0:
        boxes = boxes.copy()
        dim_diff = np.abs(img_w - img_h)
        max_size = max(img_w, img_h)
        if img_w > img_h:
            boxes[:, 1::2] += dim_diff // 2
        else:
            boxes[:, 0::2] += dim_diff // 2
        boxes[:, 0::2] /= max_size
        boxes[:, 1::2] /= max_size
    resized_image = cv2.resize(img, (new_w, new_h), interpolation = cv2.INTER_CUBIC)
    canvas = np.full((resize_wh[0], resize_wh[1], 3), 128)
    canvas[(h-new_h)//2:(h-new_h)//2 + new_h,(w-new_w)//2:(w-new_w)//2 + new_w,  :] = resized_image
    img_ = canvas[:,:,::-1].transpose((2,0,1)).copy()
    img_ = torch.from_numpy(img_).float().div(255.0)
    
    return img_, boxes


class preproc(object):

    def __init__(self):
        self.means = [128, 128, 128]
        self.p = 0.5

    def __call__(self, image, targets, resize_wh, use_pad=True):
        boxes = targets[:, :-1].copy()
        labels = targets[:, -1].copy()
        height, width, _ = image.shape
        if len(boxes) == 0:
            targets = np.zeros((1,5))
            image, _ = letterbox_image(image, resize_wh, boxes)
            return image, targets
        image_o = image.copy()
        targets_o = targets.copy()
        image_t, boxes, labels = _crop(image, boxes, labels)
        image_t = _distort(image_t)
        image_t, boxes = _expand(image_t, boxes, self.means, self.p)
        image_t, boxes = _mirror(image_t, boxes)
        image_t, boxes = letterbox_image(image_t, resize_wh, boxes)

        boxes = boxes.copy()
        b_w = (boxes[:, 2] - boxes[:, 0])*1.
        b_h = (boxes[:, 3] - boxes[:, 1])*1.
        mask_b= np.minimum(b_w, b_h) > 0.01
        boxes_t = boxes[mask_b]
        labels_t = labels[mask_b].copy()

        if len(boxes_t) == 0:
            boxes_t = targets_o[:, :4].copy()
            labels_t = targets_o[:, -1].copy()
            image_t, boxes_t = letterbox_image(image_o, resize_wh, boxes_t)

        boxes_t[:, 0::2] *= resize_wh[0]
        boxes_t[:, 1::2] *= resize_wh[1]

        labels_t = np.expand_dims(labels_t, 1)
        targets_t = np.hstack((boxes_t, labels_t))

        return image_t, targets_t


