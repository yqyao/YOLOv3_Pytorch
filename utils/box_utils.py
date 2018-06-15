from __future__ import division

import torch 
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable
import numpy as np
import math
import cv2
import time
from utils.nms_wrapper import nms


def get_rects(detection, input_wh, ori_wh, use_pad=False):
    if len(detection) > 0:
        if use_pad:
            scaling_factor = min(input_wh[0] / ori_wh[0], input_wh[1] / ori_wh[1])
            detection[:,[1,3]] -= (input_wh[0] - scaling_factor * ori_wh[0]) / 2
            detection[:,[2,4]] -= (input_wh[1] - scaling_factor * ori_wh[1]) / 2
            detection[:,1:5] /= scaling_factor
        else:
            detection[:,[1,3]] /= input_wh[0]
            detection[:,[2,4]] /= input_wh[1]
            detection[:, [1,3]] *= ori_wh[0]
            detection[:, [2,4]] *= ori_wh[1]
        for i in range(detection.shape[0]):
            detection[i, [1,3]] = torch.clamp(detection[i, [1,3]], 0.0, ori_wh[0])
            detection[i, [2,4]] = torch.clamp(detection[i, [2,4]], 0.0, ori_wh[1])
    return detection

def draw_rects(img, rects, classes):
    print(rects)
    for rect in rects:
        if rect[5] > 0.1:
            left_top = (int(rect[0]), int(rect[1]))
            right_bottom = (int(rect[2]), int(rect[3]))
            score = round(rect[4], 3)
            cls_id = int(rect[-1])
            label = "{0}".format(classes[cls_id])
            class_len = len(classes)
            offset = cls_id * 123457 % class_len
            red   = get_color(2, offset, class_len)
            green = get_color(1, offset, class_len)
            blue  = get_color(0, offset, class_len)        
            color = (blue, green, red)
            cv2.rectangle(img, left_top, right_bottom, color, 2)
            t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
            right_bottom = left_top[0] + t_size[0] + 3, left_top[1] - t_size[1] - 4
            cv2.rectangle(img, left_top, right_bottom, color, -1)
            cv2.putText(img, str(label)+str(score), (left_top[0], left_top[1] - t_size[1] - 4), cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1)
    return img        

def get_color(c, x, max_val):
    colors = torch.FloatTensor([[1,0,1],[0,0,1],[0,1,1],[0,1,0],[1,1,0],[1,0,0]])
    ratio = float(x) / max_val * 5
    i = int(math.floor(ratio))
    j = int(math.ceil(ratio))
    ratio = ratio - i
    r = (1-ratio) * colors[i][c] + ratio * colors[j][c]
    return int(r*255)


def unique(tensor):
    tensor_np = tensor.cpu().numpy()
    unique_np = np.unique(tensor_np)
    unique_tensor = torch.from_numpy(unique_np)
    
    tensor_res = tensor.new(unique_tensor.shape)
    tensor_res.copy_(unique_tensor)
    return tensor_res

def point_form(boxes):
    """ Convert prior_boxes to (xmin, ymin, xmax, ymax)
    representation for comparison to point form ground truth data.
    Args:
        boxes: (tensor) center-size default boxes from priorbox layers.
    Return:
        boxes: (tensor) Converted xmin, ymin, xmax, ymax form of boxes.
    """
    return torch.cat((boxes[:, :2] - boxes[:, 2:]/2,     # xmin, ymin
                     boxes[:, :2] + boxes[:, 2:]/2), 1)  # xmax, ymax

def center_size(boxes):
    """ Convert prior_boxes to (cx, cy, w, h)
    representation for comparison to center-size form ground truth data.
    Args:
        boxes: (tensor) point_form boxes
    Return:
        boxes: (tensor) Converted xmin, ymin, xmax, ymax form of boxes.
    """
    return torch.cat([(boxes[:, 2:] + boxes[:, :2])/2, boxes[:, 2:] - boxes[:, :2]], 1)  # w, h


def intersect(box_a, box_b):
    """ We resize both tensors to [A,B,2] without new malloc:
    [A,2] -> [A,1,2] -> [A,B,2]
    [B,2] -> [1,B,2] -> [A,B,2]
    Then we compute the area of intersect between box_a and box_b.
    Args:
      box_a: (tensor) bounding boxes, Shape: [A,4].
      box_b: (tensor) bounding boxes, Shape: [B,4].
    Return:
      (tensor) intersection area, Shape: [A,B].
    """
    # print(box_a)
    A = box_a.size(0)
    B = box_b.size(0)
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                       box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                       box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, :, 0] * inter[:, :, 1]


def jaccard(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.  Here we operate on
    ground truth boxes and default boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]
        box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_priors,4]
    Return:
        jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    area_b = ((box_b[:, 2]-box_b[:, 0]) *
              (box_b[:, 3]-box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]

def trans_anchors(anchors):
    new_anchors = torch.zeros((anchors.size(0), 4))
    new_anchors[:, :2] += 2000
    new_anchors[:, 2:] = anchors[:,]
    return point_form(new_anchors)

def trans_truths(truths):
    new_truths = torch.zeros((truths.size(0), 4))
    new_truths[:, :2] += 2000
    new_truths[:, 2:] = truths[:, 2:4]
    return point_form(new_truths)

def int_index(anchors_mask, val):
    for i in range(len(anchors_mask)):
        if val == anchors_mask[i]:
            return i
    return -1

def encode_targets_all(input_wh, truths, labels, best_anchor_idx, anchors, feature_dim, num_pred, back_mask):
    scale = torch.ones(num_pred).cuda()
    encode_truths = torch.zeros((num_pred, 6)).cuda()
    fore_mask = torch.zeros(num_pred).cuda()
    # l_dim, m_dim, h_dim = feature_dim
    l_grid_wh, m_grid_wh, h_grid_wh = feature_dim
    for i in range(best_anchor_idx.size(0)):
        index = 0
        grid_wh = (0, 0)
        # mask [0, 1, 2]
        if best_anchor_idx[i].item() < 2.1:
            grid_wh = l_grid_wh
            index_begin = 0
        # mask [3, 4, 5]
        elif best_anchor_idx[i].item() < 5.1:
            grid_wh = m_grid_wh
            index_begin = l_grid_wh[0] * l_grid_wh[1] * 3
        # mask [6, 7, 8]
        else:
            grid_wh = h_grid_wh
            index_begin = (l_grid_wh[0]*l_grid_wh[1] + m_grid_wh[0]*m_grid_wh[1]) * 3
        x = (truths[i][0] / input_wh[0]) * grid_wh[0]  
        y = (truths[i][1] / input_wh[1]) * grid_wh[1]
        floor_x, floor_y = math.floor(x), math.floor(y)
        anchor_idx = best_anchor_idx[i].int().item() % 3
        index = index_begin + floor_y * grid_wh[0] * 3 + floor_x * 3 + anchor_idx

        scale[index] = scale[index] + 1. - (truths[i][2] / input_wh[0]) * (truths[i][3] / input_wh[1])

        # encode targets x, y, w, h, objectness, class
        truths[i][0] = x - floor_x
        truths[i][1] = y - floor_y
        truths[i][2] = torch.log(truths[i][2] / anchors[best_anchor_idx[i]][0] + 1e-8)
        truths[i][3] = torch.log(truths[i][3] / anchors[best_anchor_idx[i]][1] + 1e-8)
        encode_truths[index, :4] = truths[i]
        encode_truths[index, 4] = 1.
        encode_truths[index, 5] = labels[i].int().item()

        # set foreground mask to 1 and background mask to 0, because  pred should have unique target
        fore_mask[index] = 1.
        back_mask[index] = 0

    return encode_truths, fore_mask > 0, scale, back_mask

def encode_targets_single(input_wh, truths, labels, best_anchor_idx, anchors, anchors_mask, back_mask, grid_wh):
    grid_w, grid_h = grid_wh[0], grid_wh[1]
    num_pred = grid_w * grid_h * len(anchors_mask)
    scale = torch.ones(num_pred).cuda()
    encode_truths = torch.zeros((num_pred, 6)).cuda()
    fore_mask = torch.zeros(num_pred).cuda()

    for i in range(best_anchor_idx.size(0)):
        mask_n = int_index(anchors_mask, best_anchor_idx[i])
        if mask_n < 0:
            continue
        x = (truths[i][0] / input_wh[0]) * grid_wh[0]  
        y = (truths[i][1] / input_wh[1]) * grid_wh[1]
        floor_x, floor_y = math.floor(x), math.floor(y)
        index = floor_y * grid_wh[0] * 3 + floor_x * 3 + mask_n
        scale[index] = scale[index] + 1. - (truths[i][2] / input_wh[0]) * (truths[i][3] / input_wh[1])
        truths[i][0] = x - floor_x
        truths[i][1] = y - floor_y
        truths[i][2] = torch.log(truths[i][2] / anchors[best_anchor_idx[i]][0] + 1e-8)
        truths[i][3] = torch.log(truths[i][3] / anchors[best_anchor_idx[i]][1] + 1e-8)
        encode_truths[index, :4] = truths[i]
        encode_truths[index, 4] = 1.
        encode_truths[index, 5] = labels[i].int().item()
        fore_mask[index] = 1.
        back_mask[index] = 0

    return encode_truths, fore_mask > 0, scale, back_mask

def targets_match_single(input_wh, threshold, targets, pred, anchors, anchors_mask, pred_t, scale_t, fore_mask_t, back_mask_t, grid_wh, idx, cuda=True):
    loc_truths = targets[:, :4].data
    labels = targets[:,-1].data
    overlaps = jaccard(
        loc_truths, 
        point_form(pred))
    # (Bipartite Matching)
    # [1,num_objects] best prior for each ground truth
    # best_prior_overlap, best_prior_idx = overlaps.max(1, keepdim=True)
    # [1,num_priors] best ground truth for each prior

    best_truth_overlap, best_truth_idx = overlaps.max(0, keepdim=True)
    best_truth_idx.squeeze_(0)
    best_truth_overlap.squeeze_(0)
    back_mask = (best_truth_overlap - threshold) < 0

    anchors = torch.FloatTensor(anchors)    
    if cuda:
        anchors = anchors.cuda()

    center_truths = center_size(loc_truths)

    # convert anchor and truths to calculate iou
    new_anchors = trans_anchors(anchors)
    new_truths = trans_truths(center_truths)
    overlaps_ = jaccard(
        new_truths,
        new_anchors)
    best_anchor_overlap, best_anchor_idx = overlaps_.max(1, keepdim=True)
    best_anchor_idx.squeeze_(1)
    best_anchor_overlap.squeeze_(1)

    encode_truths, fore_mask, scale, back_mask = encode_targets_single(input_wh, center_truths, labels, best_anchor_idx, anchors, anchors_mask, back_mask, grid_wh)

    pred_t[idx] = encode_truths
    scale_t[idx] = scale
    fore_mask_t[idx] = fore_mask
    back_mask_t[idx] = back_mask       

def targets_match_all(input_wh, threshold, targets, pred, anchors, feature_dim, pred_t, scale_t, fore_mask_t, back_mask_t, num_pred, idx, cuda=True):
    loc_truths = targets[:, :4].data
    labels = targets[:,-1].data
    overlaps = jaccard(
        loc_truths, 
        point_form(pred))
    # (Bipartite Matching)
    # [1,num_objects] best prior for each ground truth
    # best_prior_overlap, best_prior_idx = overlaps.max(1, keepdim=True)
    # [1,num_priors] best ground truth for each prior

    best_truth_overlap, best_truth_idx = overlaps.max(0, keepdim=True)
    best_truth_idx.squeeze_(0)
    best_truth_overlap.squeeze_(0)
    back_mask = (best_truth_overlap - threshold) < 0

    anchors = torch.FloatTensor(anchors)    
    if cuda:
        anchors = anchors.cuda()

    center_truths = center_size(loc_truths)
    new_anchors = trans_anchors(anchors)
    new_truths = trans_truths(center_truths)
    overlaps_ = jaccard(
        new_truths,
        new_anchors)
    best_anchor_overlap, best_anchor_idx = overlaps_.max(1, keepdim=True)
    best_anchor_idx.squeeze_(1)
    best_anchor_overlap.squeeze_(1)

    encode_truths, fore_mask, scale, back_mask = encode_targets_all(input_wh, center_truths, labels, best_anchor_idx, anchors, feature_dim, num_pred, back_mask)

    pred_t[idx] = encode_truths
    scale_t[idx] = scale
    fore_mask_t[idx] = fore_mask
    back_mask_t[idx] = back_mask

def decode(prediction, input_wh, anchors, num_classes, stride_wh, cuda=True):
    grid_wh = (input_wh[0] // stride_wh[0], input_wh[1] // stride_wh[1])
    grid_w = np.arange(grid_wh[0])
    grid_h = np.arange(grid_wh[1])
    a,b = np.meshgrid(grid_w, grid_h)    

    num_anchors = len(anchors)
    x_offset = torch.FloatTensor(a).view(-1,1)
    y_offset = torch.FloatTensor(b).view(-1,1)
    anchors = [(a[0]/stride_wh[0], a[1]/stride_wh[1]) for a in anchors]
    if cuda:
        x_offset = x_offset.cuda()
        y_offset = y_offset.cuda()
    x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1, num_anchors).view(-1,2).unsqueeze(0)
    prediction[:,:,:2] += x_y_offset
    anchors = torch.FloatTensor(anchors)    
    if cuda:
        anchors = anchors.cuda()
    anchors = anchors.repeat(grid_wh[0]*grid_wh[1], 1).unsqueeze(0)
    prediction[:,:,2:4] = torch.exp(prediction[:,:,2:4]) * anchors
    prediction[:,:,0] *= stride_wh[0]
    prediction[:,:,2] *= stride_wh[0]
    prediction[:,:,1] *= stride_wh[1]
    prediction[:,:,3] *= stride_wh[1]
    return prediction

def permute_sigmoid(x, input_wh, num_anchors, num_classes):
    batch_size = x.size(0)
    grid_wh = (x.size(3), x.size(2))
    input_w, input_h = input_wh
    stride_wh = (input_w // grid_wh[0], input_h // grid_wh[1])
    bbox_attrs = 5 + num_classes
    x = x.view(batch_size, bbox_attrs*num_anchors, grid_wh[0] * grid_wh[1])
    x = x.transpose(1,2).contiguous()
    x = x.view(batch_size, grid_wh[0]*grid_wh[1]*num_anchors, bbox_attrs)
    x[:,:,0] = torch.sigmoid(x[:,:,0])
    x[:,:,1] = torch.sigmoid(x[:,:,1])             
    x[:,:, 4 : bbox_attrs] = torch.sigmoid((x[:,:, 4 : bbox_attrs]))
    return x, stride_wh

def detection_postprecess(detection, iou_thresh, num_classes, input_wh, ori_wh, use_pad=False, nms_conf=0.4):
    assert detection.size(0) == 1, "only support batch_size == 1"
    conf_mask = (detection[:,:,4] > iou_thresh).float().unsqueeze(2)
    detection = detection * conf_mask
    try:
        ind_nz = torch.nonzero(detection[:,:,4]).transpose(0,1).contiguous()
    except:
        print("detect no results")
        return np.empty([0, 5], dtype=np.float32)
    bbox_pred = point_form(detection[:, :, :4].view(-1, 4))
    conf_pred = detection[:, :, 4].view(-1, 1)
    cls_pred = detection[:, :, 5:].view(-1, num_classes)

    max_conf, max_conf_idx = torch.max(cls_pred, 1) 

    max_conf = max_conf.float().unsqueeze(1)
    max_conf_idx = max_conf_idx.float().unsqueeze(1)

    # score = (conf_pred * max_conf).view(-1, 1)
    score = conf_pred
    image_pred = torch.cat((bbox_pred, score, max_conf, max_conf_idx), 1)

    non_zero_ind =  (torch.nonzero(image_pred[:,4]))
    image_pred_ = image_pred[non_zero_ind.squeeze(),:].view(-1, 7)
    try:
        img_classes = unique(image_pred_[:,-1])
    except:
        print("no class find")
        return np.empty([0, 7], dtype=np.float32)
    flag = False
    out_out = None
    for cls in img_classes:
        cls_mask = image_pred_*(image_pred_[:,-1] == cls).float().unsqueeze(1)
        class_mask_ind = torch.nonzero(cls_mask[:,-2]).squeeze()
        
        image_pred_class = image_pred_[class_mask_ind].view(-1,7)
        keep = nms(image_pred_class.cpu().numpy(), nms_conf, force_cpu=True)
        image_pred_class = image_pred_class[keep]
        if not flag:
            out_put = image_pred_class
            flag = True
        else:
            out_put = torch.cat((out_put, image_pred_class), 0)


    image_pred_class = out_put
    if use_pad:
        scaling_factor = min(input_wh[0] / ori_wh[0], input_wh[1] / ori_wh[1])
        image_pred_class[:,[0,2]] -= (input_wh[0] - scaling_factor * ori_wh[0]) / 2
        image_pred_class[:,[1,3]] -= (input_wh[1] - scaling_factor * ori_wh[1]) / 2
        image_pred_class[:,:4] /= scaling_factor
    else:
        image_pred_class[:,[0,2]] /= input_wh[0]
        image_pred_class[:,[1,3]] /= input_wh[1]
        image_pred_class[:, [0,2]] *= ori_wh[0]
        image_pred_class[:, [1,3]] *= ori_wh[1]

    for i in range(image_pred_class.shape[0]):
        image_pred_class[i, [0,2]] = torch.clamp(image_pred_class[i, [0,2]], 0.0, ori_wh[0])
        image_pred_class[i, [1,3]] = torch.clamp(image_pred_class[i, [1,3]], 0.0, ori_wh[1])
    return image_pred_class.cpu().numpy()

