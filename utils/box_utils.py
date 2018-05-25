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


def get_rects(detection, input_dim, ori_dim, use_pad=False):
    if use_pad:
        scaling_factor = min(input_dim / ori_dim[0], input_dim / ori_dim[1])
        detection[:,[1,3]] -= (input_dim - scaling_factor * ori_dim[0]) / 2
        detection[:,[2,4]] -= (input_dim - scaling_factor * ori_dim[1]) / 2
        detection[:,1:5] /= scaling_factor
    else:
        detection[:,[1,3]] /= input_dim
        detection[:,[2,4]] /= input_dim
        detection[:, [1,3]] *= ori_dim[0]
        detection[:, [2,4]] *= ori_dim[1]
    for i in range(detection.shape[0]):
        detection[i, [1,3]] = torch.clamp(detection[i, [1,3]], 0.0, ori_dim[0])
        detection[i, [2,4]] = torch.clamp(detection[i, [2,4]], 0.0, ori_dim[1])
    return detection

def draw_rects(img, rects, classes):
    for rect in rects:
        left_top = tuple(rect[1:3].int())
        right_bottom = tuple(rect[3:5].int())
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
        cv2.putText(img, label, (left_top[0], left_top[1] - t_size[1] - 4), cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1)
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

def bbox_iou(box1, box2):
    """
    Returns the IoU of two bounding boxes 
    
    """
    #Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:,0], box1[:,1], box1[:,2], box1[:,3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:,0], box2[:,1], box2[:,2], box2[:,3]
    
    #get the corrdinates of the intersection rectangle
    inter_rect_x1 =  torch.max(b1_x1, b2_x1)
    inter_rect_y1 =  torch.max(b1_y1, b2_y1)
    inter_rect_x2 =  torch.min(b1_x2, b2_x2)
    inter_rect_y2 =  torch.min(b1_y2, b2_y2)
    
    #Intersection area
    if torch.cuda.is_available():
            inter_area = torch.max(inter_rect_x2 - inter_rect_x1 + 1,torch.zeros(inter_rect_x2.shape).cuda())*torch.max(inter_rect_y2 - inter_rect_y1 + 1, torch.zeros(inter_rect_x2.shape).cuda())
    else:
            inter_area = torch.max(inter_rect_x2 - inter_rect_x1 + 1,torch.zeros(inter_rect_x2.shape))*torch.max(inter_rect_y2 - inter_rect_y1 + 1, torch.zeros(inter_rect_x2.shape))
    
    #Union Area
    b1_area = (b1_x2 - b1_x1 + 1)*(b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1)*(b2_y2 - b2_y1 + 1)
    
    iou = inter_area / (b1_area + b2_area - inter_area)
    
    return iou

def predict_transform(prediction, inp_dim, anchors, num_classes, CUDA = True):
    batch_size = prediction.size(0)
    stride =  inp_dim // prediction.size(2)
    grid_size = inp_dim // stride
    bbox_attrs = 5 + num_classes
    num_anchors = len(anchors)
    
    anchors = [(a[0]/stride, a[1]/stride) for a in anchors]

    prediction = prediction.view(batch_size, bbox_attrs*num_anchors, grid_size*grid_size)
    prediction = prediction.transpose(1,2).contiguous()
    prediction = prediction.view(batch_size, grid_size*grid_size*num_anchors, bbox_attrs)

    #Sigmoid the  centre_X, centre_Y. and object confidencce
    prediction[:,:,0] = torch.sigmoid(prediction[:,:,0])
    prediction[:,:,1] = torch.sigmoid(prediction[:,:,1])
    prediction[:,:,4] = torch.sigmoid(prediction[:,:,4])
    
    #Add the center offsets
    grid_len = np.arange(grid_size)
    a,b = np.meshgrid(grid_len, grid_len)
    
    x_offset = torch.FloatTensor(a).view(-1,1)
    y_offset = torch.FloatTensor(b).view(-1,1)
    
    if CUDA:
        x_offset = x_offset.cuda()
        y_offset = y_offset.cuda()
    
    x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1,num_anchors).view(-1,2).unsqueeze(0)
    
    prediction[:,:,:2] += x_y_offset
      
    #log space transform height and the width
    anchors = torch.FloatTensor(anchors)
    
    if CUDA:
        anchors = anchors.cuda()
    
    anchors = anchors.repeat(grid_size*grid_size, 1).unsqueeze(0)
    prediction[:,:,2:4] = torch.exp(prediction[:,:,2:4])*anchors

    #Softmax the class scores
    prediction[:,:,5: 5 + num_classes] = torch.sigmoid((prediction[:,:, 5 : 5 + num_classes]))

    prediction[:,:,:4] *= stride 
    return prediction

def get_results(prediction, confidence, num_classes, nms_conf = 0.4):
    st = time.time()
    conf_mask = (prediction[:,:,4] > confidence).float().unsqueeze(2)
    prediction = prediction*conf_mask
    try:
        ind_nz = torch.nonzero(prediction[:,:,4]).transpose(0,1).contiguous()
    except:
        return 0
    box_a = prediction.new(prediction.shape)
    box_a[:,:,0] = (prediction[:,:,0] - prediction[:,:,2]/2)
    box_a[:,:,1] = (prediction[:,:,1] - prediction[:,:,3]/2)
    box_a[:,:,2] = (prediction[:,:,0] + prediction[:,:,2]/2) 
    box_a[:,:,3] = (prediction[:,:,1] + prediction[:,:,3]/2)
    prediction[:,:,:4] = box_a[:,:,:4]
    batch_size = prediction.size(0)
    output = prediction.new(1, prediction.size(2) + 1)
    write = False

    for ind in range(batch_size):
        st = time.time()
        image_pred = prediction[ind]
        #Get the class having maximum score, and the index of that class
        #Get rid of num_classes softmax scores 
        #Add the class index and the class score of class having maximum score
        max_conf, max_conf_score = torch.max(image_pred[:,5:5+ num_classes], 1)
        max_conf = max_conf.float().unsqueeze(1)
        max_conf_score = max_conf_score.float().unsqueeze(1)
        seq = (image_pred[:,:5], max_conf, max_conf_score)
        image_pred = torch.cat(seq, 1)
        #Get rid of the zero entries
        non_zero_ind =  (torch.nonzero(image_pred[:,4]))
 
        image_pred_ = image_pred[non_zero_ind.squeeze(),:].view(-1,7)
        #Get the various classes detected in the image
        try:
            img_classes = unique(image_pred_[:,-1])
        except:
             continue

        for cls in img_classes:
            cls_mask = image_pred_*(image_pred_[:,-1] == cls).float().unsqueeze(1)
            class_mask_ind = torch.nonzero(cls_mask[:,-2]).squeeze()
            
            image_pred_class = image_pred_[class_mask_ind].view(-1,7)
            keep = nms(image_pred_class.cpu().numpy(), nms_conf, force_cpu=True)
            image_pred_class = image_pred_class[keep]
            # print(image_pred_class)
            batch_ind = image_pred_class.new(image_pred_class.size(0), 1).fill_(ind)
            seq = batch_ind, image_pred_class
            if not write:
                output = torch.cat(seq,1)
                write = True
            else:
                out = torch.cat(seq, 1)
                output = torch.cat((output, out))
    
    return output


# def get_results(prediction, confidence, num_classes, nms = True, nms_conf = 0.4):
#     conf_mask = (prediction[:,:,4] > confidence).float().unsqueeze(2)
#     prediction = prediction*conf_mask
    

#     try:
#         ind_nz = torch.nonzero(prediction[:,:,4]).transpose(0,1).contiguous()
#     except:
#         return 0
    
    
#     box_a = prediction.new(prediction.shape)
#     box_a[:,:,0] = (prediction[:,:,0] - prediction[:,:,2]/2)
#     box_a[:,:,1] = (prediction[:,:,1] - prediction[:,:,3]/2)
#     box_a[:,:,2] = (prediction[:,:,0] + prediction[:,:,2]/2) 
#     box_a[:,:,3] = (prediction[:,:,1] + prediction[:,:,3]/2)
#     prediction[:,:,:4] = box_a[:,:,:4]
    

    
#     batch_size = prediction.size(0)
    
#     output = prediction.new(1, prediction.size(2) + 1)
#     write = False


#     for ind in range(batch_size):
#         #select the image from the batch
#         image_pred = prediction[ind]
        

        
#         #Get the class having maximum score, and the index of that class
#         #Get rid of num_classes softmax scores 
#         #Add the class index and the class score of class having maximum score
#         max_conf, max_conf_score = torch.max(image_pred[:,5:5+ num_classes], 1)
#         max_conf = max_conf.float().unsqueeze(1)
#         max_conf_score = max_conf_score.float().unsqueeze(1)
#         seq = (image_pred[:,:5], max_conf, max_conf_score)
#         image_pred = torch.cat(seq, 1)
        

        
#         #Get rid of the zero entries
#         non_zero_ind =  (torch.nonzero(image_pred[:,4]))

        
#         image_pred_ = image_pred[non_zero_ind.squeeze(),:].view(-1,7)
#         #Get the various classes detected in the image
#         try:
#             img_classes = unique(image_pred_[:,-1])
#         except:
#              continue

#         print(image_pred_.size())

#         for cls in img_classes:
#             #get the detections with one particular class
#             cls_mask = image_pred_*(image_pred_[:,-1] == cls).float().unsqueeze(1)
#             class_mask_ind = torch.nonzero(cls_mask[:,-2]).squeeze()
            

#             image_pred_class = image_pred_[class_mask_ind].view(-1,7)

#             print(image_pred_class)
        
        
#              #sort the detections such that the entry with the maximum objectness
#              #confidence is at the top
#             conf_sort_index = torch.sort(image_pred_class[:,4], descending = True )[1]
#             image_pred_class = image_pred_class[conf_sort_index]
#             print("sort ", image_pred_class)
#             idx = image_pred_class.size(0)
            
#             #if nms has to be done
#             if nms:
#                 #For each detection
#                 for i in range(idx):
#                     #Get the IOUs of all boxes that come after the one we are looking at 
#                     #in the loop
#                     try:
#                         ious = bbox_iou(image_pred_class[i].unsqueeze(0), image_pred_class[i+1:])
#                     except ValueError:
#                         break
        
#                     except IndexError:
#                         break
                    
#                     #Zero out all the detections that have IoU > treshhold
#                     iou_mask = (ious < nms_conf).float().unsqueeze(1)
#                     image_pred_class[i+1:] *= iou_mask       
                    
#                     #Remove the non-zero entries
#                     non_zero_ind = torch.nonzero(image_pred_class[:,4]).squeeze()
#                     image_pred_class = image_pred_class[non_zero_ind].view(-1,7)
                    
                    

#             #Concatenate the batch_id of the image to the detection
#             #this helps us identify which image does the detection correspond to 
#             #We use a linear straucture to hold ALL the detections from the batch
#             #the batch_dim is flattened
#             #batch is identified by extra batch column
            
            
#             batch_ind = image_pred_class.new(image_pred_class.size(0), 1).fill_(ind)
#             seq = batch_ind, image_pred_class
#             if not write:
#                 output = torch.cat(seq,1)
#                 write = True
#             else:
#                 out = torch.cat(seq,1)
#                 output = torch.cat((output,out))
    
#     return output
