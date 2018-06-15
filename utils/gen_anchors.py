import random
import argparse
import numpy as np
import os
import sys
if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET
import pickle

import json

def parse_voc_annotation(ann_dir, img_dir, train_val_list, cache_name, labels=[]):
    if os.path.exists(cache_name):
        with open(cache_name, 'rb') as handle:
            cache = pickle.load(handle)
        all_insts, seen_labels = cache['all_insts'], cache['seen_labels']
    else:
        all_insts = []
        seen_labels = {}
        
        for ann in sorted(train_val_list):
            img = {'object':[]}

            try:
                tree = ET.parse(os.path.join(ann_dir, ann + ".xml"))
            except Exception as e:
                print(e)
                print('Ignore this bad annotation: ' + ann_dir + ann)
                continue
            
            for elem in tree.iter():
                if 'filename' in elem.tag:
                    img['filename'] = os.path.join(img_dir, elem.text + ".jpg")
                if 'width' in elem.tag:
                    img['width'] = int(elem.text)
                if 'height' in elem.tag:
                    img['height'] = int(elem.text)
                if 'object' in elem.tag or 'part' in elem.tag:
                    obj = {}
                    
                    for attr in list(elem):
                        if 'name' in attr.tag:
                            obj['name'] = attr.text

                            if obj['name'] in seen_labels:
                                seen_labels[obj['name']] += 1
                            else:
                                seen_labels[obj['name']] = 1
                            
                            if len(labels) > 0 and obj['name'] not in labels:
                                break
                            else:
                                img['object'] += [obj]
                                
                        if 'bndbox' in attr.tag:
                            for dim in list(attr):
                                if 'xmin' in dim.tag:
                                    obj['xmin'] = int(round(float(dim.text)))
                                if 'ymin' in dim.tag:
                                    obj['ymin'] = int(round(float(dim.text)))
                                if 'xmax' in dim.tag:
                                    obj['xmax'] = int(round(float(dim.text)))
                                if 'ymax' in dim.tag:
                                    obj['ymax'] = int(round(float(dim.text)))

            if len(img['object']) > 0:
                all_insts += [img]

        cache = {'all_insts': all_insts, 'seen_labels': seen_labels}
        with open(cache_name, 'wb') as handle:
            pickle.dump(cache, handle, protocol=pickle.HIGHEST_PROTOCOL)    
                        
    return all_insts, seen_labels

def IOU(ann, centroids):
    w, h = ann
    similarities = []

    for centroid in centroids:
        c_w, c_h = centroid

        if c_w >= w and c_h >= h:
            similarity = w*h/(c_w*c_h)
        elif c_w >= w and c_h <= h:
            similarity = w*c_h/(w*h + (c_w-w)*c_h)
        elif c_w <= w and c_h >= h:
            similarity = c_w*h/(w*h + c_w*(c_h-h))
        else: #means both w,h are bigger than c_w and c_h respectively
            similarity = (c_w*c_h)/(w*h)
        similarities.append(similarity) # will become (k,) shape

    return np.array(similarities)

def avg_IOU(anns, centroids):
    n,d = anns.shape
    sum = 0.

    for i in range(anns.shape[0]):
        sum+= max(IOU(anns[i], centroids))

    return sum/n

def print_anchors(centroids):
    out_string = ''

    anchors = centroids.copy()

    widths = anchors[:, 0]
    sorted_indices = np.argsort(widths)

    r = "anchors: ["
    for i in sorted_indices:
        out_string += str(int(anchors[i,0]*416)) + ',' + str(int(anchors[i,1]*416)) + ', '
            
    print(out_string[:-2])

def run_kmeans(ann_dims, anchor_num):
    ann_num = ann_dims.shape[0]
    iterations = 0
    prev_assignments = np.ones(ann_num)*(-1)
    iteration = 0
    old_distances = np.zeros((ann_num, anchor_num))

    indices = [random.randrange(ann_dims.shape[0]) for i in range(anchor_num)]
    centroids = ann_dims[indices]
    anchor_dim = ann_dims.shape[1]

    while True:
        distances = []
        iteration += 1
        for i in range(ann_num):
            d = 1 - IOU(ann_dims[i], centroids)
            distances.append(d)
        distances = np.array(distances) # distances.shape = (ann_num, anchor_num)

        print("iteration {}: dists = {}".format(iteration, np.sum(np.abs(old_distances-distances))))

        #assign samples to centroids
        assignments = np.argmin(distances,axis=1)

        if (assignments == prev_assignments).all() :
            return centroids

        #calculate new centroids
        centroid_sums=np.zeros((anchor_num, anchor_dim), np.float)
        for i in range(ann_num):
            centroid_sums[assignments[i]]+=ann_dims[i]
        for j in range(anchor_num):
            centroids[j] = centroid_sums[j]/(np.sum(assignments==j) + 1e-6)

        prev_assignments = assignments.copy()
        old_distances = distances.copy()

def _main_(argv):
    num_anchors = args.anchors
    train_annot_folder = "/localSSD/yyq/VOCdevkit0712/VOC0712/Annotations/"
    train_image_folder = "/localSSD/yyq/VOCdevkit0712/VOC0712/JPEGImages/"
    train_val_txt = "/localSSD/yyq/VOCdevkit0712/VOC0712/ImageSets/Main/0712_trainval_test.txt"
    with open(train_val_txt, "r") as f:
        train_val_list = [i.strip() for i in f.readlines()]
    cache_name = "voc_train.pkl"
    labels = ["aeroplane", "bicycle", "bird", "boat",
             "bottle", "bus", "car", "cat", "chair", 
             "cow", "diningtable", "dog", "horse", 
             "motorbike", "person", "pottedplant",
            "sheep", "sofa", "train", "tvmonitor"]

    train_imgs, train_labels = parse_voc_annotation(
        train_annot_folder,
        train_image_folder,
        train_val_list,
        cache_name,
        labels
    )

    # run k_mean to find the anchors
    annotation_dims = []
    for image in train_imgs:
        # print(image['filename'])
        for obj in image['object']:
            relative_w = (float(obj['xmax']) - float(obj['xmin']))/image['width']
            relatice_h = (float(obj["ymax"]) - float(obj['ymin']))/image['height']
            annotation_dims.append(tuple(map(float, (relative_w,relatice_h))))

    annotation_dims = np.array(annotation_dims)
    centroids = run_kmeans(annotation_dims, num_anchors)

    # write anchors to file
    print('\naverage IOU for', num_anchors, 'anchors:', '%0.2f' % avg_IOU(annotation_dims, centroids))
    print_anchors(centroids)

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()

    argparser.add_argument(
        '-c',
        '--conf',
        default='config.json',
        help='path to configuration file')
    argparser.add_argument(
        '-a',
        '--anchors',
        default=9,
        help='number of anchors to use')

    args = argparser.parse_args()
    _main_(args)