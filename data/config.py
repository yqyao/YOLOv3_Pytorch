# config.py
import os
import os.path

pwd = os.getcwd()
VOCroot = os.path.join(pwd, "data/datasets/VOCdevkit0712/")
COCOroot = os.path.join(pwd, "data/datasets/coco2015")

datasets_dict = {"VOC": [('0712', '0712_trainval')],
            "VOC0712++": [('0712', '0712_trainval_test')],
            "VOC2012" : [('2012', '2012_trainval')],
            "COCO": [('2014', 'train'), ('2014', 'valminusminival')],
            "VOC2007": [('0712', "2007_test")],
            "COCOval": [('2014', 'minival')]}


voc_config = {
    'anchors' : [[116, 90], [156, 198], [373, 326], 
                [30, 61], [62, 45], [59, 119], 
                [10, 13], [16, 30], [33, 23]],
    'root': VOCroot,
    'num_classes': 20,
    'multiscale' : True,
    'name_path' : "./model/voc.names",
    'anchors_mask' : [[0,1,2], [3,4,5], [6,7,8]]
}

coco_config = {
    'anchors' : [[116, 90], [156, 198], [373, 326], 
                [30, 61], [62, 45], [59, 119], 
                [10, 13], [16, 30], [33, 23]],
    'root': COCOroot,
    'num_classes': 80,
    'multiscale' : True,
    'name_path' : "./model/coco.names",
    'anchors_mask' : [[0,1,2], [3,4,5], [6,7,8]]
}

    # anchors = [[214,327], [326,193], [359,359],
    #         [116,286], [122,97], [171,180],
    #         [24,34], [46,84], [68,185]]