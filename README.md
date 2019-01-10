## YOLO v3 implementation With pytorch 
> this repository only contain the detection module and we don't need the cfg from original YOLOv3, we implement it with pytorch.

This repository is based on the official code of [YOLOv3](https://github.com/pjreddie/darknet) and [pytorch-yolo-v3](https://github.com/ayooshkathuria/pytorch-yolo-v3). There's also an implementation for YOLOv3 already for pytorch, but it uses a config file rather than a normal pytorch approch to defining the network. One of the goals of this repository is to remove the cfg file.

## Requirements

* Python 3.5
* OpenCV
* PyTorch 0.4

## Installation

* Install PyTorch-0.4.0 by selecting your environment on the website and running the appropriate command.
* Clone this repository
* Compile the nms
* convert yolov3.weights to pytorch

```shell
cd YOLOv3_Pytorch
./make.sh

mkdir weights
cd weights
wget https://pjreddie.com/media/files/yolov3.weights
cd ..
python convert_darknet.py --version coco --weights ./weights/yolov3.weights --save_name ./weights/convert_yolov3_coco.pth
# we will get the convert_yolov3_coco.pth
```

## Train
> We only train voc dataset because we don't have enough gpus to train coco datatset. This is still an experimental repository, we don't reproduce the original results very well.

### dataset
[merge VOC dataset](https://github.com/yqyao/DRFNet#voc-dataset)

* structure

./data/datasets/VOCdevkit0712/VOC0712/Annotations  
./data/datasets/VOCdevkit0712/VOC0712/ImageSets  
./data/datasets/VOCdevkit0712/VOC0712/JPEGImages  

* COCO 

Same with [COCO](https://github.com/yqyao/DRFNet#coco-dataset)

### train
> you can train multiscale by changing data/config voc_config multiscale

* convert weights
```shell
cd weights
wget wget https://pjreddie.com/media/files/darknet53.conv.74
cd ../
python convert_darknet.py --version darknet53 --weights ./weights/darknet53.conv.74 --save_name ./weights/convert_darknet53.pth
```

* train yolov3

```python
python train.py --input_wh 416 416 -b 64 --subdivisions 4 -d VOC --basenet ./weights/convert_darknet53.pth

```

### eval

```python

python eval.py --weights ./weights/convert_yolov3_voc.pth --dataset VOC --input_wh 416 416
```
> darknet voc is trained by darknet, pytorch voc is trained by this repository

**results**

| darknet voc 608 | darknet voc 416 | pytorch voc 608| pytorch voc 416|
|:-:              |:-:              | :-:            |:-:             |
| 77.2 %          |      76.2%      |      74.7%     |          74.9% |
|       27ms      |       18ms      |        27ms    |       18ms     |

## Demo

```python

python demo.py --images images --save_path ./output --weights ./weights/convert_yolov3_coco.pth -d COCO

```

## Example
<img align="center" src= "https://github.com/yqyao/YOLOv3_Pytorch/blob/master/output/output_person.jpg">
<!-- ![](https://github.com/yqyao/YOLOv3_Pytorch.git/output/output_person.jpg) -->


## References
- [YOLOv3: An Incremental Improvemet](https://pjreddie.com/media/files/papers/YOLOv3.pdf)

- [Original Implementation (Darknet)](https://github.com/pjreddie/darknet)

- [pytorch-yolo-v3](https://github.com/ayooshkathuria/pytorch-yolo-v3)

- [pytorch-yolo2](https://github.com/marvis/pytorch-yolo2)
