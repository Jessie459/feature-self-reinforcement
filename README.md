## Progressive Uncertain Feature Self-reinforcement for Weakly Supervised Semantic Segmentation

This repo is a PyTorch implementation for paper: Progressive Uncertain Feature Self-reinforcement for Weakly Supervised Semantic Segmentation

<img src="assets/overview.png" alt="overview" width="100%"/>

## Data Preparation

### PASCAL VOC 2012

#### 1. Download

``` bash
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
```
#### 2. Segmentation Labels

The augmented annotations are from [SBD dataset](http://home.bharathh.info/pubs/codes/SBD/download.html). Here is a download link of the augmented annotations at
[DropBox](https://www.dropbox.com/s/oeu149j8qtbs1x0/SegmentationClassAug.zip?dl=0). After downloading ` SegmentationClassAug.zip `, you should unzip it and move it to `VOCdevkit/VOC2012/`. 

``` bash
VOCdevkit/
└── VOC2012
    ├── Annotations
    ├── ImageSets
    ├── JPEGImages
    ├── SegmentationClass
    ├── SegmentationClassAug
    └── SegmentationObject
```

### MSCOCO 2014

#### 1. Download
``` bash
wget http://images.cocodataset.org/zips/train2014.zip
wget http://images.cocodataset.org/zips/val2014.zip
```

#### 2. Segmentation Labels

To generate VOC style segmentation labels for COCO, you could use the scripts provided at this [repo](https://github.com/alicranck/coco2voc), or just download the generated masks from [Google Drive](https://drive.google.com/file/d/147kbmwiXUnd2dW9_j8L5L0qwFYHUcP9I/view?usp=share_link).

``` bash
COCO/
├── JPEGImages
│    ├── train2014
│    └── val2014
└── SegmentationClass
     ├── train2014
     └── val2014
```


## Requirement

Please refer to requirements.txt

Our implementation incorporates a regularization term for segmentation. Please download and compile [the python extension](https://github.com/meng-tang/rloss/tree/master/pytorch#build-python-extension-module).


## Train

```bash
bash train_voc.sh

bash train_coco.sh
```

## Evaluation

```bash
bash infer_voc.sh

bash infer_coco.sh
```

## Checkpoints

Coming soon. 

## Acknowledgement

This repo is built upon [ToCo](https://github.com/rulixiang/ToCo). 
Our work is greatly inspired by [DINO](https://github.com/facebookresearch/dino). 
Many thanks to their brilliant works! 
