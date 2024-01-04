This repo is a PyTorch implementation for paper: **Progressive Feature Self-Reinforcement for Weakly Supervised Semantic Segmentation**

## Data Preparation

### PASCAL VOC 2012

#### 1. Download

``` bash
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
```
#### 2. Segmentation Labels

The augmented annotations are from [SBD dataset](http://home.bharathh.info/pubs/codes/SBD/download.html). Here is a download link of the augmented annotations at
[DropBox](https://www.dropbox.com/s/oeu149j8qtbs1x0/SegmentationClassAug.zip?dl=0). After downloading ` SegmentationClassAug.zip `, you should unzip it and move it to `VOCdevkit/VOC2012/`. 

```
VOCdevkit
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

```
COCO
├── JPEGImages
│    ├── train2014
│    └── val2014
└── SegmentationClass
     ├── train2014
     └── val2014
```


## Requirement

Please refer to [requirements.txt](https://github.com/Jessie459/feature-self-reinforcement/blob/master/requirements.txt)

Our implementation incorporates a regularization term for segmentation. Please download and compile [the python extension](https://github.com/meng-tang/rloss/tree/master/pytorch#build-python-extension-module).


## Train

The encoder is `vit_base_patch16_224` pretrained on ImageNet. Download the [weights](https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth) to `./pretrained/`. 

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node 4 train_voc.py --data_folder [VOCdevkit/VOC2012]
```
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node 4 train_coco.py --data_folder [COCO]
```

```
arguments most related to this project:

--cls_depth     number of aggregation modules
--out_dim       dimension of the projector output
--momentum      EMA update parameter for teacher
--use_mim       whether to enable masking
--block_size    masking block size, must be a multiple of ViT patch size
--mask_ratio    masking ratio
--w_class       FSR loss weight for the aggregated token
--w_patch       FSR loss weight for masked patch tokens
```

## Evaluation

`infer_*.py` will apply [dense CRF](https://github.com/lucasb-eyer/pydensecrf) to the predicted segmentation labels. 

```
python infer_voc.py --checkpoint [PATH_TO_CHECKPOINT] --data_folder [VOCdevkit/VOC2012] --infer_set [val | test] --save_cam [True | False]
```
```
python infer_coco.py --checkpoint [PATH_TO_CHECKPOINT] --data_folder [COCO] --infer_set val --save_cam [True | False]
```

## Acknowledgement

This repo is built upon [ToCo](https://github.com/rulixiang/ToCo). 
Our work is greatly inspired by [DINO](https://github.com/facebookresearch/dino). 
Many thanks to their brilliant works! 
