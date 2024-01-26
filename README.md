## imgcls_pytorch
A image classification pytorch framework supporting SOTA backbones

## Prepare dataset
Put your images under a directory as below
* your_dataset_directory
  * class1
    * 1.jpg
    * 2.jpg
  * class2
    * 1.jpg
    * 2.jpg
    * ... 
  * ...

## Train
```shell
CUDA_VISIBLE_DEVICES=0 python3 -u train.py --backbone resnet101 --workers 32 --lr=0.001 --epochs 30 --train_bs 160 --datadir your_dataset_directory
```

## Demo
CUDA_VISIBLE_DEVICES=0 PORT=8000 python3 -u app.py

## Tensorboard displaying Chinese support
```shell
python3 fixfont.py
```
Follow its instructions.

## Supported backbones
* alexnet
* resnet18,resnet34,resnet50,resnet101, resnet152, resnext101_32x4d, resnext101_64x4d
* vgg11_bn, vgg16_bn
* densenet121, densenet169, densenet161
* inceptionv3, inceptionv4, inceptionresnetv2, bninception
* xception, xception_att
* dpn98, dpn107, dpn131
* senet154, se_resnet50, se_resnet101, se_resnet152, se_resnext50_32x4d
* pnasnet5large
* polynet
* efficientnet
