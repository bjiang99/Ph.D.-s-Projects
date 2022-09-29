# Jaccard-based SSL
A PyTorch implementation of Jaccard-based modified self-supervised learning model [Refining self-supervised learning in imaging: beyong linear metric] (https://arxiv.org/abs/2202.12921) based on MoCo in CVPR 2020 [Momentum Contrast for Unsupervised Visual Representation Learning](https://arxiv.org/abs/1911.05722).

![Network Architecture image from the paper](results/structure.png)

## Requirements
- [Anaconda](https://www.anaconda.com/download/)
- [PyTorch](https://pytorch.org)
- [Tiny-Imagenet-200]
```
conda install pytorch=1.6.0 torchvision cudatoolkit=10.2 -c pytorch
```

## Dataset
`Tiny-Imagenet-200` dataset is used in this work, the dataset should be downloaded and uploaded into the same directory manually online.

## Usage

### Train MoCo
```
python train.py --batch_size 128 --epochs 50
optional arguments:
--feature_dim                 Feature dim for each image [default value is 128]
--m                           Negative sample number [default value is 4096]
--temperature                 Temperature used in softmax [default value is 0.5]
--momentum                    Momentum used for the update of memory bank [default value is 0.999]
--k                           Top k most similar images used to predict the label [default value is 200]
--batch_size                  Number of images in each mini-batch [default value is 256]
--epochs                      Number of sweeps over the dataset to train [default value is 500]
```

### Linear Evaluation
```
python linear.py --batch_size 1024 --epochs 200 
optional arguments:
--model_path                  The pretrained model path [default value is 'results/...']
--batch_size                  Number of images in each mini-batch [default value is 256]
--epochs                      Number of sweeps over the dataset to train [default value is 100]
```

## Results
There are some difference between this implementation and official implementation, the model (`ResNet50`) is trained on 
one NVIDIA GeForce GTX TITAN GPU:
1. No `Gaussian blur` used;
2. `Adam` optimizer with learning rate `1e-3` is used to replace `SGD` optimizer;
3. No `Linear learning rate scaling` used.




