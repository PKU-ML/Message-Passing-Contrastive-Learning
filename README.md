# Message-Passing-Contrastive-Learning

This repository includes a PyTorch implementation of the ICLR 2023 paper [A Message Passing Perspective on Learning Dynamics of Contrastive Learning](https://openreview.net/pdf?id=VBTJqqWjxMv) authored by [Yifei Wang*](https://yifeiwang.me), Qi Zhang*, Tianqi Du, Jiansheng Yang, Zhouchen Lin, [Yisen Wang](https://yisenwang.github.io/).

Multi-stage Graph Aggregation and Graph-Attention are two methods inspired by the connection between message passing and contrastive learning and they can siginificantly improve the performance of sefl-supervised paradigms.

  

| Backbone  | Method        | CIFAR-10 | CIFAR-100 | ImageNet-100 |
|-----------|---------------|:--------:|:---------:|:------------:|
| ResNet-18 | SimSiam       |   83.8   |    56.3   |     68.8     |
|           | SimSiam-Multi |   84.8   |    58.9   |     70.5     |
| ResNet-50 | SimSiam       |   85.9   |    58.4   |     70.9     |
|           | SimSiam-Multi |   87.0   |    59.8   |     72.3     |





| Backbone  | Method      | CIFAR-10 | CIFAR-100 | ImageNet-100 |
|-----------|-------------|----------|-----------|--------------|
| ResNet-18 | SimCLR      | 84.5     | 56.1      | 62.3         |
|           | SimCLR-Attn | 85.4     | 56.9      | 63.1         |
| ResNet-50 | SimCLR      | 88.2     | 59.8      | 66.0         |
|           | SimCLR-Attn | 89.4     | 60.7      | 66.7         |









## Instructions

### Environment Setup

To install the environment for Multi-stage Graph Aggregation with the following commands
```
cd MULTI-STAGE-GRAPH-AGGREGATION
pip3 install .[dali,umap,h5] --extra-index-url https://developer.download.nvidia.com/compute/redist
```

To install the environment for Graph-Attention with the following commands
```
cd GRAPH-ATTENTION
conda env create -f environment.yml
conda activate simclr_pytorch
```

When pretraining the model with the proposed methods, please first enter the corresponding directory (``MULTI-STAGE-GRAPH-AGGREGATION``/``GRAPH-ATTENTION``).

### Pretraining with Multi-stage Graph Aggregation



Taking Simsiam on CIFAR-10 as an example, we pretrain the model with Multi-stage Graph Aggregation technique with following commands


```
./scripts/pretrain/cifar/simsiam.sh
```

The codes provide an online linear classifier. And the offline downstream linear performance can be evaluated with

```
./scripts/linear/simsiam_linear.sh
```


### Pretraining with Graph-Attention



Taking SimCLR on CIFAR-10 as an example, we pretrain the model with Graph-Attention technique with following commands

```
python train.py --config configs/cifar_train_epochs200_bs512.yaml
```

And the downstream linear performance can be evaluated with

```
python train.py --config configs/cifar_eval.yaml --encoder_ckpt <path-to-encoder>
```

More running details can be found in [MULTI-STAGE-GRAPH-AGGREGATION/README_simsiam.md](MULTI-STAGE-GRAPH-AGGREGATION/README_simsiam.md) and [GRAPH-ATTENTION/README_simclr.md](GRAPH-ATTENTION/README_simclr.md).


## Modifications

We follow the default settings of SimSiam and SimCLR,  and the main modifications are:

In [MULTI-STAGE-GRAPH-AGGREGATION/solo/method/simsiam.py](MULTI-STAGE-GRAPH-AGGREGATION/solo/methods/simsiam.py), to implement the Multi-stage Graph Aggregation, we add a memory bank to store the latest features and modify the loss function by combing the latest features of the last epoch .

In [GRAPH-ATTENTION/models/losses.py](GRAPH-ATTENTION/models/losses.py), to implement the Graph Attention, we evaluate the similarity between the features in the same batch and reweight the InfoNCE loss with that.



## Citing this work


If you find our code useful, please cite
```
@inproceedings{
wang2023message,
title={A Message Passing Perspective on Learning Dynamics of Contrastive Learning},
author={Yifei Wang and Qi Zhang and Tianqi Du and Jiansheng Yang and Zhouchen Lin and Yisen Wang},
booktitle={International Conference on Learning Representations},
year={2023},
}
```


## Acknowledgement

Our codes borrows the implementations of SimSiam and SimCLR in these respositories:

https://github.com/vturrisi/solo-learn

https://github.com/google-research/simclr


