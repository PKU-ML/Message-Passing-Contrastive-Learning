# SimCLR 



## Enviroment Setup


Create a python enviroment with the provided config file and [miniconda](https://docs.conda.io/en/latest/miniconda.html):

```(bash)
conda env create -f environment.yml
conda activate simclr_pytorch

export IMAGENET_PATH=... # If you have enough RAM using /dev/shm usually accelerates data loading time
export EXMAN_PATH=... # A path to logs
```

## Training
Model training consists of two steps: (1) self-supervised encoder pretraining and (2) classifier learning with the encoder representations. Both steps are done with the `train.py` script. To see the help for `sim-clr/eval` problem call the following command: `python source/train.py --help --problem sim-clr/eval`.

### Self-supervised pretraining

#### CIFAR-10
The config `cifar_train_epochs200_bs512.yaml` contains the parameters to reproduce results for CIFAR dataset. The pretraining command is:

```(bash)
python train.py --config configs/cifar_train_epochs200_bs512.yaml
```

#### ImageNet-100
The configs `imagenet_params_epochs100_bs512.yaml` contain the parameters to reproduce results for ImageNet-100 dataset. The single-node (4 v100 GPUs) pretraining command is:

```(bash)
python train.py --config configs/imagenet_train_epochs100_bs512.yaml
```

#### Logs
The logs and the model will be stored at `./logs/exman-train.py/runs/<experiment-id>/`. You can access all the experiments from python with `exman.Index('./logs/exman-train.py').info()`.


### Linear Evaluation
To train a linear classifier on top of the pretrained encoder, run the following command:

```(bash)
python train.py --config configs/cifar_eval.yaml --encoder_ckpt <path-to-encoder>
```

 
### Pretraining with `DistributedDataParallel`
To train a model with larger batch size on several nodes you need to set `--dist ddp` flag and specify the following parameters: 
- `--dist_address`: the address and a port of the main node in the `<address>:<port>` format
- `--node_rank`: 0 for the main node and 1,... for the others.
- `--world_size`: the number of nodes.

For example, to train with two nodes you need to run the following command on the main node:
```(bash)
python train.py --config configs/cifar_train_epochs200_bs512.yaml --dist ddp --dist_address <address>:<port> --node_rank 0 --world_size 2
```
and on the second node:
```(bash)
python train.py --config configs/cifar_train_epochs200_bs512.yaml --dist ddp --dist_address <address>:<port> --node_rank 1 --world_size 2
```

The ImageNet the pretaining on 4 nodes all with 4 GPUs looks as follows:
```
node1: python train.py --config configs/imagenet_train_epochs100_bs512.yaml --dist ddp --world_size 4 --dist_address <address>:<port> --node_rank 0
node2: python train.py --config configs/imagenet_train_epochs100_bs512.yaml --dist ddp --world_size 4 --dist_address <address>:<port> --node_rank 1
node3: python train.py --config configs/imagenet_train_epochs100_bs512.yaml --dist ddp --world_size 4 --dist_address <address>:<port> --node_rank 2
node4: python train.py --config configs/imagenet_train_epochs100_bs512.yaml --dist ddp --world_size 4 --dist_address <address>:<port> --node_rank 3
```




