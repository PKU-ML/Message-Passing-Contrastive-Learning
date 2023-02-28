# Copyright 2022 solo-learn development team.

# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
# Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies
# or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
# FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.



import logging
from argparse import ArgumentParser
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from solo.methods.base import BaseMethod
from solo.utils.lars import LARS
from solo.utils.metrics import accuracy_at_k, weighted_mean
from solo.utils.misc import param_groups_layer_decay, remove_bias_and_norm_from_weight_decay
from torch.optim.lr_scheduler import ExponentialLR, MultiStepLR, ReduceLROnPlateau
import numpy as np
from sklearn.cluster import SpectralClustering
class LinearModel(pl.LightningModule):
    _OPTIMIZERS = {
        "sgd": torch.optim.SGD,
        "lars": LARS,
        "adam": torch.optim.Adam,
        "adamw": torch.optim.AdamW,
    }
    _SCHEDULERS = [
        "reduce",
        "warmup_cosine",
        "step",
        "exponential",
        "none",
    ]

    def __init__(
        self,
        backbone: nn.Module,
        loss_func: Callable,
        num_classes: int,
        max_epochs: int,
        batch_size: int,
        optimizer: str,
        lr: float,
        weight_decay: float,
        extra_optimizer_args: dict,
        scheduler: str,
        min_lr: float,
        warmup_start_lr: float,
        warmup_epochs: float,
        finetune: bool = False,
        mixup_func: Callable = None,
        scheduler_interval: str = "step",
        lr_decay_steps: Optional[Sequence[int]] = None,
        no_channel_last: bool = False,
        **kwargs,
    ):
        """Implements linear evaluation.

        Args:
            backbone (nn.Module): backbone architecture for feature extraction.
            loss_func (Callable): loss function to use (for mixup, label smoothing or default).
            num_classes (int): number of classes in the dataset.
            max_epochs (int): total number of epochs.
            batch_size (int): batch size.
            optimizer (str): optimizer to use.
            weight_decay (float): weight decay.
            extra_optimizer_args (dict): extra optimizer arguments.
            scheduler (str): learning rate scheduler.
            min_lr (float): minimum learning rate for warmup scheduler.
            warmup_start_lr (float): initial learning rate for warmup scheduler.
            warmup_epochs (float): number of warmup epochs.
            mixup_func (Callable, optional). function to convert data and targets with mixup/cutmix.
                Defaults to None.
            finetune (bool): whether or not to finetune the backbone. Defaults to False.
            scheduler_interval (str): interval to update the lr scheduler. Defaults to 'step'.
            lr_decay_steps (Optional[Sequence[int]], optional): list of epochs where the learning
                rate will be decreased. Defaults to None.
            no_channel_last (bool). Disables channel last conversion operation which
                speeds up training considerably. Defaults to False.
                https://pytorch.org/tutorials/intermediate/memory_format_tutorial.html#converting-existing-models
        """

        super().__init__()

        self.backbone = backbone
        if hasattr(self.backbone, "inplanes"):
            features_dim = self.backbone.inplanes
        else:
            features_dim = self.backbone.num_features
        self.classifier = nn.Linear(features_dim, num_classes)  # type: ignore

        if loss_func is None:
            loss_func = nn.CrossEntropyLoss()
        self.loss_func = loss_func

        # training related
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.lr = lr
        self.weight_decay = weight_decay
        self.extra_optimizer_args = extra_optimizer_args
        self.scheduler = scheduler
        self.min_lr = min_lr
        self.warmup_start_lr = warmup_start_lr
        self.warmup_epochs = warmup_epochs
        self.finetune = finetune
        self.mixup_func = mixup_func
        assert scheduler_interval in ["step", "epoch"]
        self.scheduler_interval = scheduler_interval
        self.lr_decay_steps = lr_decay_steps
        self.no_channel_last = no_channel_last

        # all the other parameters
        self.extra_args = kwargs

        if not finetune:
            for param in self.backbone.parameters():
                param.requires_grad = False

        if scheduler_interval == "step":
            logging.warn(
                f"Using scheduler_interval={scheduler_interval} might generate "
                "issues when resuming a checkpoint."
            )

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        """Adds basic linear arguments.

        Args:
            parent_parser (ArgumentParser): argument parser that is used to create a
                argument group.

        Returns:
            ArgumentParser: same as the argument, used to avoid errors.
        """

        parser = parent_parser.add_argument_group("linear")

        # backbone args
        parser.add_argument("--backbone", choices=BaseMethod._BACKBONES, type=str)
        # for ViT
        parser.add_argument("--patch_size", type=int, default=16)

        # if we want to finetune the backbone
        parser.add_argument("--finetune", action="store_true")

        # general train
        parser.add_argument("--batch_size", type=int, default=128)
        parser.add_argument("--lr", type=float, default=0.3)
        parser.add_argument("--classifier_lr", type=float, default=0.3)
        parser.add_argument("--weight_decay", type=float, default=0.0001)
        parser.add_argument("--num_workers", type=int, default=4)

        # wandb
        parser.add_argument("--name")
        parser.add_argument("--project")
        parser.add_argument("--entity", default=None, type=str)
        parser.add_argument("--wandb", action="store_true")
        parser.add_argument("--offline", action="store_true")

        parser.add_argument(
            "--optimizer", choices=LinearModel._OPTIMIZERS.keys(), type=str, required=True
        )
        parser.add_argument("--exclude_bias_n_norm_wd", action="store_true")
        # lars args
        parser.add_argument("--exclude_bias_n_norm_lars", action="store_true")
        # adamw args
        parser.add_argument("--adamw_beta1", default=0.9, type=float)
        parser.add_argument("--adamw_beta2", default=0.999, type=float)

        parser.add_argument(
            "--scheduler", choices=LinearModel._SCHEDULERS, type=str, default="reduce"
        )
        parser.add_argument("--lr_decay_steps", default=None, type=int, nargs="+")
        parser.add_argument("--min_lr", default=0.0, type=float)
        parser.add_argument("--warmup_start_lr", default=0.003, type=float)
        parser.add_argument("--warmup_epochs", default=10, type=int)
        parser.add_argument(
            "--scheduler_interval", choices=["step", "epoch"], default="step", type=str
        )

        # disables channel last optimization
        parser.add_argument("--no_channel_last", action="store_true")

        return parent_parser

    def configure_optimizers(self) -> Tuple[List, List]:
        """Configures the optimizer for the linear layer.

        Raises:
            ValueError: if the optimizer is not in (sgd, adam).
            ValueError: if the scheduler is not in not in (warmup_cosine, cosine, reduce, step,
                exponential).

        Returns:
            Tuple[List, List]: two lists containing the optimizer and the scheduler.
        """

        assert self.optimizer in self._OPTIMIZERS
        optimizer = self._OPTIMIZERS[self.optimizer]

        layer_decay = self.extra_args.get("layer_decay", 0)
        if layer_decay > 0:
            assert self.finetune, "Only with use layer weight decay with finetune on."
            learnable_params = param_groups_layer_decay(
                self.backbone,
                self.weight_decay,
                no_weight_decay_list=self.backbone.no_weight_decay(),
                layer_decay=layer_decay,
            )
            learnable_params.append({"name": "classifier", "params": self.classifier.parameters()})
        else:
            learnable_params = (
                self.classifier.parameters()
                if not self.finetune
                else [
                    {"name": "backbone", "params": self.backbone.parameters()},
                    {"name": "classifier", "params": self.classifier.parameters()},
                ]
            )

        # exclude bias and norm from weight decay
        if self.extra_args.get("exclude_bias_n_norm_wd", False):
            learnable_params = remove_bias_and_norm_from_weight_decay(learnable_params)

        optimizer = optimizer(
            learnable_params,
            lr=self.lr,
            weight_decay=self.weight_decay,
            **self.extra_optimizer_args,
        )

        # select scheduler
        if self.scheduler == "none":
            return optimizer

        if self.scheduler == "warmup_cosine":
            max_warmup_steps = (
                self.warmup_epochs * (self.trainer.estimated_stepping_batches / self.max_epochs)
                if self.scheduler_interval == "step"
                else self.warmup_epochs
            )
            max_scheduler_steps = (
                self.trainer.estimated_stepping_batches
                if self.scheduler_interval == "step"
                else self.max_epochs
            )
            scheduler = {
                "scheduler": LinearWarmupCosineAnnealingLR(
                    optimizer,
                    warmup_epochs=max_warmup_steps,
                    max_epochs=max_scheduler_steps,
                    warmup_start_lr=self.warmup_start_lr if self.warmup_epochs > 0 else self.lr,
                    eta_min=self.min_lr,
                ),
                "interval": self.scheduler_interval,
                "frequency": 1,
            }
        elif self.scheduler == "reduce":
            scheduler = ReduceLROnPlateau(optimizer)
        elif self.scheduler == "step":
            scheduler = MultiStepLR(optimizer, self.lr_decay_steps, gamma=0.1)
        elif self.scheduler == "exponential":
            scheduler = ExponentialLR(optimizer, self.weight_decay)
        else:
            raise ValueError(
                f"{self.scheduler} not in (warmup_cosine, cosine, reduce, step, exponential)"
            )

        return [optimizer], [scheduler]

    def forward(self, X: torch.tensor) -> Dict[str, Any]:
        """Performs forward pass of the frozen backbone and the linear layer for evaluation.

        Args:
            X (torch.tensor): a batch of images in the tensor format.

        Returns:
            Dict[str, Any]: a dict containing features and logits.
        """

        if not self.no_channel_last:
            X = X.to(memory_format=torch.channels_last)

        with torch.set_grad_enabled(self.finetune):
            feats = self.backbone(X)

        logits = self.classifier(feats)
        return {"logits": logits, "feats": feats}

    def shared_step(
        self, batch: Tuple, batch_idx: int
    ) -> Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Performs operations that are shared between the training nd validation steps.

        Args:
            batch (Tuple): a batch of images in the tensor format.
            batch_idx (int): the index of the batch.

        Returns:
            Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]:
                batch size, loss, accuracy @1 and accuracy @5.
        """

        X, target = batch

        metrics = {"batch_size": X.size(0)}
        if self.training and self.mixup_func is not None:
            X, target = self.mixup_func(X, target)
            out = self(X)["logits"]
            loss = self.loss_func(out, target)
            metrics.update({"loss": loss})
        else:
            out = self(X)["logits"]
            loss = F.cross_entropy(out, target)
            acc1, acc5 = accuracy_at_k(out, target, top_k=(1, 5))
            metrics.update({"loss": loss, "acc1": acc1, "acc5": acc5,"fea": self(X)[
                "feats"],"targets":target})

        return metrics

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """Performs the training step for the linear eval.

        Args:
            batch (torch.Tensor): a batch of images in the tensor format.
            batch_idx (int): the index of the batch.

        Returns:
            torch.Tensor: cross-entropy loss between the predictions and the ground truth.
        """

        # set backbone to eval mode
        if not self.finetune:
            self.backbone.eval()

        out = self.shared_step(batch, batch_idx)

        log = {"train_loss": out["loss"]}
        if self.mixup_func is None:
            log.update({"train_acc1": out["acc1"], "train_acc5": out["acc5"]})

        self.log_dict(log, on_epoch=True, sync_dist=True)
        return {"loss": out["loss"],
                "feats":out["fea"],
                "targets":out["targets"],
                }

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> Dict[str, Any]:
        """Performs the validation step for the linear eval.

        Args:
            batch (torch.Tensor): a batch of images in the tensor format.
            batch_idx (int): the index of the batch.

        Returns:
            Dict[str, Any]:
                dict with the batch_size (used for averaging),
                the classification loss and accuracies.
        """

        out = self.shared_step(batch, batch_idx)

        results = {
            "batch_size": out["batch_size"],
            "val_loss": out["loss"],
            "val_acc1": out["acc1"],
            "val_acc5": out["acc5"],
        }
        return results

    def validation_epoch_end(self, outs: List[Dict[str, Any]]):
        """Averages the losses and accuracies of all the validation batches.
        This is needed because the last batch can be smaller than the others,
        slightly skewing the metrics.

        Args:
            outs (List[Dict[str, Any]]): list of outputs of the validation step.
        """

        val_loss = weighted_mean(outs, "val_loss", "batch_size")
        val_acc1 = weighted_mean(outs, "val_acc1", "batch_size")
        val_acc5 = weighted_mean(outs, "val_acc5", "batch_size")

        log = {"val_loss": val_loss, "val_acc1": val_acc1, "val_acc5": val_acc5}
        self.log_dict(log, sync_dist=True)

    def training_epoch_end(self,outs: List[Dict[str, Any]]):
        print(outs[0])
        total =  outs[0]["feats"] 
        for i in range(len(outs)):
            if len(total)<=10000:
                #outs[i]["feats"] = F.normalize(outs[i]["feats"])
                total = torch.cat((total,outs[i]["feats"]),0)
            else:
                break
        print(total.size())

        '''
        #total = total[0:2000]
        total = total.detach()
        sim = total@total.t()
        Ad = myKNN(sim.detach().cpu().numpy(),20)
        Lap = CalLap(Ad)
        lam,H = np.linalg.eig(Lap)
        Lap = torch.tensor(Lap)
        print(lam,H)
        centers,distances, avgdis= _kmeans_batch(total,distance_function=l2_distance,batch_size=256,k=100)
        print(centers)
        print(distances)
        print(avgdis)
        '''

        total = total.float()
        total = total[0:10000]
        total = F.normalize(total)
        total2 = total@total.t()
        y_pred = SpectralClustering(n_clusters=100,affinity='precomputed').fit_predict(total2.detach().cpu().numpy())
        print(y_pred)
        dis = [total[0:2] for i in range(100)]
        for i in range(len(y_pred)):
            print(i)
            dis[y_pred[i]] = torch.cat((dis[y_pred[i]],total[i:i+1]),0)
        mean_dis = 0
        for i in range(100):
            avg_dis = torch.cdist(dis[i],dis[i])
            mean_dis+=torch.mean(torch.mean(avg_dis)).item()
        mean_dis = mean_dis/100
        out_dis = torch.cdist(total,total)
        out_dis = torch.mean(torch.mean(out_dis))
        print(mean_dis,out_dis)

        exit()






def cosine_distance(obs, centers):
    obs_norm = obs / obs.norm(dim=1, keepdim=True)
    centers_norm = centers / centers.norm(dim=1, keepdim=True)
    cos = torch.matmul(obs_norm, centers_norm.transpose(1, 0))
    return 1 - cos


def l2_distance(obs, centers):
    dis = ((obs.unsqueeze(dim=1) - centers.unsqueeze(dim=0)) ** 2.0).sum(dim=-1).squeeze()
    return dis


def _kmeans_batch(obs: torch.Tensor,
                  k: int,
                  distance_function,
                  batch_size=0,
                  thresh=1e-5,
                  norm_center=False):
    # k x D
    centers = obs[torch.randperm(obs.size(0))[:k]].clone()
    history_distances = [float('inf')]
    avg_distances =[float('inf')]
    if batch_size == 0:
        batch_size = obs.shape[0]
    while True:
        # (N x D, k x D) -> N x k
        segs = torch.split(obs, batch_size)
        seg_center_dis = []
        seg_center_ids = []
        seg_center_avg = []
        for seg in segs:
            distances = distance_function(seg, centers)
            center_dis, center_ids = distances.min(dim=1)
            seg_center_ids.append(center_ids)
            seg_center_dis.append(center_dis)
            avg_dis = distances.mean(dim=1)
            seg_center_avg.append(avg_dis)


        obs_center_dis_mean = torch.cat(seg_center_dis).mean()
        obs_center_ids = torch.cat(seg_center_ids)
        obs_avg_dis = torch.cat(seg_center_avg).mean()
        

        history_distances.append(obs_center_dis_mean.item())
        avg_distances.append(obs_avg_dis.item())
        diff = history_distances[-2] - history_distances[-1]
        if diff < thresh:
            if diff < 0:
                warnings.warn("Distance diff < 0, distances: " + ", ".join(map(str, history_distances)))
            break
        for i in range(k):
            obs_id_in_cluster_i = obs_center_ids == i
            if obs_id_in_cluster_i.sum() == 0:
                continue
            obs_in_cluster = obs.index_select(0, obs_id_in_cluster_i.nonzero().squeeze())
            c = obs_in_cluster.mean(dim=0)
            if norm_center:
                c /= c.norm()
            centers[i] = c
    return centers, history_distances[-1],avg_distances[-1]
  
  
def myKNN(S, k, sigma=1.0):
    N = len(S)
    A = np.zeros((N,N))
 
    for i in range(N):
        dist_with_index = zip(S[i], range(N))
        dist_with_index = sorted(dist_with_index, key=lambda x:x[0],reverse=True)
        neighbours_id = [dist_with_index[m][1] for m in range(k+1)] # xi's k nearest neighbours
 
        for j in neighbours_id: # xj is xi's neighbour
            A[i][j] = np.exp(-S[i][j]/2/sigma/sigma)
            A[j][i] = A[i][j] # mutually
 
    return A



def CalLap(adjacentMatrix):
 
    # compute the Degree Matrix: D=sum(A)
    degreeMatrix = np.sum(adjacentMatrix, axis=1)
 
    # compute the Laplacian Matrix: L=D-A
    laplacianMatrix = np.diag(degreeMatrix) - adjacentMatrix
 
    # normailze
    # D^(-1/2) L D^(-1/2)
    sqrtDegreeMatrix = np.diag(1.0 / (degreeMatrix ** (0.5)))
    return np.dot(np.dot(sqrtDegreeMatrix, laplacianMatrix), sqrtDegreeMatrix)
