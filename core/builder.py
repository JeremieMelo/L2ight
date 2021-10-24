"""
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2021-10-24 16:36:55
LastEditors: Jiaqi Gu (jqgu@utexas.edu)
LastEditTime: 2021-10-24 16:40:35
"""

from typing import Tuple

import torch
import torch.nn as nn
import torchvision
from .dataset import get_dataset
from pyutils.typing import DataLoader, Optimizer, Scheduler
from torch.types import Device
from pyutils.config import configs

from core.models import *
from core.optimizer import FLOPSOptimizer, MixedTrainOptimizer

__all__ = ["make_dataloader", "make_model", "make_optimizer", "make_scheduler", "make_criterion"]


def make_dataloader() -> Tuple[DataLoader, DataLoader]:
    train_dataset, validation_dataset = get_dataset(
        configs.dataset.name,
        configs.dataset.img_height,
        configs.dataset.img_width,
        dataset_dir=configs.dataset.root,
        transform=configs.dataset.transform,
    )

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=configs.run.batch_size,
        shuffle=int(configs.dataset.shuffle),
        pin_memory=True,
        num_workers=configs.dataset.num_workers,
    )
    validation_loader = torch.utils.data.DataLoader(
        dataset=validation_dataset,
        batch_size=configs.run.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=configs.dataset.num_workers,
    )
    return train_loader, validation_loader


def make_model(device: Device, random_state: int = None) -> nn.Module:
    if "mlp" in configs.model.name.lower():
        model = eval(configs.model.name)(
            n_feat=configs.dataset.img_height * configs.dataset.img_width,
            n_class=configs.dataset.n_class,
            hidden_list=configs.model.hidden_list,
            block_list=configs.model.block_list,
            in_bit=configs.quantize.input_bit,
            w_bit=configs.quantize.weight_bit,
            mode=configs.model.mode,
            v_max=configs.quantize.v_max,
            v_pi=configs.quantize.v_pi,
            act_thres=configs.model.act_thres,
            photodetect=False,
            bias=False,
            device=device,
        ).to(device)
        model.reset_parameters(random_state)
    elif "cnn" in configs.model.name.lower():
        model = eval(configs.model.name)(
            img_height=configs.dataset.img_height,
            img_width=configs.dataset.img_width,
            in_channel=configs.dataset.in_channel,
            n_class=configs.dataset.n_class,
            kernel_list=configs.model.kernel_list,
            kernel_size_list=configs.model.kernel_size_list,
            pool_out_size=configs.model.pool_out_size,
            stride_list=configs.model.stride_list,
            padding_list=configs.model.padding_list,
            hidden_list=configs.model.hidden_list,
            block_list=configs.model.block_list,
            in_bit=configs.quantize.input_bit,
            w_bit=configs.quantize.weight_bit,
            mode=configs.model.mode,
            v_max=configs.quantize.v_max,
            v_pi=configs.quantize.v_pi,
            act_thres=configs.model.act_thres,
            photodetect=False,
            bias=False,
            device=device,
        ).to(device)
        model.reset_parameters(random_state)
    elif "vgg" in configs.model.name.lower():
        model = eval(configs.model.name)(
            img_height=configs.dataset.img_height,
            img_width=configs.dataset.img_width,
            in_channel=configs.dataset.in_channel,
            n_class=configs.dataset.n_class,
            block_list=configs.model.block_list,
            in_bit=configs.quantize.input_bit,
            w_bit=configs.quantize.weight_bit,
            mode=configs.model.mode,
            v_max=configs.quantize.v_max,
            v_pi=configs.quantize.v_pi,
            act_thres=configs.model.act_thres,
            photodetect=False,
            bias=False,
            device=device,
        ).to(device)
        model.reset_parameters(random_state)
    elif "resnet" in configs.model.name.lower():
        model = eval(configs.model.name)(
            img_height=configs.dataset.img_height,
            img_width=configs.dataset.img_width,
            in_channel=configs.dataset.in_channel,
            n_class=configs.dataset.n_class,
            block_list=configs.model.block_list,
            in_bit=configs.quantize.input_bit,
            w_bit=configs.quantize.weight_bit,
            mode=configs.model.mode,
            v_max=configs.quantize.v_max,
            v_pi=configs.quantize.v_pi,
            act_thres=configs.model.act_thres,
            photodetect=False,
            bias=False,
            device=device,
        ).to(device)
        model.reset_parameters(random_state)
    else:
        model = None
        raise NotImplementedError(f"Not supported model name: {configs.model.name}")

    return model


def make_optimizer(model: nn.Module) -> Optimizer:
    if configs.optimizer.name == "sgd":
        optimizer = torch.optim.SGD(
            (p for p in model.parameters() if p.requires_grad),
            lr=configs.optimizer.lr,
            momentum=configs.optimizer.momentum,
            weight_decay=configs.optimizer.weight_decay,
            nesterov=True,
        )
    elif configs.optimizer.name == "adam":
        optimizer = torch.optim.Adam(
            (p for p in model.parameters() if p.requires_grad),
            lr=configs.optimizer.lr,
            weight_decay=configs.optimizer.weight_decay,
        )
    elif configs.optimizer.name == "adamw":
        optimizer = torch.optim.AdamW(
            (p for p in model.parameters() if p.requires_grad),
            lr=configs.optimizer.lr,
            weight_decay=configs.optimizer.weight_decay,
        )
    elif configs.optimizer.name == "mixedtrain":
        optimizer = MixedTrainOptimizer(
            model,
            lr=configs.optimizer.lr,
            param_sparsity=configs.optimizer.param_sparsity,
            grad_sparsity=configs.optimizer.grad_sparsity,
            criterion=make_criterion(),
            random_state=configs.noise.random_state,
        )
    elif configs.optimizer.name == "flops":
        optimizer = FLOPSOptimizer(
            model,
            lr=configs.optimizer.lr,
            sigma=configs.optimizer.sigma,
            n_sample=configs.optimizer.n_sample,
            criterion=make_criterion(),
            random_state=configs.noise.random_state,
        )
    else:
        raise NotImplementedError(configs.optimizer.name)

    return optimizer


def make_scheduler(optimizer: Optimizer) -> Scheduler:
    if configs.scheduler.name == "constant":
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1)
    elif configs.scheduler.name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=configs.run.n_epochs, eta_min=configs.scheduler.lr_min
        )
    elif configs.scheduler.name == "exp":
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=configs.scheduler.lr_gamma)
    else:
        raise NotImplementedError(configs.scheduler.name)

    return scheduler


def make_criterion() -> nn.Module:
    if configs.criterion.name == "nll":
        criterion = nn.NLLLoss()
    elif configs.criterion.name == "mse":
        criterion = nn.MSELoss()
    elif configs.criterion.name == "ce":
        criterion = nn.CrossEntropyLoss()
    else:
        raise NotImplementedError(configs.criterion.name)
    return criterion
