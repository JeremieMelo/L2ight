'''
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2021-10-24 16:23:50
LastEditors: Jiaqi Gu (jqgu@utexas.edu)
LastEditTime: 2021-10-24 16:23:50
'''
from collections import OrderedDict
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from pyutils.general import logger
from torch import Tensor, nn
from torch.types import Device, _size

from .layers.activation import ReLUN
from .layers.custom_conv2d import MZIBlockConv2d
from .layers.custom_linear import MZIBlockLinear
from .sparse_bp_base import SparseBP_Base

__all__ = ["SparseBP_MZI_CNN"]


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        kernel_size: int = 3,
        miniblock: int = 8,
        bias: bool = False,
        stride: Union[int, _size] = 1,
        padding: Union[int, _size] = 0,
        mode: str = "weight",
        v_max: float = 4.36,  # 0-pi for clements, # 6.166 is v_2pi, 0-2pi for reck
        v_pi: float = 4.36,
        w_bit: int = 16,
        in_bit: int = 16,
        photodetect: bool = False,
        act_thres: int = 6,
        device: Device = torch.device("cuda"),
    ) -> None:
        super().__init__()
        self.conv = MZIBlockConv2d(
            in_channel,
            out_channel,
            kernel_size,
            miniblock,
            bias,
            stride,
            padding,
            mode=mode,
            v_max=v_max,
            v_pi=v_pi,
            w_bit=w_bit,
            in_bit=in_bit,
            photodetect=photodetect,
            device=device,
        )

        self.bn = nn.BatchNorm2d(out_channel)

        self.activation = ReLUN(act_thres, inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.activation(self.bn(self.conv(x)))


class LinearBlock(nn.Module):
    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        miniblock: int = 8,
        bias: bool = False,
        mode: str = "weight",
        v_max: float = 4.36,  # 0-pi for clements, # 6.166 is v_2pi, 0-2pi for reck
        v_pi: float = 4.36,
        w_bit: int = 16,
        in_bit: int = 16,
        photodetect: bool = False,
        activation: bool = True,
        act_thres: int = 6,
        device: Device = torch.device("cuda"),
    ) -> None:
        super().__init__()
        self.linear = MZIBlockLinear(
            in_channel, out_channel, miniblock, bias, mode, v_max, v_pi, w_bit, in_bit, photodetect, device
        )

        self.activation = ReLUN(act_thres, inplace=True) if activation else None

    def forward(self, x: Tensor) -> Tensor:
        x = self.linear(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class SparseBP_MZI_CNN(SparseBP_Base):
    """MZI CNN (Shen+, Nature Photonics 2017). Support sparse backpropagation. Blocking matrix multiplication."""

    def __init__(
        self,
        img_height: int,
        img_width: int,
        in_channel: int,
        n_class: int,
        kernel_list: List[int] = [32],
        kernel_size_list: List[int] = [3],
        pool_out_size: int = 5,
        stride_list=[1],
        padding_list=[1],
        hidden_list: List[int] = [32],
        block_list: List[int] = [8],
        in_bit: int = 32,
        w_bit: int = 32,
        mode: str = "usv",
        v_max: float = 10.8,
        v_pi: float = 4.36,
        act_thres: float = 6.0,
        photodetect: bool = True,
        bias: bool = False,
        device: Device = torch.device("cuda"),
    ) -> None:
        super().__init__()
        self.img_height = img_height
        self.img_width = img_width
        self.in_channel = in_channel
        self.n_class = n_class
        self.kernel_list = kernel_list
        self.kernel_size_list = kernel_size_list
        self.stride_list = stride_list
        self.padding_list = padding_list

        self.pool_out_size = pool_out_size

        self.hidden_list = hidden_list
        self.block_list = block_list
        self.in_bit = in_bit
        self.w_bit = w_bit
        self.mode = mode
        self.v_max = v_max
        self.v_pi = v_pi
        self.act_thres = act_thres

        self.photodetect = photodetect
        self.bias = bias

        self.device = device

        self.build_layers()
        self.drop_masks = None

        self.reset_parameters()
        self.gamma_noise_std = 0
        self.crosstalk_factor = 0

    def build_layers(self):
        self.features = OrderedDict()
        for idx, out_channel in enumerate(self.kernel_list, 0):
            layer_name = "conv" + str(idx + 1)
            in_channel = self.in_channel if (idx == 0) else self.kernel_list[idx - 1]
            self.features[layer_name] = ConvBlock(
                in_channel,
                out_channel,
                self.kernel_size_list[idx],
                self.block_list[idx],
                self.bias,
                self.stride_list[idx],
                self.padding_list[idx],
                self.mode,
                self.v_max,
                self.v_pi,
                self.w_bit,
                self.in_bit,
                self.photodetect,
                self.act_thres,
                self.device,
            )
        self.features = nn.Sequential(self.features)

        if self.pool_out_size > 0:
            self.pool2d = nn.AdaptiveAvgPool2d(self.pool_out_size)
            feature_size = self.kernel_list[-1] * self.pool_out_size * self.pool_out_size
        else:
            self.pool2d = None
            img_height, img_width = self.img_height, self.img_width
            for layer in self.modules():
                if isinstance(layer, MZIBlockConv2d):
                    img_height, img_width = layer.get_output_dim(img_height, img_width)
            feature_size = img_height * img_width * self.kernel_list[-1]

        self.classifier = OrderedDict()
        for idx, hidden_dim in enumerate(self.hidden_list, 0):
            layer_name = "fc" + str(idx + 1)
            in_channel = feature_size if idx == 0 else self.hidden_list[idx - 1]
            out_channel = hidden_dim
            self.classifier[layer_name] = LinearBlock(
                in_channel,
                out_channel,
                miniblock=self.block_list[idx + len(self.kernel_list)],
                bias=self.bias,
                mode=self.mode,
                v_max=self.v_max,
                v_pi=self.v_pi,
                in_bit=self.in_bit,
                w_bit=self.w_bit,
                photodetect=self.photodetect,
                activation=True,
                act_thres=self.act_thres,
                device=self.device,
            )

        layer_name = "fc" + str(len(self.hidden_list) + 1)
        self.classifier[layer_name] = MZIBlockLinear(
            self.hidden_list[-1] if len(self.hidden_list) > 0 else feature_size,
            self.n_class,
            miniblock=self.block_list[-1],
            bias=self.bias,
            mode=self.mode,
            v_max=self.v_max,
            v_pi=self.v_pi,
            in_bit=self.in_bit,
            w_bit=self.w_bit,
            photodetect=self.photodetect,
            device=self.device,
        )
        self.classifier = nn.Sequential(self.classifier)

    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)
        if self.pool2d is not None:
            x = self.pool2d(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x
