
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.types import Device, _size
from torchpack.utils.logging import logger

from .layers.activation import ReLUN
from .layers.custom_conv2d import MZIBlockConv2d
from .layers.custom_linear import MZIBlockLinear
from .sparse_bp_base import SparseBP_Base

__all__ = ["SparseBP_MZI_VGG8", "SparseBP_MZI_VGG11",
           "SparseBP_MZI_VGG13", "SparseBP_MZI_VGG16", "SparseBP_MZI_VGG19"]

cfg_32 = {
    'vgg8': [64, 'M', 128, 'M', 256, 'M', 512, 'M', 512, 'M'],
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

cfg_64 = {
    'vgg8': [64, 'M', 128, 'M', 256, 'M', 512, 'M', 512, 'GAP'],
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'GAP'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'GAP'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'GAP'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'GAP'],
}


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
        device: Device = torch.device("cuda")
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
            device=device)

        self.bn = nn.BatchNorm2d(out_channel)

        self.activation = ReLUN(act_thres, inplace=True) if act_thres <= 6 else nn.ReLU(inplace=True)


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
        device: Device = torch.device("cuda")
    ) -> None:
        super().__init__()
        self.linear = MZIBlockLinear(
            in_channel,
            out_channel,
            miniblock,
            bias,
            mode,
            v_max,
            v_pi,
            w_bit,
            in_bit,
            photodetect,
            device)

        self.activation = (ReLUN(
            act_thres, inplace=True) if act_thres <= 6 else nn.ReLU(inplace=True)) if activation else None

    def forward(self, x: Tensor) -> Tensor:
        x = self.linear(x)
        if(self.activation is not None):
            x = self.activation(x)
        return x


class VGG(SparseBP_Base):
    '''MZI VGG (Shen+, Nature Photonics 2017). Support sparse backpropagation. Blocking matrix multiplication.
    '''

    def __init__(self,
                 vgg_name: str,
                 img_height: int,
                 img_width: int,
                 in_channel: int,
                 n_class: int,
                 block_list: List[int] = [8],
                 in_bit: int = 32,
                 w_bit: int = 32,
                 mode: str = "usv",
                 v_max: float = 10.8,
                 v_pi: float = 4.36,
                 act_thres: float = 6.0,
                 photodetect: bool = True,
                 bias: bool = False,
                 device: Device = torch.device("cuda")
                 ) -> None:
        super().__init__()

        self.vgg_name = vgg_name
        self.img_height = img_height
        self.img_width = img_width
        self.in_channel = in_channel
        self.n_class = n_class

        # list of block size
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
        cfg = cfg_32 if self.img_height == 32 else cfg_64
        self.features, convNum = self._make_layers(cfg[self.vgg_name])
        # build FC layers
        ## lienar layer use the last miniblock
        if(self.img_height == 64 and self.vgg_name == "vgg8"): ## model is too small, do not use dropout
            classifier = []
        else:
            classifier = [nn.Dropout(0.5)]
        classifier += [MZIBlockLinear(
                512,
                self.n_class,
                miniblock=self.block_list[-1],
                bias=self.bias,
                mode=self.mode,
                v_max=self.v_max,
                v_pi=self.v_pi,
                in_bit=self.in_bit,
                w_bit=self.w_bit,
                photodetect=self.photodetect,
                device=self.device
            )]
        self.classifier = nn.Sequential(*classifier)

    def _make_layers(self, cfg):
        layers = []
        in_channel = self.in_channel
        convNum = 0

        for x in cfg:
            # MaxPool2d
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            elif x == "GAP":
                layers += [nn.AdaptiveAvgPool2d((1, 1))]
            else:
                # conv + BN + RELU
                layers += [ConvBlock(
                    in_channel,
                    x,
                    kernel_size=3,
                    miniblock=self.block_list[0], ## miniblock will not change, conv layer use the first miniblock
                    bias=self.bias,
                    stride=1,
                    padding=1,
                    mode=self.mode,
                    v_max=self.v_max,
                    v_pi=self.v_pi,
                    w_bit=self.w_bit,
                    in_bit=self.in_bit,
                    photodetect=self.photodetect,
                    act_thres=self.act_thres,
                    device=self.device
                )]
                in_channel = x
                convNum += 1
        # layers += [nn.AdaptiveAvgPool2d(1)] #[nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers), convNum

    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x


def SparseBP_MZI_VGG8(*args, **kwargs):
    return VGG("vgg8", *args, **kwargs)


def SparseBP_MZI_VGG11(*args, **kwargs):
    return VGG("vgg11", *args, **kwargs)


def SparseBP_MZI_VGG13(*args, **kwargs):
    return VGG("vgg13", *args, **kwargs)


def SparseBP_MZI_VGG16(*args, **kwargs):
    return VGG("vgg16", *args, **kwargs)


def SparseBP_MZI_VGG19(*args, **kwargs):
    return VGG("vgg19", *args, **kwargs)


def test():
    device = torch.device("cuda")
    net = SparseBP_MZI_VGG8(
        32,
        32,
        3,
        10,
        [4, 4, 4, 4, 4, 4, 4, 4],
        32, 32,
        mode='usv',
        v_max=10.8,
        v_pi=4.36,
        act_thres=6.0,
        photodetect=True,
        bias=False,
        device=device
    ).to(device)

    x = torch.randn(2, 3, 32, 32).to(device)
    print(net)
    y = net(x)
    print(y.shape)


if __name__ == "__main__":
    test()
