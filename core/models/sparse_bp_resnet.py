from torch.nn.modules.activation import ReLU
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

__all__ = ["SparseBP_MZI_ResNet18", "SparseBP_MZI_ResNet34",
           "SparseBP_MZI_ResNet50", "SparseBP_MZI_ResNet101", "SparseBP_MZI_ResNet152"]


def conv3x3(in_planes, out_planes, miniblock: int = 8,
            bias: bool = False,
            stride: Union[int, _size] = 1,
            padding: Union[int, _size] = 0,
            mode: str = "weight",
            v_max: float = 4.36,  # 0-pi for clements, # 6.166 is v_2pi, 0-2pi for reck
            v_pi: float = 4.36,
            w_bit: int = 16,
            in_bit: int = 16,
            photodetect: bool = False,
            device: Device = torch.device("cuda")):

    conv = MZIBlockConv2d(
        in_planes,
        out_planes,
        3,
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

    return conv


def conv1x1(in_planes, out_planes, miniblock: int = 8,
            bias: bool = False,
            stride: Union[int, _size] = 1,
            padding: Union[int, _size] = 0,
            mode: str = "weight",
            v_max: float = 4.36,  # 0-pi for clements, # 6.166 is v_2pi, 0-2pi for reck
            v_pi: float = 4.36,
            w_bit: int = 16,
            in_bit: int = 16,
            photodetect: bool = False,
            device: Device = torch.device("cuda")):

    conv = MZIBlockConv2d(
        in_planes,
        out_planes,
        1,
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

    return conv


def Linear(in_channel, out_channel, miniblock: int = 8,
           bias: bool = False,
           mode: str = "weight",
           v_max: float = 4.36,  # 0-pi for clements, # 6.166 is v_2pi, 0-2pi for reck
           v_pi: float = 4.36,
           w_bit: int = 16,
           in_bit: int = 16,
           photodetect: bool = False,
           device: Device = torch.device("cuda")
           ):
    # linear = nn.Linear(in_channel, out_channel)
    linear = MZIBlockLinear(
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
        device=device)
    return linear


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1,
                 # unique parameters
                 miniblock: int = 8,
                 mode: str = "weight",
                 v_max: float = 4.36,  # 0-pi for clements, # 6.166 is v_2pi, 0-2pi for reck
                 v_pi: float = 4.36,
                 w_bit: int = 16,
                 in_bit: int = 16,
                 photodetect: bool = False,
                 act_thres: int = 6,
                 device: Device = torch.device("cuda")
                 ) -> None:
        super(BasicBlock, self).__init__()
        # self.conv1 = nn.Conv2d(
        #     in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv1 = conv3x3(in_planes, planes,
                             miniblock=miniblock,
                             bias=False,
                             stride=stride,
                             padding=1,
                             mode=mode,
                             v_max=v_max,
                             v_pi=v_pi,
                             in_bit=in_bit,
                             w_bit=w_bit,
                             photodetect=photodetect,
                             device=device)
        self.bn1 = nn.BatchNorm2d(planes)
        self.act1 = ReLUN(act_thres, inplace=True) if act_thres <= 6 else ReLU(inplace=True)
        # self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
        #                        stride=1, padding=1, bias=False)
        self.conv2 = conv3x3(planes, planes,
                             miniblock=miniblock,
                             bias=False,
                             stride=1,
                             padding=1,
                             mode=mode,
                             v_max=v_max,
                             v_pi=v_pi,
                             in_bit=in_bit,
                             w_bit=w_bit,
                             photodetect=photodetect,
                             device=device)
        self.bn2 = nn.BatchNorm2d(planes)
        self.act2 = ReLUN(act_thres, inplace=True) if act_thres <= 6 else ReLU(inplace=True)

        self.shortcut = nn.Identity()
        # self.shortcut.conv1_spatial_sparsity = self.conv1.bp_input_sampler.spatial_sparsity
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                conv1x1(in_planes, self.expansion*planes,
                        miniblock=miniblock,
                        bias=False,
                        stride=stride,
                        padding=0,
                        mode=mode,
                        v_max=v_max,
                        v_pi=v_pi,
                        in_bit=in_bit,
                        w_bit=w_bit,
                        photodetect=photodetect,
                        device=device),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.act2(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes: int, planes: int, stride: int = 1,
                 # unique parameters
                 miniblock: int = 8,
                 mode: str = "weight",
                 v_max: float = 4.36,  # 0-pi for clements, # 6.166 is v_2pi, 0-2pi for reck
                 v_pi: float = 4.36,
                 w_bit: int = 16,
                 in_bit: int = 16,
                 photodetect: bool = False,
                 act_thres: int = 6,
                 device: Device = torch.device("cuda")
                 ) -> None:
        super(Bottleneck, self).__init__()
        # self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.conv1 = conv1x1(in_planes, planes,
                             miniblock=miniblock,
                             bias=False,
                             stride=1,
                             padding=0,
                             mode=mode,
                             v_max=v_max,
                             v_pi=v_pi,
                             in_bit=in_bit,
                             w_bit=w_bit,
                             photodetect=photodetect,
                             device=device)
        self.bn1 = nn.BatchNorm2d(planes)
        self.act1 = ReLUN(act_thres, inplace=True) if act_thres <= 6 else ReLU(inplace=True)
        # self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv2 = conv3x3(planes, planes,
                             miniblock=miniblock,
                             bias=False,
                             stride=stride,
                             padding=1,
                             mode=mode,
                             v_max=v_max,
                             v_pi=v_pi,
                             in_bit=in_bit,
                             w_bit=w_bit,
                             photodetect=photodetect,
                             device=device)
        self.bn2 = nn.BatchNorm2d(planes)
        self.act2 = ReLUN(act_thres, inplace=True) if act_thres <= 6 else ReLU(inplace=True)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.conv3 = conv1x1(planes, self.expansion*planes,
                             miniblock=miniblock,
                             bias=False,
                             stride=1,
                             padding=0,
                             mode=mode,
                             v_max=v_max,
                             v_pi=v_pi,
                             in_bit=in_bit,
                             w_bit=w_bit,
                             photodetect=photodetect,
                             device=device)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)
        self.act3 = ReLUN(act_thres, inplace=True) if act_thres <= 6 else ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                # nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                conv1x1(in_planes, self.expansion*planes,
                        miniblock=miniblock,
                        bias=False,
                        stride=stride,
                        padding=0,
                        mode=mode,
                        v_max=v_max,
                        v_pi=v_pi,
                        in_bit=in_bit,
                        w_bit=w_bit,
                        photodetect=photodetect,
                        device=device),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.act2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = self.act3(out)
        return out


class ResNet(SparseBP_Base):
    '''MZI ResNet (Shen+, Nature Photonics 2017). Support sparse backpropagation. Blocking matrix multiplication.
    '''

    def __init__(self, block, num_blocks,
                 img_height : int,
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

        # resnet params
        self.block = block
        self.num_blocks = num_blocks
        self.in_planes = 64
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

        self.device = device

        # build layers
        blkIdx = 0
        self.conv1 = conv3x3(in_channel, 64,
                             miniblock=self.block_list[0],
                             bias=False,
                             stride=1 if img_height <= 64 else 2,#downsample for imagenet, dogs, cars
                             padding=1,
                             mode=mode,
                             v_max=self.v_max,
                             v_pi=self.v_pi,
                             in_bit=self.in_bit,
                             w_bit=self.w_bit,
                             photodetect=self.photodetect,
                             device=self.device)
        self.bn1 = nn.BatchNorm2d(64)
        blkIdx += 1

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1,
                                       miniblock=self.block_list[0],
                                       mode=self.mode,
                                       v_max=self.v_max,
                                       v_pi=self.v_pi,
                                       in_bit=self.in_bit,
                                       w_bit=self.w_bit,
                                       photodetect=self.photodetect,
                                       device=device)
        blkIdx += 1

        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2,
                                       miniblock=self.block_list[0],
                                       mode=self.mode,
                                       v_max=self.v_max,
                                       v_pi=self.v_pi,
                                       in_bit=self.in_bit,
                                       w_bit=self.w_bit,
                                       photodetect=self.photodetect,
                                       device=device)
        blkIdx += 1

        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2,
                                       miniblock=self.block_list[0],
                                       mode=self.mode,
                                       v_max=self.v_max,
                                       v_pi=self.v_pi,
                                       in_bit=self.in_bit,
                                       w_bit=self.w_bit,
                                       photodetect=self.photodetect,
                                       device=device)
        blkIdx += 1

        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2,
                                       miniblock=self.block_list[0],
                                       mode=self.mode,
                                       v_max=self.v_max,
                                       v_pi=self.v_pi,
                                       in_bit=self.in_bit,
                                       w_bit=self.w_bit,
                                       photodetect=self.photodetect,
                                       device=device)
        blkIdx += 1

        self.linear = Linear(512 * block.expansion,
                             self.n_class,
                             miniblock=self.block_list[0],
                             bias=False,
                             mode=self.mode,
                             v_max=self.v_max,
                             v_pi=self.v_pi,
                             in_bit=self.in_bit,
                             w_bit=self.w_bit,
                             photodetect=self.photodetect,
                             device=device)

        self.drop_masks = None

        self.reset_parameters()
        self.gamma_noise_std = 0
        self.crosstalk_factor = 0

    def _make_layer(self, block, planes, num_blocks, stride,
                    # unique parameters
                    miniblock: int = 8,
                    mode: str = "usv",
                    v_max: float = 10.8,
                    v_pi: float = 4.36,
                    in_bit: int = 32,
                    w_bit: int = 32,
                    act_thres: float = 6.0,
                    photodetect: bool = True,
                    device: Device = torch.device("cuda")):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride,
                                miniblock=miniblock,
                                mode=mode,
                                v_max=v_max,
                                v_pi=v_pi,
                                in_bit=in_bit,
                                w_bit=w_bit,
                                act_thres=act_thres,
                                photodetect=photodetect,
                                device=device
                                ))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        if(x.size(-1) > 64): # 224 x 224, e.g., cars, dogs, imagenet
            out = F.max_pool2d(out, kernel_size=3, stride=2, padding=1)
        out = self.layer1(out)
        out = self.layer2(out)

        out = self.layer3(out)
        out = self.layer4(out)

        out = F.adaptive_avg_pool2d(out, 1)
        out = torch.flatten(out, 1)
        out = self.linear(out)
        return out


def SparseBP_MZI_ResNet18(*args, **kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], *args, **kwargs)


def SparseBP_MZI_ResNet34(*args, **kwargs):
    return ResNet(BasicBlock, [3, 4, 6, 3], *args, **kwargs)


def SparseBP_MZI_ResNet50(*args, **kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], *args, **kwargs)


def SparseBP_MZI_ResNet101(*args, **kwargs):
    return ResNet(Bottleneck, [3, 4, 23, 3], *args, **kwargs)


def SparseBP_MZI_ResNet152(*args, **kwargs):
    return ResNet(Bottleneck, [3, 8, 36, 3], *args, **kwargs)


def test():
    device = torch.device("cuda")
    net = SparseBP_MZI_ResNet18(
        in_channel=3,
        n_class=10,
        block_list=[8, 8, 8, 8, 8, 8],
        in_bit=32,
        w_bit=32,
        mode='usv',
        v_max=10.8,
        v_pi=4.36,
        act_thres=6,
        photodetect=True,
        device=device
    ).to(device)

    x = torch.randn(2, 3, 32, 32).to(device)
    print(net)
    y = net(x)
    print(y.shape)


if __name__ == "__main__":
    test()
