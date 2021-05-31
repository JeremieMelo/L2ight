from collections import OrderedDict
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor, nn
from torch.types import Device
from torchpack.utils.logging import logger

from .layers.activation import ReLUN
from .layers.custom_linear import MZIBlockLinear, SparseBPLinear
from .sparse_bp_base import SparseBP_Base

__all__ = ["SparseBP_MZI_MLP"]


class SparseBP_MLP(SparseBP_Base):
    def __init__(self, n_feat, n_class, device=torch.device("cuda")) -> None:
        super().__init__()
        self.n_feat = n_feat
        self.n_class = n_class
        self.device = device
        self.fc1 = SparseBPLinear(
            n_feat, 64, bias=False, miniblock=8, device=device)
        self.fc2 = SparseBPLinear(
            64, 64, bias=False, miniblock=8, device=device)
        self.fc3 = SparseBPLinear(
            64, 10, bias=False, miniblock=8, device=device)
        self.layers = {"fc1": self.fc1, "fc2": self.fc2, "fc3": self.fc3}
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.layers.values():
            layer.reset_parameters()

    def gen_bp_mask(self, bp_sparsity=None):
        for layer in self.layers.values():
            layer.gen_deterministic_gradient_mask(bp_sparsity)

    def set_bp_rank(self, bp_rank):
        for layer in self.layers.values():
            layer.set_bp_rank(bp_rank)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        n_layer = len(self.layers)
        for i, layer in enumerate(self.layers.values()):
            x = layer(x)
            if(i < n_layer - 1):
                x = torch.relu(x)

        return x


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

        self.activation = ReLUN(
            act_thres, inplace=True) if activation else None

    def forward(self, x: Tensor) -> Tensor:
        x = self.linear(x)
        if(self.activation is not None):
            x = self.activation(x)
        return x


class SparseBP_MZI_MLP(torch.nn.Module):
    '''MZI MLP (Shen+, Nature Photonics 2017). Support sparse backpropagation. Blocking matrix multiplication.
    '''

    def __init__(self,
                 n_feat: int,
                 n_class: int,
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
                 device: Device = torch.device("cuda")
                 ) -> None:
        super().__init__()
        self.n_feat = n_feat
        self.n_class = n_class
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

    def build_layers(self) -> None:
        self.classifier = OrderedDict()

        for idx, hidden_dim in enumerate(self.hidden_list, 0):
            layer_name = "fc" + str(idx+1)
            in_channel = self.n_feat if idx == 0 else self.hidden_list[idx-1]
            out_channel = hidden_dim
            self.classifier[layer_name] = LinearBlock(
                in_channel,
                out_channel,
                miniblock=self.block_list[idx],
                bias=self.bias,
                mode=self.mode,
                v_max=self.v_max,
                v_pi=self.v_pi,
                in_bit=self.in_bit,
                w_bit=self.w_bit,
                photodetect=self.photodetect,
                activation=True,
                act_thres=self.act_thres,
                device=self.device
            )

        layer_name = "fc"+str(len(self.hidden_list)+1)
        self.classifier[layer_name] = MZIBlockLinear(
            self.hidden_list[-1] if len(self.hidden_list) > 0 else self.n_feat,
            self.n_class,
            miniblock=self.block_list[idx],
            bias=self.bias,
            mode=self.mode,
            v_max=self.v_max,
            v_pi=self.v_pi,
            in_bit=self.in_bit,
            w_bit=self.w_bit,
            photodetect=self.photodetect,
            device=self.device
        )
        self.classifier = nn.Sequential(self.classifier)

    def forward(self, x: Tensor) -> Tensor:
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
