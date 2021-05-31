
import logging
import os
import sys

from torch import random
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Parameter, init
from model.layers import MORRSeparableConv2d
sys.path.pop(0)

def test():
    device = 'cuda'
    layer = MORRSeparableConv2d(
        in_channel=4,
        out_channel=4,
        kernel_size=3,
        stride=1,
        padding=1,
        mini_block=4,
        bias=False,
        S_trainable=True,
        S_scale=3,
        mode="weight",
        v_max=10.8,
        v_pi=4.36,
        w_bit=16,
        in_bit=16,
        photodetect=True,
        mrr_a=0.8682,
        mrr_r=0.8602,
        ### waveguide parameters
        wg_gap=10, ## waveguide length (um) between crossings
        n_eff=2.35, ## effective index
        n_g=4, ## group index
        lambda_0=1550, ## center wavelength (nm)
        delta_lambda=5, ## delta wavelength (nm)
        ### crossbar parameters
        max_col=16, ## maximum number of columns
        max_row=16, ## maximum number of rows
        device=torch.device("cuda")
        )
    x = torch.randn(3, 4, 16, 16, device=device)
    y1 = layer(x)
    print(y1)


if __name__ == "__main__":
    test()

