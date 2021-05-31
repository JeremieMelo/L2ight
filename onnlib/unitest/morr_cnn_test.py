
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
from model import MORR_CLASS_CNN
from model import MORRConfig_20um_MQ
sys.path.pop(0)

def test():
    device = 'cuda'
    model = model = MORR_CLASS_CNN(
        img_height=8,
        img_width=8,
        in_channel=8,
        n_class=10,
        kernel_list=[16, 32],
        kernel_size_list=[3, 3],
        block_list=[4, 4, 4],
        stride_list=[1, 2],
        padding_list=[1, 1],
        pool_out_size=3,
        hidden_list=[],
        in_bit=16,
        w_bit=16,
        mode='weight',
        bias=False,
        ### morr configuartion
        MORRConfig=MORRConfig_20um_MQ,
        trainable_morr_bias=False,
        trainable_morr_scale=False,
        device=device
    ).to(device)

    model.reset_parameters()
    x = torch.randn(1, 8, 8, 8, device=device)

    y1 = model(x)
    print(y1)
    # print(layer.U, layer.S, layer.V)

if __name__ == "__main__":
    test()

