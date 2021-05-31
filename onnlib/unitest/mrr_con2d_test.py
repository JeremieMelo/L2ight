
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
from model.layers import AddDropMRRConv2d
sys.path.pop(0)

def test():
    device = 'cuda'
    layer = AddDropMRRConv2d(
        32,
        64,
        3,
        stride=1,
        padding=1,
        bias=False,
        miniblock=4,
        mode="weight",
        in_bit=16,
        w_bit=8,
        device=device
    )
    layer.reset_parameters()
    layer.enable_dynamic_weight(2,44, relu=False, nonlinear=False)
    print(layer.basis.size(), layer.coeff_in.size())
    layer.assign_separate_weight_bit(6,5,4)
    print(layer.get_param_size(fullrank=False))
    gt1 = (44 * 2 * 3**2 * 6 + 44 * 32 * 2 * 5 + 64 * 44 * 4) / 8
    gt2 = (32 * 64 * 3 **2 ) * 4
    print(gt1, gt2)
    print(layer.get_param_size(fullrank=True))
    x = torch.randn(2, 64, 8, 8, device=device)

    # y1 = layer(x)
    # print(y1.size())
    # print(layer.U, layer.S, layer.V)

if __name__ == "__main__":
    test()

