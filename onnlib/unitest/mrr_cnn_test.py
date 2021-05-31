
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
from model import AddDropMRR_CLASS_CNN_WG
sys.path.pop(0)

def test():
    device = 'cuda'
    model = model = AddDropMRR_CLASS_CNN_WG(
        img_height=8,
        img_width=8,
        in_channel=8,
        n_class=10,
        kernel_list=[16, 32],
        kernel_size_list=[3, 3],
        block_list=[],
        stride_list=[1, 2],
        padding_list=[1, 1],
        pool_out_size=3,
        hidden_list=[],
        in_bit=16,
        w_bit=16,
        mode='weight',
        bias=True,
        device=device
    ).to(device)

    model.enable_dynamic_weight(2,2)
    print(model.get_num_params(fullrank=False))
    print(model.get_num_params(fullrank=True))
    x = torch.randn(1, 8, 8, 8, device=device)

    y1 = model(x)
    # print(y1)
    # print(layer.U, layer.S, layer.V)

if __name__ == "__main__":
    test()

