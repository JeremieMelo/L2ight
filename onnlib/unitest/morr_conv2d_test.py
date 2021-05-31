
import logging
from pyutils.general import TimerCtx
from model.layers.device.mrr import MORRConfig_20um_MQ
import os
import sys

from torch import random
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Parameter, init
from model.layers import AllPassMORRCirculantConv2d
from pyutils import print_stat
sys.path.pop(0)

def test():
    device = 'cuda'
    layer = AllPassMORRCirculantConv2d(
        in_channel=64,
        out_channel=64,
        kernel_size=3,
        stride=1,
        padding=1,
        miniblock=4,
        bias=False,
        mode="weight",
        v_max=10.8,
        v_pi=4.36,
        w_bit=16,
        in_bit=16,
        MORRConfig=MORRConfig_20um_MQ,
        trainable_morr_bias=True,
        trainable_morr_scale=False,
        device=torch.device("cuda")
        )
    layer.reset_parameters()
    x = torch.randn(3, 64, 16, 16, device=device)
    c = 10
    torch.cuda.synchronize()
    with TimerCtx() as t:
        for _ in range(c):
            y1 = layer.forward_slow(x)

    torch.cuda.synchronize()
    print(f"FFT: {t.interval/c} s")
    # print(y1)
    torch.cuda.synchronize()
    with TimerCtx() as t:
        for _ in range(c):
            y2 = layer.forward(x)
    torch.cuda.synchronize()
    print(f"Matmul: {t.interval/c} s")
    print_stat(y1)
    print_stat(y2)


if __name__ == "__main__":
    test()

