
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
from model.layers import MZILinear
sys.path.pop(0)

def test():
    device = 'cuda'
    layer = MZILinear(4,4, mode="usv", device=device)
    x = torch.randn(1, 4, device=device)
    y1 = layer(x)
    # layer.sync_parameters(src="usv")
    # layer.switch_mode_to("phase")
    # print(layer.delta_list_U, layer.delta_list_V)
    layer.set_gamma_noise(2e-5)
    y2 = layer(x)
    print(y1)
    print(y2)
    # print(layer.U, layer.S, layer.V)

if __name__ == "__main__":
    test()

