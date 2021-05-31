import os
import sys

from numpy.core.numeric import cross
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from collections import OrderedDict

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

if __name__ == "__main__":
    from layers import *
else:
    from .layers import *
from pyutils import *
sys.path.pop(0)


class FFTONN_MLP(nn.Module):
    def __init__(self,
        n_feat,
        n_output,
        block_list=[4, 4],
        hidden_list=[32],
        in_bit=32,
        w_bit=32,
        S_scale=1,
        mode="phase",
        v_max=10.8,
        v_pi=4.36,
        act_thres=6,
        photodetect=True,
        transform="trainable",
        bias=False,
        device=torch.device("cuda")
    ):
        super().__init__()
        self.n_feat = n_feat
        self.n_output = n_output
        self.block_list = block_list
        self.hidden_list = hidden_list
        self.in_bit = in_bit
        self.w_bit = w_bit
        self.S_scale = S_scale
        self.mode = mode
        self.v_max = v_max
        self.v_pi = v_pi
        self.act_thres = act_thres

        self.photodetect = photodetect
        self.transform = transform
        self.bias = bias

        self.device = device

        self.build_layers()
        self.drop_masks = None

        self.reset_parameters()

    def build_layers(self):
        self.fc_layers = OrderedDict()
        self.acts = OrderedDict()

        for idx, hidden_dim in enumerate(self.hidden_list, 0):
            layer_name = "fc" + str(idx+1)
            act_name = "fc_act" + str(idx+1)
            in_channel = self.n_feat if idx == 0 else self.hidden_list[idx-1]
            out_channel = hidden_dim
            fc = FOLinear(
                in_channel,
                out_channel,
                mini_block=self.block_list[idx],
                bias=self.bias,
                S_scale=self.S_scale,
                mode=self.mode,
                v_max=self.v_max,
                v_pi=self.v_pi,
                in_bit=self.in_bit,
                w_bit=self.w_bit,
                photodetect=self.photodetect,
                transform=self.transform,
                device=self.device
            )

            activation = ReLUN(4)

            self.fc_layers[layer_name] = fc
            self.acts[layer_name] = activation
            super().__setattr__(layer_name, fc)
            super().__setattr__(act_name, activation)

        layer_name = "fc"+str(len(self.hidden_list)+1)
        fc = FOLinear(
                self.hidden_list[-1] if len(self.hidden_list) > 0 else self.n_feat,
                self.n_output,
                self.block_list[-1],
                bias=self.bias,
                S_scale=self.S_scale,
                mode=self.mode,
                v_max=self.v_max,
                v_pi=self.v_pi,
                in_bit=self.in_bit,
                w_bit=self.w_bit,
                photodetect=self.photodetect,
                transform=self.transform,
                device=self.device
            )
        super().__setattr__(layer_name, fc)
        self.fc_layers[layer_name] = fc
        ### dict that stores all layers
        self.layers = OrderedDict()
        self.layers.update(self.fc_layers)
        # self.layers.update(self.acts)

    def reset_parameters(self):
        for layer in self.fc_layers:
            self.fc_layers[layer].reset_parameters()

    def backup_phases(self):
        self.phase_backup = {}
        for layer_name, layer in self.fc_layers.items():
            self.phase_backup[layer_name] = {
                "weight": layer.weight.data.clone() if layer.weight is not None else None,
                "U":layer.U.data.clone() if layer.U is not None else None,
                "S":layer.S.data.clone() if layer.S is not None else None,
                "V":layer.V.data.clone() if layer.V is not None else None,
                "phase_U": layer.phase_U.data.clone() if layer.phase_U is not None else None,
                "phase_S": layer.phase_S.data.clone() if layer.phase_S is not None else None,
                "phase_V": layer.phase_V.data.clone() if layer.phase_V is not None else None
                }

    def restore_phases(self):
        for layer_name, layer in self.fc_layers.items():
            backup = self.phase_backup[layer_name]
            for param_name, param_src in backup.items():
                param_dst = getattr(layer, param_name)
                if(param_src is not None and param_dst is not None):
                    param_dst.data.copy_(param_src.data)

    def set_gamma_noise(self, noise_std=0, random_state=None):
        for layer in self.fc_layers.values():
            layer.set_gamma_noise(noise_std, random_state=random_state)

    def set_crosstalk_factor(self, crosstalk_factor=0):
        for layer in self.fc_layers.values():
            layer.set_crosstalk_factor(crosstalk_factor)

    def get_unitary_loss(self):
        return torch.zeros(1, device=self.device)

    def unitary_projection(self):
        pass

    def get_num_device(self):
        total_dc = 0
        total_ps = 0
        for layer in self.fc_layers.values():
            total_dc += (layer.mini_block // 2) * int(np.log2(layer.mini_block)) * 2
            total_ps += layer.in_channel_pad * layer.out_channel_pad * ((int(np.log2(layer.mini_block)) + 2) * layer.mini_block * 2)
        return {"dc": total_dc, "ps": total_ps}

    def load_parameters(self, param_dict):
        '''
        description: update parameters based on this parameter dictionary\\
        param param_dict {dict of dict} {layer_name: {param_name: param_tensor, ...}, ...}
        '''
        for layer_name, layer_param_dict in param_dict.items():
            self.layers[layer_name].load_parameters(layer_param_dict)

    def build_obj_fn(self, X, y, criterion):
        def obj_fn(X_cur=None, y_cur=None, param_dict=None):
            if(param_dict is not None):
                self.load_parameters(param_dict)
            if(X_cur is None or y_cur is None):
                data, target = X, y
            else:
                data, target = X_cur, y_cur
            pred = self.forward(data)
            return criterion(pred, target)
        return obj_fn

    def enable_fast_forward(self):
        for layer in self.fc_layers.values():
            layer.enable_fast_forward()

    def disable_fast_forward(self):
        for layer in self.fc_layers.values():
            layer.disable_fast_forward()

    def sync_parameters(self, src="phase"):
        for layer in self.fc_layers.values():
            layer.sync_parameters(src=src)

    def gen_mixedtraining_mask(self, sparsity, random_state=None):
        return {layer_name: layer.gen_mixedtraining_mask(sparsity, random_state) for layer_name, layer in self.fc_layers.items()}

    def switch_mode_to(self, mode):
        for layer in self.fc_layers.values():
            layer.switch_mode_to(mode)

    def get_power(self, mixedtraining_mask=None):
        power = sum(layer.get_power(mixedtraining_mask[layer_name]) for layer_name, layer in self.fc_layers.items())
        return power

    def forward(self, x):
        x = x.view(-1, self.n_feat)
        n_layer = len(self.fc_layers)
        x = real_to_complex(x)

        for idx, layer in enumerate(self.fc_layers):
            x = self.fc_layers[layer](x)
            if(idx < n_layer - 1):
                x = self.acts[layer](x)
        x = get_complex_energy(x)
        return x

if __name__ == "__main__":
    device = torch.device("cuda")

    model = FFTONN_MLP(
        n_feat=28,
        n_output=10,
        block_list=[4, 4],
        hidden_list=[32],
        in_bit=8,
        w_bit=8,
        mode="phase",
        S_scale=1,
        bias=False,
        v_max=10.8,
        v_pi=4.36,
        photodetect=False,
        transform="trainable",
        device=device).to(device)
    model.train()
    x = torch.randn(4, 28, dtype=torch.float32, device=device)
    y = model(x)
    print(model.layers["fc1"].weight)
    print(y)
    model.set_gamma_noise(1e-3)
    y = model(x)
    print(y)
