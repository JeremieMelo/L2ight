import os
import sys

from numpy.core.numeric import cross
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from collections import OrderedDict

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .layers import *
from pyutils import *
sys.path.pop(0)


class MZI_MLP(nn.Module):
    '''MZI MLP (Shen+, Nature Photonics 2017)
    '''
    def __init__(self,
        n_feat,
        n_output,
        hidden_list=[32],
        in_bit=32,
        w_bit=32,
        S_trainable=True,
        S_scale=3,
        mode="usv",
        v_max=10.8,
        v_pi=4.36,
        act_thres=6,
        photodetect=True,
        bias=False,
        device=torch.device("cuda")
    ):
        super().__init__()
        self.n_feat = n_feat
        self.n_output = n_output
        self.hidden_list = hidden_list
        self.in_bit = in_bit
        self.w_bit = w_bit
        self.S_trainable = S_trainable
        self.S_scale = S_scale
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

    def build_layers(self):
        self.fc_layers = OrderedDict()
        self.acts = OrderedDict()

        for idx, hidden_dim in enumerate(self.hidden_list, 0):
            layer_name = "fc" + str(idx+1)
            act_name = "fc_act" + str(idx+1)
            in_channel = self.n_feat if idx == 0 else self.hidden_list[idx-1]
            out_channel = hidden_dim
            fc = MZILinear(
                in_channel,
                out_channel,
                bias=self.bias,
                S_trainable=self.S_trainable,
                S_scale=self.S_scale,
                mode=self.mode,
                v_max=self.v_max,
                v_pi=self.v_pi,
                in_bit=self.in_bit,
                w_bit=self.w_bit,
                photodetect=self.photodetect,
                device=self.device
            )

            # activation = nn.ReLU()
            activation = ReLUN(4)

            self.fc_layers[layer_name] = fc
            self.acts[layer_name] = activation
            super().__setattr__(layer_name, fc)
            super().__setattr__(act_name, activation)

        layer_name = "fc"+str(len(self.hidden_list)+1)
        fc = MZILinear(
                self.hidden_list[-1] if len(self.hidden_list) > 0 else self.n_feat,
                self.n_output,
                bias=self.bias,
                S_trainable=self.S_trainable,
                S_scale=self.S_scale,
                mode=self.mode,
                v_max=self.v_max,
                v_pi=self.v_pi,
                in_bit=self.in_bit,
                w_bit=self.w_bit,
                photodetect=self.photodetect,
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
                "delta_list_U": layer.delta_list_U.data.clone() if layer.delta_list_U is not None else None,
                "phase_U": layer.phase_U.data.clone() if layer.phase_U is not None else None,
                "delta_list_V": layer.delta_list_V.data.clone() if layer.delta_list_V is not None else None,
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

    def get_num_device(self):
        total_mzi = 0
        for layer in self.fc_layers.values():
            total_mzi += layer.in_channel * layer.out_channel
        return {"mzi": total_mzi}

    def unitary_projection(self):
        for layer in self.fc_layers.values():
            if(layer.U is not None):
                layer.U.data.copy_(projection_matrix_to_unitary(layer.U.data))
                layer.V.data.copy_(projection_matrix_to_unitary(layer.V.data))

    def get_unitary_loss(self):
        loss_list = []
        self.unitary_loss_cache = {}
        for layer_name, layer in self.fc_layers.items():
            if(layer.U is not None and layer.U.requires_grad == True):
                if(layer_name not in self.unitary_loss_cache):
                    eye_U, eye_V = self.unitary_loss_cache[layer_name] = [torch.eye(layer.U.size(0), device=layer.U.device), torch.eye(layer.V.size(0), device=layer.V.device)]
                else:
                    eye_U, eye_V = self.unitary_loss_cache[layer_name]
                loss_list.extend([F.mse_loss(torch.matmul(layer.U, layer.U.t()), eye_U), F.mse_loss(torch.matmul(layer.V, layer.V.t()), eye_V)])

        return sum(loss_list) / len(loss_list)

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

    def sync_parameters(self, src="weight"):
        for layer in self.fc_layers.values():
            layer.sync_parameters(src=src)

    def gen_mixedtraining_mask(self, sparsity, prefer_small=False, random_state=None):
        return {layer_name: layer.gen_mixedtraining_mask(sparsity, prefer_small, random_state) for layer_name, layer in self.fc_layers.items()}

    def switch_mode_to(self, mode):
        for layer in self.fc_layers.values():
            layer.switch_mode_to(mode)

    def get_power(self, mixedtraining_mask=None):
        power = sum(layer.get_power(mixedtraining_mask[layer_name]) for layer_name, layer in self.fc_layers.items())
        return power

    def forward(self, x):
        x = x.view(-1, self.n_feat)
        n_layer = len(self.fc_layers)

        for idx, layer in enumerate(self.fc_layers):
            x = self.fc_layers[layer](x)
            if(idx < n_layer - 1):
                x = self.acts[layer](x)

        return x

if __name__ == "__main__":
    device = torch.device("cuda")

    model = MZI_MLP(
        n_feat=28,
        n_output=10,
        hidden_list=[32],
        in_bit=8,
        w_bit=8,
        S_trainable=True,
        mode="phase",
        bias=False,
        v_max=10.8,
        v_pi=4.36,
        photodetect=True,
        device=device).to(device)
    model.set_gamma_noise(1e-3)
    x = torch.randn(4, 28, dtype=torch.float, device=device)
    y = model(x)
    print(y)
