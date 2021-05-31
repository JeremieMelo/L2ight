import os
import sys
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


class MZI_CLASS_CNN(nn.Module):
    '''MZI CNN for classification
    '''
    def __init__(self,
        img_height,
        img_width,
        in_channel,
        n_class,
        kernel_list=[16],
        kernel_size_list=[3],
        stride_list=[1],
        padding_list=[1],
        pool_out_size=5,
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
        self.img_height = img_height
        self.img_width = img_width
        self.kernel_list = kernel_list
        self.kernel_size_list = kernel_size_list
        self.hidden_list = hidden_list
        self.stride_list = stride_list
        self.padding_list = padding_list

        self.pool_out_size=pool_out_size
        self.in_bit = in_bit
        self.w_bit = w_bit
        self.S_trainable = S_trainable
        self.S_scale = S_scale
        self.mode = mode
        self.v_max = v_max
        self.v_pi = v_pi
        self.in_channel = in_channel
        self.n_class = n_class
        self.act_thres = act_thres

        self.photodetect = photodetect
        self.bias = bias

        self.device = device

        self.build_layers()
        self.drop_masks = None

        self.reset_parameters()

    def build_layers(self):
        self.conv_layers = OrderedDict()
        self.bn_layers = OrderedDict()
        self.fc_layers = OrderedDict()
        self.acts = OrderedDict()
        for idx, out_channel in enumerate(self.kernel_list, 0):
            layer_name = "conv" + str(idx+1)
            bn_name = "conv_bn" + str(idx+1)
            act_name = "conv_act" + str(idx+1)
            in_channel = self.in_channel if(idx == 0) else self.kernel_list[idx-1]
            conv = MZIConv2d(
                            in_channel,
                            out_channel,
                            self.kernel_size_list[idx],
                            stride=self.stride_list[idx],
                            padding=self.padding_list[idx],
                            S_trainable=self.S_trainable,
                            S_scale=self.S_scale,
                            mode=self.mode,
                            v_max=self.v_max,
                            v_pi=self.v_pi,
                            in_bit=self.in_bit,
                            w_bit=self.w_bit,
                            bias=self.bias,
                            photodetect=self.photodetect,
                            device=self.device)

            bn = nn.BatchNorm2d(out_channel)

            activation = nn.ReLU()

            self.conv_layers[layer_name] = conv
            self.bn_layers[layer_name] = bn
            self.acts[layer_name] = activation
            super().__setattr__(layer_name, conv)
            super().__setattr__(bn_name, bn)
            super().__setattr__(act_name, activation)

        if(self.pool_out_size > 0):
            self.pool2d = nn.AdaptiveAvgPool2d(self.pool_out_size)
            feature_size = self.kernel_list[-1] * self.pool_out_size * self.pool_out_size
        else:
            self.pool2d = None
            img_height, img_width = self.img_height, self.img_width
            for layer in self.conv_layers:
                img_height, img_width = self.conv_layers[layer].get_output_dim(img_height, img_width)
            feature_size = img_height * img_width * self.kernel_list[-1]

        for idx, hidden_dim in enumerate(self.hidden_list, 0):
            layer_name = "fc" + str(idx+1)
            bn_name = "fc_bn" + str(idx+1)
            act_name = "fc_act" + str(idx+1)
            in_channel = feature_size if idx == 0 else self.hidden_list[idx-1]
            out_channel = hidden_dim
            # fc = nn.Linear(in_channel, out_channel, bias=False)
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

            # bn = nn.BatchNorm1d(out_channel)
            activation = nn.ReLU()

            self.fc_layers[layer_name] = fc
            # self.bn_layers[layer_name] = bn
            self.acts[layer_name] = activation
            super().__setattr__(layer_name, fc)
            # super().__setattr__(bn_name, bn)
            super().__setattr__(act_name, activation)

        layer_name = "fc"+str(len(self.hidden_list)+1)
        fc = MZILinear(
                self.hidden_list[-1] if len(self.hidden_list) > 0 else feature_size,
                self.n_class,
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
        self.layers.update(self.conv_layers)
        # self.layers.update(self.bn_layers)
        self.layers.update(self.fc_layers)
        # self.layers.update(self.acts)

    def reset_parameters(self):
        for layer in self.conv_layers:
            self.conv_layers[layer].reset_parameters()
        for layer in self.fc_layers:
            self.fc_layers[layer].reset_parameters()

    def backup_phases(self):
        self.phase_backup = {}
        for layer_name, layer in self.conv_layers.items():
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
        for layer_name, layer in self.conv_layers.items():
            backup = self.phase_backup[layer_name]
            for param_name, param_src in backup.items():
                param_dst = getattr(layer, param_name)
                if(param_src is not None and param_dst is not None):
                    param_dst.data.copy_(param_src.data)
        for layer_name, layer in self.fc_layers.items():
            backup = self.phase_backup[layer_name]
            for param_name, param_src in backup.items():
                param_dst = getattr(layer, param_name)
                if(param_src is not None and param_dst is not None):
                    param_dst.data.copy_(param_src.data)

    def set_gamma_noise(self, noise_std=0, random_state=None):
        for layer in self.fc_layers.values():
            layer.set_gamma_noise(noise_std, random_state=random_state)
        for layer in self.conv_layers.values():
            layer.set_gamma_noise(noise_std, random_state=random_state)

    def set_crosstalk_factor(self, crosstalk_factor=0):
        for layer in self.fc_layers.values():
            layer.set_crosstalk_factor(crosstalk_factor)
        for layer in self.conv_layers.values():
            layer.set_crosstalk_factor(crosstalk_factor)

    def get_num_device(self):
        total_mzi = 0
        for layer in self.conv_layers.values():
            total_mzi += layer.in_channel * layer.out_channel * layer.kernel_size**2
        for layer in self.fc_layers.values():
            total_mzi += layer.in_channel * layer.out_channel
        return {"mzi": total_mzi}

    def unitary_projection(self):
        for layer in self.conv_layers.values():
            if(layer.U is not None):
                layer.U.data.copy_(projection_matrix_to_unitary(layer.U.data))
                layer.V.data.copy_(projection_matrix_to_unitary(layer.V.data))
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
        for layer_name, layer in self.conv_layers.items():
            if(layer.U is not None and layer.U.requires_grad == True):
                if(layer_name not in self.unitary_loss_cache):
                    eye_U, eye_V = self.unitary_loss_cache[layer_name] = [torch.eye(layer.U.size(0), device=layer.U.device), torch.eye(layer.V.size(0), device=layer.V.device)]
                else:
                    eye_U, eye_V = self.unitary_loss_cache[layer_name]
                loss_list.extend([F.mse_loss(torch.matmul(layer.U, layer.U.t()), eye_U), F.mse_loss(torch.matmul(layer.V, layer.V.t()), eye_V)])

        return sum(loss_list) / len(loss_list)

    def load_parameters(self, param_dict):
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
        for layer in self.conv_layers.values():
            layer.enable_fast_forward()
        for layer in self.fc_layers.values():
            layer.enable_fast_forward()

    def disable_fast_forward(self):
        for layer in self.conv_layers.values():
            layer.disable_fast_forward()
        for layer in self.fc_layers.values():
            layer.disable_fast_forward()

    def sync_parameters(self, src="weight"):
        for layer in self.fc_layers.values():
            layer.sync_parameters(src=src)
        for layer in self.conv_layers.values():
            layer.sync_parameters(src=src)

    def gen_mixedtraining_mask(self, sparsity, prefer_small=False, random_state=None):
        masks = {layer_name: layer.gen_mixedtraining_mask(sparsity, prefer_small, random_state) for layer_name, layer in self.fc_layers.items()}
        masks.update({layer_name: layer.gen_mixedtraining_mask(sparsity, prefer_small, random_state) for layer_name, layer in self.conv_layers.items()})
        return masks

    def switch_mode_to(self, mode):
        for layer in self.fc_layers.values():
            layer.switch_mode_to(mode)
        for layer in self.conv_layers.values():
            layer.switch_mode_to(mode)

    def get_power(self, mixedtraining_mask=None):
        power = sum(layer.get_power(mixedtraining_mask[layer_name]) for layer_name, layer in self.fc_layers.items())
        power += sum(layer.get_power(mixedtraining_mask[layer_name]) for layer_name, layer in self.conv_layers.items())
        return power

    def forward(self, x):
        for idx, layer in enumerate(self.conv_layers):
            x = self.conv_layers[layer](x)
            x = self.bn_layers[layer](x)
            x = self.acts[layer](x)

        if(self.pool2d is not None):
            x = self.pool2d(x)

        x = x.view(x.size(0), -1)
        n_layer = len(self.fc_layers)

        for idx, layer in enumerate(self.fc_layers):
            x = self.fc_layers[layer](x)
            if(idx < n_layer - 1):
                # x = self.bn_layers[layer](x)
                x = self.acts[layer](x)

        return x

if __name__ == "__main__":
    device = torch.device("cuda")

    model = MZI_CLASS_CNN(
        img_height=28,
        img_width=28,
        in_channel=1,
        n_class=10,
        kernel_list=[8],
        kernel_size_list=[3],
        stride_list=[1],
        padding_list=[1],
        pool_out_size=5,
        hidden_list=[32],
        in_bit=8,
        w_bit=8,
        S_trainable=True,
        mode="phase",
        v_max=10.8,
        v_pi=4.36,
        photodetect=True,
        device=device).to(device)
    model.set_gamma_noise(1e-3)
    x = torch.randn(4, 1, 28, 28, dtype=torch.float, device=device)
    y = model(x)
    print(y)
