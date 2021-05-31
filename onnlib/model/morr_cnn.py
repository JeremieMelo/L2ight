import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from collections import OrderedDict

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .layers import *
from pyutils import *
sys.path.pop(0)


class MORR_CLASS_SeparableCNN(nn.Module):
    '''MORR CNN for classification (MORR-ONN). MORR crossbar-based separable convolution [SqueezeLight, DATE'21] [O2NN, DATE'21]
    '''
    def __init__(self,
        img_height,
        img_width,
        in_channel,
        n_class,
        kernel_list=[16],
        kernel_size_list=[3],
        block_list=[4],
        stride_list=[1],
        padding_list=[1],
        pool_out_size=5,
        hidden_list=[32],
        in_bit=32,
        w_bit=32,
        mode="usv",
        v_max=10.8,
        v_pi=4.36,
        act_thres=6,
        photodetect=True,
        bias=False,
        ### mrr parameter
        mrr_a=0.8578,
        mrr_r=0.8985,
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
    ):
        super().__init__()
        self.img_height = img_height
        self.img_width = img_width
        self.in_channel = in_channel
        self.n_class = n_class
        self.kernel_list = kernel_list
        self.kernel_size_list = kernel_size_list
        self.block_list = block_list
        self.hidden_list = hidden_list
        self.stride_list = stride_list
        self.padding_list = padding_list
        self.pool_out_size=pool_out_size
        self.in_bit = in_bit
        self.w_bit = w_bit

        self.mode = mode
        self.v_max = v_max
        self.v_pi = v_pi
        self.act_thres = act_thres

        self.photodetect = photodetect
        self.bias = bias

        self.mrr_a = mrr_a
        self.mrr_r = mrr_r
        self.wg_gap = wg_gap
        self.n_eff_0 = n_eff
        self.n_g = n_g
        self.lambda_0 = lambda_0
        self.delta_lamdba = delta_lambda
        self.max_col = max_col
        self.max_row = max_row

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
            conv = MORRSeparableConv2d(
                in_channel,
                out_channel,
                self.kernel_size_list[idx],
                stride=self.stride_list[idx],
                padding=self.padding_list[idx],
                mini_block=1,
                bias=self.bias,
                mode="weight",
                v_max=self.v_max,
                v_pi=self.v_pi,
                in_bit=self.in_bit,
                w_bit=self.w_bit,
                photodetect=self.photodetect,
                mrr_a=self.mrr_a,
                mrr_r=self.mrr_r,
                ### waveguide parameters
                wg_gap=self.wg_gap, ## waveguide length (um) between crossings
                n_eff=self.n_eff, ## effective index
                n_g=self.n_g, ## group index
                lambda_0=self.lambda_0, ## center wavelength (nm)
                delta_lambda=self.delta_lamdba, ## delta wavelength (nm)
                ### crossbar parameters
                max_col=self.max_col, ## maximum number of columns
                max_row=self.max_row, ## maximum number of rows
                device=self.device
            )

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
            fc = nn.Linear(in_channel, out_channel, bias=False)

            # bn = nn.BatchNorm1d(out_channel)
            activation = nn.ReLU()

            self.fc_layers[layer_name] = fc
            # self.bn_layers[layer_name] = bn
            self.acts[layer_name] = activation
            super().__setattr__(layer_name, fc)
            # super().__setattr__(bn_name, bn)
            super().__setattr__(act_name, activation)

        layer_name = "fc"+str(len(self.hidden_list)+1)
        fc = nn.Linear(
            self.hidden_list[-1] if len(self.hidden_list) > 0 else feature_size,
            self.n_class,
            bias=self.bias
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


class MORR_CLASS_CNN(nn.Module):
    '''MORR CNN for classification (MORR-ONN). MORR array-based convolution with learnable nonlienarity [SqueezeLight, DATE'21] [ON-GOING, TCAD'21]
    '''
    def __init__(self,
        img_height,
        img_width,
        in_channel,
        n_class,
        kernel_list=[16],
        kernel_size_list=[3],
        block_list=[4],
        stride_list=[1],
        padding_list=[1],
        pool_out_size=5,
        hidden_list=[32],
        in_bit=32,
        w_bit=32,
        mode="weight",
        v_max=10.8,
        v_pi=4.36,
        act_thres=6,
        photodetect=True,
        bias=False,
        ### morr configuartion
        MORRConfig=MORRConfig_20um_MQ,
        trainable_morr_bias=False,
        trainable_morr_scale=False,
        device=torch.device("cuda")
    ):
        super().__init__()
        self.img_height = img_height
        self.img_width = img_width
        self.in_channel = in_channel
        self.n_class = n_class
        self.kernel_list = kernel_list
        self.kernel_size_list = kernel_size_list
        self.block_list = block_list
        self.hidden_list = hidden_list
        self.stride_list = stride_list
        self.padding_list = padding_list
        self.pool_out_size=pool_out_size
        self.in_bit = in_bit
        self.w_bit = w_bit

        self.mode = mode
        self.v_max = v_max
        self.v_pi = v_pi
        self.act_thres = act_thres

        self.photodetect = photodetect
        self.bias = bias

        self.MORRConfig = MORRConfig
        self.trainable_morr_bias = trainable_morr_bias
        self.trainable_morr_scale = trainable_morr_scale

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
            conv = AllPassMORRCirculantConv2d(
                in_channel,
                out_channel,
                self.kernel_size_list[idx],
                stride=self.stride_list[idx],
                padding=self.padding_list[idx],
                miniblock=self.block_list[idx],
                bias=self.bias,
                mode="weight",
                v_max=self.v_max,
                v_pi=self.v_pi,
                in_bit=self.in_bit,
                w_bit=self.w_bit,
                MORRConfig=self.MORRConfig,
                trainable_morr_scale=self.trainable_morr_scale,
                trainable_morr_bias=self.trainable_morr_bias,
                device=self.device
            )

            bn = nn.BatchNorm2d(out_channel)

            activation = nn.Hardtanh(-1, 1)

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
            fc = AllPassMORRCirculantLinear(
                in_channel,
                out_channel,
                miniblock=self.block_list[len(self.conv_layers)+idx],
                bias=False,
                mode="weight",
                v_max=self.v_max,
                v_pi=self.v_pi,
                in_bit=self.in_bit,
                w_bit=self.w_bit,
                MORRConfig=self.MORRConfig,
                trainable_morr_scale=self.trainable_morr_scale,
                trainable_morr_bias=self.trainable_morr_bias,
                device=self.device
            )

            # bn = nn.BatchNorm1d(out_channel)
            activation = nn.Hardtanh(-1, 1)

            self.fc_layers[layer_name] = fc
            # self.bn_layers[layer_name] = bn
            self.acts[layer_name] = activation
            super().__setattr__(layer_name, fc)
            # super().__setattr__(bn_name, bn)
            super().__setattr__(act_name, activation)

        layer_name = "fc"+str(len(self.hidden_list)+1)

        fc = AllPassMORRCirculantLinear(
                self.hidden_list[-1] if len(self.hidden_list) > 0 else feature_size,
                self.n_class,
                miniblock=self.block_list[-1],
                bias=False,
                mode="weight",
                v_max=self.v_max,
                v_pi=self.v_pi,
                in_bit=self.in_bit,
                w_bit=self.w_bit,
                MORRConfig=self.MORRConfig,
                trainable_morr_scale=self.trainable_morr_scale,
                trainable_morr_bias=self.trainable_morr_bias,
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


    def inject_phase_shifter_noise(self, noise_std=0.002, v_pi=4.36):
        gamma = np.pi / (v_pi**2)
        ### em stage phase shifter
        for layer in self.layers:
            phases = self.layers[layer].eigens

            new_phases = phases.data % (2*np.pi)
            # mask = new_phases.data.abs() > 1e-4 # 0 phases will elimiate the phase shifter, thus no noise
            mask = self.drop_masks[layer]
            # print(mask.sum()/mask.numel(), "phases have been injected noise")
            voltages = phase_to_voltage_cpu(new_phases.data.cpu().numpy(), gamma)
            gamma_with_noise = add_gaussian_noise_cpu(np.zeros_like(voltages, dtype=np.float32), noise_mean=gamma, noise_std=noise_std, trunc_range=())
            new_phases_n = voltage_to_phase_cpu(voltages, gamma_with_noise)
            # print("before:", self.layers[layer].eigens[..., 1].data.mean().item())
            phases.data[mask] = torch.from_numpy(new_phases_n).float().to(self.device)[mask]
            # print("after:", self.layers[layer].eigens[..., 1].data.mean().item())

        ### offt
        for layer in self.layers:
            T = self.layers[layer].T
            for i in range(len(T)):
                T[i].phases.require_grads = False
                phases = T[i].phases
                new_phases = phases.data % (2*np.pi)
                # mask = new_phases.data.abs() > 1e-4 # 0 phases will elimiate the phase shifter, thus no noise

                print(mask.sum()/mask.numel(), "phases have been injected noise")
                voltages = phase_to_voltage_cpu(new_phases.data.cpu().numpy(), gamma)
                gamma_with_noise = add_gaussian_noise_cpu(np.zeros_like(voltages, dtype=np.float32), noise_mean=gamma, noise_std=noise_std, trunc_range=())
                new_phases_n = voltage_to_phase_cpu(voltages, gamma_with_noise)
                print("before:", T[i].phases.data.mean().item())
                phases.data[mask] = torch.from_numpy(new_phases_n).float().to(self.device)[mask]
                print("after:", T[i].phases.data.mean().item())
                # phases[mask] = 0

    def restore_phases(self):
        for layer in self.layers:
            T = self.layers[layer].T
            for i in range(len(T)):
                T[i].phases.data.copy_(self.phase_backup[layer]["offt"][i].data)
            self.layers[layer].weight.data.copy_(self.phase_backup[layer]["weight"].data)

    def enable_morr_phase_loss(self):
        self.morr_phase_loss_flag = True

    def disable_morr_phase_loss(self):
        self.morr_phase_loss_flag = False

    def calc_morr_phase_loss(self, phase, threshold=1):
        return torch.relu(phase - threshold).mean()

    def register_morr_phase_loss(self, loss):
        self.morr_phase_loss = loss

    def get_morr_phase_loss(self):
        return self.morr_phase_loss

    def enable_morr_gradient_loss(self):
        self.morr_gradient_loss_flag = True

    def disable_morr_gradient_loss(self):
        self.morr_gradient_loss_flag = False

    def calc_morr_gradient_loss(self, layer, phase):
        # return polynomial(phase, layer.morr_lambda_to_mag_curve_coeff_half_grad).abs().mean()
        return mrr_roundtrip_phase_to_tr_grad_fused(phase, layer.MORRConfig.attenuation_factor, layer.MORRConfig.coupling_factor, intensity=True)

    def register_morr_gradient_loss(self, loss):
        self.morr_gradient_loss = loss

    def get_morr_gradient_loss(self):
        return self.morr_gradient_loss

    def get_finegrain_drop_mask(self, drop_perc=0):
        self.finegrain_drop_mask = {}
        for layer_name, layer in self.conv_layers.items():
            weight = layer.weight_quantizer(layer.weight.data) # [p, q, k]
            indices = weight.argsort(dim=-1)
            mask = torch.ones_like(weight, dtype=torch.bool, device=weight.device)
            drop_idx = int(drop_perc*weight.size(2))
            drop_idx = weight.size(2) - max(4, weight.size(2) - drop_idx)
            drop_indices = indices[:,:,0:drop_idx]
            mask.scatter_(2, drop_indices, 0)
            self.finegrain_drop_mask[layer_name] = mask
        for layer_name, layer in self.fc_layers.items():
            weight = layer.weight_quantizer(layer.weight.data) # [p, q, k]
            indices = weight.argsort(dim=-1)
            mask = torch.ones_like(weight, dtype=torch.bool, device=weight.device)
            drop_idx = int(drop_perc*weight.size(2))
            drop_idx = weight.size(2) - max(4, weight.size(2) - drop_idx)
            drop_indices = indices[:,:,0:drop_idx]
            mask.scatter_(2, drop_indices, 0)
            self.finegrain_drop_mask[layer_name] = mask
        return self.finegrain_drop_mask

    def apply_finegrain_drop_mask(self):
        if(self.finegrain_drop_mask is None):
            print("[W] No finegrained drop mask is available.")
            return
        for layer_name, layer in self.conv_layers.items():
            mask = self.finegrain_drop_mask[layer_name]
            # layer.weight.data.mul_(mask)
            layer.weight.data.masked_fill_(~mask, -1000)
        for layer_name, layer in self.fc_layers.items():
            mask = self.finegrain_drop_mask[layer_name]
            # layer.weight.data.mul_(mask)
            layer.weight.data.masked_fill_(~mask, -1000)

    def enable_crosstalk(self):
        for layer in self.conv_layers.values():
            layer.enable_crosstalk()
        for layer in self.fc_layers.values():
            layer.enable_crosstalk()

    def disable_crosstalk(self):
        for layer in self.conv_layers.values():
            layer.disable_crosstalk()
        for layer in self.fc_layers.values():
            layer.disable_crosstalk()

    def set_crosstalk_coupling_matrix(self, coupling_factor, drop_perc=0):
        for layer in self.conv_layers.values():
            layer.set_crosstalk_coupling_matrix(coupling_factor, drop_perc)
        for layer in self.fc_layers.values():
            layer.set_crosstalk_coupling_matrix(coupling_factor, drop_perc)

    def enable_phase_variation(self):
        for layer in self.conv_layers.values():
            layer.enable_phase_variation()
        for layer in self.fc_layers.values():
            layer.enable_phase_variation()

    def disable_phase_variation(self):
        for layer in self.conv_layers.values():
            layer.disable_phase_variation()
        for layer in self.fc_layers.values():
            layer.disable_phase_variation()

    def set_phase_variation(self, phase_noise_std=0):
        for layer in self.conv_layers.values():
            layer.set_phase_variation(phase_noise_std)
        for layer in self.fc_layers.values():
            layer.set_phase_variation(phase_noise_std)

    def get_num_MORR(self):
        n_morr = {}
        for layer in self.conv_layers.values():
            k = layer.miniblock
            n_morr[k] = n_morr.get(k, 0) + layer.grid_dim_x * layer.grid_dim_y
            n_morr[1] = n_morr.get(1, 0) + layer.grid_dim_x
        for layer in self.fc_layers.values():
            k = layer.miniblock
            n_morr[k] = n_morr.get(k, 0) + layer.grid_dim_x * layer.grid_dim_y
            n_morr[1] = n_morr.get(1, 0) + layer.grid_dim_x
        return n_morr, sum(i for i in n_morr.values())

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
