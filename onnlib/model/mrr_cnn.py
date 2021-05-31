from builtins import NotImplementedError
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


class AddDropMRR_CLASS_CNN_WG(nn.Module):
    '''Add drop MRR CNN for classification with dynamic in situ weight generation
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
        bias=False,
        ### mrr parameter
        mrr_a=MORRConfig_10um_MQ.attenuation_factor,
        mrr_r=MORRConfig_10um_MQ.coupling_factor,
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

        self.bias = bias

        self.mrr_a = mrr_a
        self.mrr_r = mrr_r

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
            conv = AddDropMRRConv2d(
                in_channel,
                out_channel,
                self.kernel_size_list[idx],
                stride=self.stride_list[idx],
                padding=self.padding_list[idx],
                bias=self.bias,
                miniblock=self.block_list[idx],
                mode=self.mode,
                v_max=self.v_max,
                v_pi=self.v_pi,
                in_bit=self.in_bit,
                w_bit=self.w_bit,
                mrr_a=self.mrr_a,
                mrr_r=self.mrr_r,
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
            miniblock = self.block_list[len(self.conv_layers) + idx]
            out_channel = hidden_dim
            # fc = nn.Linear(in_channel, out_channel, bias=False)
            fc = AddDropMRRLinear(
                in_channel,
                out_channel,
                bias=self.bias,
                miniblock=miniblock,
                mode=self.mode,
                v_max=self.v_max,
                v_pi=self.v_pi,
                in_bit=self.in_bit,
                w_bit=self.w_bit,
                mrr_a=self.mrr_a,
                mrr_r=self.mrr_r,
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
        # fc = nn.Linear(
        #     self.hidden_list[-1] if len(self.hidden_list) > 0 else feature_size,
        #     self.n_class,
        #     bias=self.bias
        #     )
        fc = AddDropMRRLinear(
                self.hidden_list[-1] if len(self.hidden_list) > 0 else feature_size,
                self.n_class,
                bias=self.bias,
                miniblock=self.block_list[-1],
                mode=self.mode,
                v_max=self.v_max,
                v_pi=self.v_pi,
                in_bit=self.in_bit,
                w_bit=self.w_bit,
                mrr_a=self.mrr_a,
                mrr_r=self.mrr_r,
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

    def get_num_params(self, fullrank=False):
        params = {}
        for layer_name, layer in self.conv_layers.items():
            params[layer_name] = layer.get_num_params(fullrank=fullrank)
        for layer_name, layer in self.bn_layers.items():
            params[layer_name+"_bn"] = layer.weight.numel() + layer.bias.numel()
        for layer_name, layer in self.fc_layers.items():
            params[layer_name] = layer.get_num_params(fullrank=fullrank)

        return params

    def get_total_num_params(self, fullrank=False):
        params = self.get_num_params(fullrank=fullrank)
        return sum(i for i in params.values())

    def get_param_size(self, fullrank=False, fullprec=False):
        params = {}

        for layer_name, layer in self.conv_layers.items():
            # w_bit = 32 if fullrank else layer.w_bit
            # params[layer_name] = layer.get_num_params(fullrank=fullrank) * w_bit / 8 ## Byte
            params[layer_name] = layer.get_param_size(fullrank)
        for layer_name, layer in self.bn_layers.items():
            params[layer_name+"_bn"] = layer.weight.numel() * 4 + layer.bias.numel() * 4 ## Byte
        for layer_name, layer in self.fc_layers.items():
            params[layer_name] = layer.get_param_size(fullrank, fullprec)
            # w_bit = 32 if fullrank else layer.w_bit
            # params[layer_name] = layer.get_num_params(fullrank=fullrank) * w_bit / 8 ## Byte
        return params

    def get_total_param_size(self, fullrank=False, fullprec=False):
        if(fullrank and fullprec):
            return self.get_total_num_params(fullrank=True) * 4 / 1024
        elif(fullrank and not fullprec):
            raise NotImplementedError
        elif(not fullrank and fullprec):
            return self.get_total_num_params(fullrank=False) * 4 / 1024
        else:
            params = self.get_param_size(fullrank=False, fullprec=False)
            return sum(i for i in params.values()) / 1024 ## KB

    def get_weight_compression_ratio(self):
        return self.get_total_num_params(fullrank=False) / self.get_total_num_params(fullrank=True)

    def get_memory_compression_ratio(self):
        return self.get_total_param_size(fullrank=False) / self.get_total_param_size(fullrank=True)

    def enable_dynamic_weight(self, base_in, base_out, relu=True, nonlinear=True, last_layer=False):
        for layer in self.conv_layers.values():
            layer.enable_dynamic_weight(base_in, base_out, relu=relu, nonlinear=nonlinear)
        for i, layer in enumerate(self.fc_layers.values()):
            if (i < len(self.fc_layers) - 1):
                layer.enable_dynamic_weight(int((layer.in_channel  + layer.miniblock - 1) / layer.miniblock), base_out, relu=relu, nonlinear=nonlinear)
            else:
                if(last_layer):
                    layer.enable_dynamic_weight(int((layer.in_channel  + layer.miniblock - 1) / layer.miniblock), base_out, relu=relu, nonlinear=nonlinear)

    def disable_dynamic_weight(self):
        for layer in self.conv_layers.values():
            layer.disable_dynamic_weight()
        for layer in self.fc_layers.values():
            layer.disable_dynamic_weight()

    def get_ortho_loss(self):
        loss = 0
        for layer in self.conv_layers.values():
            loss = loss + layer.get_ortho_loss()
        for layer in self.fc_layers.values():
            loss = loss + layer.get_ortho_loss()
        return loss

    def get_approximation_loss(self, model, cache=False):
        loss = 0
        for layer_name, layer in self.conv_layers.items():
            if(cache):
                w = layer.weight_2
            else:
                w = layer.build_weight()

            ref_layer = model.conv_layers[layer_name]
            w_ref =  ref_layer.build_weight().data
            loss = loss + F.mse_loss(w.view(-1), w_ref.view(-1))

        for layer_name, layer in self.fc_layers.items():
            if(cache):
                w = layer.weight_2
            else:
                w = layer.build_weight()

            ref_layer = model.fc_layers[layer_name]
            w_ref = ref_layer.build_weight().data
            loss = loss + F.mse_loss(w.view(-1), w_ref.view(-1))
        return loss

    def approximate_target_model(self, model, n_step=10000, alg="svd"):
        ### first copy bias
        for layer_name, layer in self.conv_layers.items():
            ref_layer = model.conv_layers[layer_name]
            if(layer.bias is not None and ref_layer.bias is not None):
                layer.bias.data.copy_(ref_layer.bias.data)
        for layer_name, layer in self.fc_layers.items():
            ref_layer = model.fc_layers[layer_name]
            if(layer.bias is not None and ref_layer.bias is not None):
                layer.bias.data.copy_(ref_layer.bias.data)
        ### then copy BN
        for layer_name, layer in self.bn_layers.items():
            ref_layer = model.bn_layers[layer_name]
            layer.bias.data.copy_(ref_layer.bias.data)
            layer.weight.data.copy_(ref_layer.weight.data)
        if(alg == "train"):
            from tqdm import tqdm
            optimizer = RAdam((p for p in self.parameters() if p.requires_grad), lr=1e-3)
            # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.985)
            for i in tqdm(range(n_step)):
                loss = self.get_approximation_loss(model, cache=False)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if(i % 1000 == 0):
                    print(f"[train] step {i}: loss: {loss.item():.4f}")
        elif(alg == "svd"):
            for layer_name, layer in self.conv_layers.items():
                ref_layer = model.conv_layers[layer_name]
                if(layer.coeff_out is not None):
                    u, v = lowrank_decompose(ref_layer.weight.data.view(ref_layer.out_channel, -1), r=layer.base_out, u_ortho=False)
                    layer.coeff_out.data.copy_(u)
                else:
                    v = ref_layer.weight.data
                if(layer.coeff_in is not None):
                    u, v = lowrank_decompose(v.view(layer.base_out, layer.in_channel, -1), r=layer.base_in, u_ortho=False)
                    layer.coeff_in.data.copy_(u)
                    layer.basis.data.copy_(v.view_as(layer.basis.data))
                else:
                    if(layer.basis is not None):
                        print(layer.basis.size(), v.size())
                        layer.basis.data.copy_(v.view_as(layer.basis.data))
                    else:
                        layer.weight.data.copy_(v)
            # for layer_name, layer in self.fc_layers.items():
            #     ref_layer = model.fc_layers[layer_name]
            #     u, v = lowrank_decompose(ref_layer.weight.data, r=layer.base_out, u_ortho=True)
            #     layer.coeff_out.data.copy_(u)
            #     layer.basis.data.copy_(v)
            with torch.no_grad():
                loss = self.get_approximation_loss(model, cache=False)
                print(f"[svd] step 0: loss: {loss.item():.4f}")

        else:
            raise NotImplementedError

    def assign_separate_weight_bit(self, qb=32, qu=32, qv=32, preserve_prec=True):
        for layer in self.conv_layers.values():
            layer.assign_separate_weight_bit(qb, qu, qv, preserve_prec=preserve_prec)

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

    model = AddDropMRR_CLASS_CNN_WG(
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
        mode="weight",
        v_max=10.8,
        v_pi=4.36,
        device=device).to(device)
    x = torch.randn(4, 4, 28, 28, dtype=torch.float, device=device)
    y = model(x)
    print(y)
