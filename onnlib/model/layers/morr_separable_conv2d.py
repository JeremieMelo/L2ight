from functools import lru_cache
import logging
import os
import sys

from torch import random
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', ".."))

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Parameter, init

from pyutils import *

sys.path.pop(0)

__all__ = ["MORRSeparableConv2d"]

class MORRSeparableConv2d(nn.Module):
    '''
    description: MORR-based separable conv2d [HPEngine, MORR-ONN]
    '''
    def __init__(self,
        in_channel,
        out_channel,
        kernel_size=3,
        stride=1,
        padding=1,
        mini_block=4,
        bias=False,
        mode="weight",
        v_max=10.8,
        v_pi=4.36,
        w_bit=16,
        in_bit=16,
        photodetect=True,
        mrr_a=0.8682,
        mrr_r=0.8602,
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
        super(MORRSeparableConv2d, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.mini_block = mini_block
        self.mode = mode
        assert mode in {"weight", "usv", "phase", "voltage"}, logging.error(f"Mode not supported. Expected one from (weight, usv, phase, voltage) but got {mode}.")
        self.v_max = v_max
        self.v_pi = v_pi
        self.gamma = np.pi / self.v_pi**2
        self.w_bit = w_bit
        self.in_bit = in_bit
        self.photodetect = photodetect
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


        self.x_zero_pad = None
        ### allocate parameters
        ### build trainable parameters
        self.build_parameters(mode)

        ### quantization tool
        self.input_quantizer = PACT_Act(self.in_bit, device=self.device)
        self.weight_quantizer = weight_quantize_fn(w_bit)
        # voltage_quantize_prune_with_gamma_noise_of_matrix_fn(self.w_bit, self.v_pi, self.v_max, device=self.device)
        self.voltage_quantizer = voltage_quantize_fn(self.w_bit, self.v_pi ,self.v_max)
        self.phase_quantizer = phase_quantize_fn(self.w_bit, self.v_pi ,self.v_max, gamma_noise_std=0)

        ### waveguide phase response
        self._lambda = torch.arange(in_channel, device=self.device, dtype=torch.float32).mul_(self.delta_lamdba).add_(self.lambda_0)
        self.n_eff = self.n_eff_0 - (self.n_g - self.n_eff_0) / self.lambda_0 * (self._lambda - self.lambda_0)
        ## [1, max_row, max_col] * [n_lambda, 1, 1] -> [n_almbda, max_row, max_col] -> [outc, inc, max_col]
        self.wg_phase = ((torch.arange(self.max_col, device=self.device, dtype=torch.float32).unsqueeze(0) - torch.arange(out_channel, device=self.device, dtype=torch.float32).unsqueeze(1)).unsqueeze(0) * (2*np.pi*self.wg_gap*self.n_eff/self._lambda).unsqueeze(-1).unsqueeze(-1)).permute(1,0,2).contiguous()


        ### default set to slow forward
        self.disable_fast_forward()
        ### default set no gamma noise
        self.set_gamma_noise(0)
        ### default set no crosstalk
        self.set_crosstalk_factor(0)
        ### default disable mixed training
        self.disable_mixedtraining()

        if(bias):
            self.bias = Parameter(torch.Tensor(out_channel).to(self.device))
        else:
            self.register_parameter('bias', None)

        # self.reset_parameters()

    def build_parameters(self, mode="weight"):
        ## weight mode
        ### depthwise weight, multipler is set to 1 by default, a kernel is mapped to one MORR
        in_channel = self.in_channel
        out_channel = self.out_channel
        kernel_size = self.kernel_size
        mini_block = self.mini_block
        self.d_weight = Parameter(torch.Tensor(in_channel, 1, kernel_size, kernel_size).to(self.device))
        self.p_weight = Parameter(torch.Tensor(out_channel, in_channel).to(self.device))

        self.in_channel_pad = int(np.ceil(self.in_channel / mini_block).item()) * mini_block
        self.out_channel_pad = int(np.ceil(self.out_channel / mini_block).item()) * mini_block
        self.grid_dim_y = self.out_channel_pad // mini_block
        self.grid_dim_x = self.in_channel_pad // mini_block

    def reset_parameters(self):
        if(self.mode == "weight"):
            init.kaiming_normal_(self.weight.data)
        elif(self.mode == "usv"):
            W = init.kaiming_normal_(torch.empty(self.out_channel, self.in_channel, dtype=self.U.dtype, device=self.device))
            U, S, V = torch.svd(W, some=False)
            V = V.t()
            self.U.data.copy_(U)
            self.V.data.copy_(V)
            # self.S.data.copy_(torch.from_numpy(S).float().to(self.device))
            self.S.data.copy_(torch.ones(S.shape[0], dtype=self.U.dtype, device=self.device))
        elif(self.mode == "phase"):
            # W = init.kaiming_normal_(torch.empty(self.out_channel, self.in_channel)).numpy().astype(np.float64)
            # U, S, V = np.linalg.svd(W, full_matrices=True)
            U = nn.init.orthogonal_(torch.zeros(self.out_channel, self.out_channel, dtype=self.phase_U.dtype, device=self.phase_U.device))
            V = nn.init.orthogonal_(torch.zeros(self.in_channel, self.in_channel, dtype=self.phase_U.dtype, device=self.phase_U.device))
            delta_list, phi_mat = self.decomposer.decompose(U)
            self.delta_list_U.data.copy_(delta_list)
            self.phase_U.data.copy_(upper_triangle_to_vector(phi_mat))
            delta_list, phi_mat = self.decomposer.decompose(V)
            self.delta_list_V = delta_list
            # print("phi_mat",phi_mat.size(), "V", V.size(), "phase_V", self.phase_V.size())
            self.phase_V.data.copy_(upper_triangle_to_vector(phi_mat))
            # self.phase_S.data.copy_(torch.from_numpy(S).float().to(self.device).clamp_(max=self.S_scale).mul_(1/self.S_scale).acos())

            self.phase_S.data.copy_(torch.ones_like(self.phase_S).mul_(1/self.S_scale).acos())

        elif(self.mode == "voltage"):
            U = nn.init.orthogonal_(torch.zeros(self.out_channel, self.out_channel, dtype=self.phase_U.dtype, device=self.phase_U.device))
            V = nn.init.orthogonal_(torch.zeros(self.in_channel, self.in_channel, dtype=self.phase_U.dtype, device=self.phase_U.device))
            delta_list, phi_mat = self.decomposer.decompose(U)
            self.delta_list_U.data.copy_(delta_list)
            self.voltage_U.data.copy_(phase_to_voltage(upper_triangle_to_vector(phi_mat), self.gamma))
            delta_list, phi_mat = self.decomposer.decompose(V)
            self.delta_list_V = delta_list
            self.voltage_V.data.copy_(phase_to_voltage(upper_triangle_to_vector(phi_mat), self.gamma))
            self.voltage_S.data.copy_(phase_to_voltage(torch.ones_like(self.voltage_S).mul_(1/self.S_scale).acos(), self.gamma))
        else:
            raise NotImplementedError

        if self.bias is not None:
            init.uniform_(self.bias, 0, 0)

    def build_weight_from_usv(self, U, S, V):
        ### differentiable feature is gauranteed
        S = S.clamp(min=-self.S_scale, max=self.S_scale)
        if(self.out_channel == self.in_channel):
            weight = torch.mm(U, S.unsqueeze(1)*V)
        elif(self.out_channel > self.in_channel):
            weight = torch.mm(U[:, :self.in_channel], S.unsqueeze(1)*V)
        else:
            weight = torch.mm(U*S.unsqueeze(0), V[:self.out_channel, :])
        self.weight.data.copy_(weight)
        return weight

    def build_weight_from_phase(self, delta_list_U, phase_U, delta_list_V, phase_V, phase_S, update_list={"phase_U", "phase_S", "phase_V"}):
        ### not differentiable
        ### reconstruct is time-consuming, a fast method is to only reconstruct based on updated phases
        if("phase_U" in update_list):
            self.U.data.copy_(self.decomposer.reconstruct(delta_list_U, vector_to_upper_triangle(phase_U)))
        if("phase_V" in update_list):
            self.V.data.copy_(self.decomposer.reconstruct(delta_list_V, vector_to_upper_triangle(phase_V)))
        if("phase_S" in update_list):
            self.S.data.copy_(phase_S.cos().mul_(self.S_scale))
        return self.build_weight_from_usv(self.U, self.S, self.V)

    def build_weight_from_voltage(self, delta_list_U, voltage_U, delta_list_V, voltage_V, voltage_S, gamma_U, gamma_V, gamma_S):
        self.phase_U = voltage_to_phase(voltage_U, gamma_U)
        self.phase_V = voltage_to_phase(voltage_V, gamma_V)
        self.phase_S = voltage_to_phase(voltage_S, gamma_S)
        return self.build_weight_from_phase(delta_list_U, self.phase_U, delta_list_V, self.phase_V, self.phase_S)

    def build_phase_from_usv(self, U, S, V):
        delta_list, phi_mat = self.decomposer.decompose(U.data.clone())
        self.delta_list_U.data.copy_(delta_list)
        self.phase_U.data.copy_(upper_triangle_to_vector(phi_mat))

        delta_list, phi_mat = self.decomposer.decompose(V.data.clone())
        self.delta_list_V.data.copy_(delta_list)
        self.phase_V.data.copy_(upper_triangle_to_vector(phi_mat))

        self.phase_S.data.copy_(S.clamp(min=-self.S_scale, max=self.S_scale).mul_(1/self.S_scale).acos())

        return self.delta_list_U, self.phase_U, self.delta_list_V, self.phase_V, self.phase_S

    def build_usv_from_weight(self, weight):
        ### differentiable feature is gauranteed
        U, S, V = weight.data.svd(some=False)
        V = V.t().contiguous()
        S = S.clamp(min=-self.S_scale, max=self.S_scale)
        self.U.data.copy_(U)
        self.S.data.copy_(S)
        self.V.data.copy_(V)
        return U, S, V

    def build_phase_from_weight(self, weight):
        return self.build_phase_from_usv(*self.build_usv_from_weight(weight))

    def build_voltage_from_phase(self, delta_list_U, phase_U, delta_list_V, phase_V, phase_S):
        self.delta_list_U = delta_list_U
        self.delta_list_V = delta_list_V
        self.voltage_U.data.copy_(phase_to_voltage(phase_U, self.gamma))
        self.voltage_S.data.copy_(phase_to_voltage(phase_S, self.gamma))
        self.voltage_V.data.copy_(phase_to_voltage(phase_V, self.gamma))

        return self.delta_list_U, self.voltage_U, self.delta_list_V, self.voltage_V, self.voltage_S

    def build_voltage_from_usv(self, U, S, V):
        return self.build_voltage_from_phase(*self.build_phase_from_usv(U, S, V))

    def build_voltage_from_weight(self, weight):
        return self.build_voltage_from_phase(*self.build_phase_from_usv(*self.build_usv_from_weight(weight)))

    def sync_parameters(self, src="weight"):
        '''
        description: synchronize all parameters from the source parameters
        '''
        if(src == "weight"):
            self.build_voltage_from_weight(self.weight)
        elif(src== "usv"):
            self.build_voltage_from_usv(self.U, self.S, self.V)
            self.build_weight_from_usv(self.U, self.S, self.V)
        elif(src == "phase"):
            self.build_weight_from_phase(self.delta_list_U, self.phase_U, self.delta_list_V, self.phase_V, self.phase_S)
            self.build_voltage_from_phase(self.delta_list_U, self.phase_U, self.delta_list_V, self.phase_V, self.phase_S)
        elif(src == "voltage"):
            self.build_weight_from_voltage(self.delta_list_U, self.voltage_U, self.delta_list_V, self.voltage_V, self.voltage_S)
        else:
            raise NotImplementedError

    def build_weight(self, update_list={"phase_U", "phase_S", 'phase_V'}):
        if(self.mode == "weight"):
            if(self.w_bit < 16):
                ### differentiable svd and quantizer based on LTE to enable QAT
                weight = self.weight_quantizer(self.weight)
            else:
                weight = self.weight
        elif(self.mode == "usv"):
            ### differentiable quantizer based on LTE to enable QAT
            if(self.w_bit < 16):
                _, U = self.unitary_quantizer(self.U)
                _, V = self.unitary_quantizer(self.U)
                S = self.diag_quantizer(self.S)
                # U = quantize_voltage_of_unitary_cpu(self.U, self.w_bit, self.v_max, output_device=self.device)
                # V = quantize_voltage_of_unitary_cpu(self.V, self.w_bit, self.v_max, output_device=self.device)
            else:
                U = self.U
                V = self.V
                S = self.S
            weight = self.build_weight_from_usv(U, S, V)
        elif(self.mode == "phase"):
            ### not differentiable
            if(self.w_bit < 16 or self.gamma_noise_std > 1e-5 or self.crosstalk_factor > 1e-5):
                phase_U = self.phase_quantizer(self.phase_U, self.mixedtraining_mask["phase_U"] if self.mixedtraining_mask is not None else None, mode="triangle")
                phase_S = self.phase_quantizer(self.phase_S, self.mixedtraining_mask["phase_S"] if self.mixedtraining_mask is not None else None, mode="diagonal")
                phase_V = self.phase_quantizer(self.phase_V, self.mixedtraining_mask["phase_V"] if self.mixedtraining_mask is not None else None, mode="triangle")
            else:
                phase_U = self.phase_U
                phase_S = self.phase_S
                phase_V = self.phase_V

            weight = self.build_weight_from_phase(self.delta_list_U, phase_U, self.delta_list_V, phase_V, phase_S, update_list=update_list)
        elif(self.mode == "voltage"):
            ### not differentiable
            if(self.gamma_noise_std > 1e-5):
                gamma_U = gen_gaussian_noise(self.voltage_U, noise_mean=self.gamma, noise_std=self.gamma_noise_std, trunc_range=())
                gamma_S = gen_gaussian_noise(self.voltage_S, noise_mean=self.gamma, noise_std=self.gamma_noise_std, trunc_range=())
                gamma_V = gen_gaussian_noise(self.voltage_V, noise_mean=self.gamma, noise_std=self.gamma_noise_std, trunc_range=())
            else:
                gamma_U = gamma_S = gamma_V = self.gamma
            if(self.w_bit < 16):
                voltage_U = clip_to_valid_quantized_voltage(self.voltage_quantizer(self.voltage_U),self.gamma, self.w_bit, self.v_max, wrap_around=True)
                voltage_S = clip_to_valid_quantized_voltage(self.voltage_quantizer(self.voltage_S),self.gamma, self.w_bit, self.v_max, wrap_around=True)
                voltage_V = clip_to_valid_quantized_voltage(self.voltage_quantizer(self.voltage_V),self.gamma, self.w_bit, self.v_max, wrap_around=True)
            else:
                voltage_U = self.voltage_U
                voltage_S = self.voltage_S
                voltage_V = self.voltage_V
            weight = self.build_weight_from_voltage(self.delta_list_U, voltage_U, self.delta_list_V, voltage_V, voltage_S, gamma_U, gamma_V, gamma_S)
        else:
            raise NotImplementedError
        return weight

    def enable_fast_forward(self):
        self.fast_forward_flag = True

    def disable_fast_forward(self):
        self.fast_forward_flag = False

    def set_gamma_noise(self, noise_std, random_state=None):
        self.gamma_noise_std = noise_std
        self.phase_quantizer.set_gamma_noise(noise_std, random_state)

    def set_crosstalk_factor(self, crosstalk_factor):
        self.crosstalk_factor = crosstalk_factor
        self.phase_quantizer.set_crosstalk_factor(crosstalk_factor)

    def load_parameters(self, param_dict):
        '''
        description: update parameters based on this parameter dictionary\\
        param param_dict {dict of dict} {layer_name: {param_name: param_tensor, ...}, ...}
        '''
        for name, param in param_dict.items():
            getattr(self, name).data.copy_(param)
        if(self.mode == "phase"):
            self.build_weight(update_list=param_dict)

    def gen_mixedtraining_mask(self, sparsity, prefer_small=False, random_state=None, enable=True):
        '''
        description: generate sparsity masks for mixed training \\
        param sparsity {float scalar} fixed parameter ratio, valid range: (0,1]
        prefer_small {bool scalar} True if select phases from small unitary first
        return mask {dict} a dict with all masks for trainable parameters in the current mode. 1/True is for trainable, 0/False is fixed.
        '''
        if(self.mode == "weight"):
            out = {"weight":self.weight.data > percentile(self.weight.data, 100*sparsity)}
        elif(self.mode == "usv"):
            ## S is forced with 0 sparsity
            out = {"U": self.U.data > percentile(self.U.data, sparsity*100), "S": torch.ones_like(self.S.data, dtype=torch.bool), "V": self.V.data > percentile(self.V.data, sparsity*100)}
        elif(self.mode == "phase"):
            ## phase_S is forced with 0 sparsity
            ## no theoretical guarantee of importance of phase. So random selection is used.
            ## for effciency, select phases from small unitary first. Another reason is that larger MZI array is less robust.
            if(prefer_small == False or self.phase_U.size(0) == self.phase_V.size(0)):
                if(random_state is not None):
                    set_torch_deterministic(random_state)
                mask_U = torch.zeros_like(self.phase_U.data).bernoulli_(p=1-sparsity)
                mask_S = torch.ones_like(self.S.data, dtype=torch.bool)
                if(random_state is not None):
                    set_torch_deterministic(random_state+1)
                mask_V = torch.zeros_like(self.phase_V.data).bernoulli_(p=1-sparsity)
            elif(self.phase_U.size(0) < self.phase_V.size(0)):
                total_nonzero = int((1-sparsity) * (self.phase_U.numel() + self.phase_V.numel()))
                if(total_nonzero <= self.phase_U.numel()):
                    indices = torch.from_numpy(np.random.choice(self.phase_U.numel(), size=[total_nonzero], replace=False)).to(self.phase_U.device).long()
                    mask_U = torch.zeros_like(self.phase_U.data, dtype=torch.bool)
                    mask_U.data[indices] = 1
                    mask_V = torch.zeros_like(self.phase_V, dtype=torch.bool)
                    mask_S = torch.ones_like(self.S.data, dtype=torch.bool)
                else:
                    indices = torch.from_numpy(np.random.choice(self.phase_V.numel(), size=[total_nonzero - self.phase_U.numel()], replace=False)).to(self.phase_V.device).long()
                    mask_V = torch.zeros_like(self.phase_V.data, dtype=torch.bool)
                    mask_V.data[indices] = 1
                    mask_U = torch.ones_like(self.phase_U, dtype=torch.bool)
                    mask_S = torch.ones_like(self.S.data, dtype=torch.bool)
            else:
                total_nonzero = int((1-sparsity) * (self.phase_U.numel() + self.phase_V.numel()))
                if(total_nonzero <= self.phase_V.numel()):
                    indices = torch.from_numpy(np.random.choice(self.phase_V.numel(), size=[total_nonzero], replace=False)).to(self.phase_V.device).long()
                    mask_V = torch.zeros_like(self.phase_V.data, dtype=torch.bool)
                    mask_V.data[indices] = 1
                    mask_U = torch.zeros_like(self.phase_U, dtype=torch.bool)
                    mask_S = torch.ones_like(self.S.data, dtype=torch.bool)
                else:
                    indices = torch.from_numpy(np.random.choice(self.phase_U.numel(), size=[total_nonzero - self.phase_V.numel()], replace=False)).to(self.phase_U.device).long()
                    mask_U = torch.zeros_like(self.phase_U.data, dtype=torch.bool)
                    mask_U.data[indices] = 1
                    mask_V = torch.ones_like(self.phase_V, dtype=torch.bool)
                    mask_S = torch.ones_like(self.S.data, dtype=torch.bool)


            out = {"phase_U": mask_U, "phase_S": mask_S, "phase_V": mask_V}
        elif(self.mode == "voltage"):
            ## voltage_S is forced with 0 sparsity
            ## no theoretical gaurantee of importance of phase. Given phase=gamma*v**2, we assume larger voltage is more important
            mask_U = self.voltage_U > percentile(self.voltage_U, 100*sparsity)
            mask_S = torch.ones_like(self.S.data, dtype=torch.bool)
            mask_V = self.voltage_V > percentile(self.voltage_V, 100*sparsity)
            out = {"voltage_U": mask_U, "voltage_S": mask_S, "voltage_V": mask_V}
        else:
            raise NotImplementedError
        if(enable):
            self.enable_mixedtraining(out)
        return out

    def enable_mixedtraining(self, masks):
        '''
        description: mixed training masks\\
        param masks {dict} {param_name: mask_tensor}
        return
        '''
        self.mixedtraining_mask = masks

    def disable_mixedtraining(self):
        self.mixedtraining_mask = None

    def switch_mode_to(self, mode):
        self.mode = mode

    def get_power(self, mixtraining_mask=None):
        masks = mixtraining_mask if mixtraining_mask is not None else (self.mixedtraining_mask if self.mixedtraining_mask is not None else None)
        if(masks is not None):
            power = ((self.phase_U.data * masks["phase_U"]) % (2 * np.pi)).sum()
            power += ((self.phase_S.data * masks["phase_S"]) % (2 * np.pi)).sum()
            power += ((self.phase_V.data * masks["phase_V"]) % (2 * np.pi)).sum()
        else:
            power = ((self.phase_U.data) % (2 * np.pi)).sum()
            power += ((self.phase_S.data) % (2 * np.pi)).sum()
            power += ((self.phase_V.data) % (2 * np.pi)).sum()
        return power.item()

    def depthwise_conv2d(self, x, weight):
        ### implemented by MORR, each MORR achieve one 2d kernel, might need some sparsity
        ### x: [bs, inc, h, w]
        ### weight: [inc, 1, ks, ks] \in [0, 1]
        ### return out: [bs, inc, h', w', 2]
        rt_phi = F.conv2d(x*x, weight, groups=self.in_channel, bias=None, stride=self.stride, padding=self.padding)
        x = mrr_roundtrip_phase_to_tr_phase(rt_phi, a=self.mrr_a, r=self.mrr_r)
        return x

    def build_complex_pointwise_weight(self, p_weight):
        ### crossbar waveguide has different phase response at different path for different wavelength: phi(i,j,lambda)
        ### relative phase can be mapped to p_weight instead of x, which is computationally cheaper
        ### p_weight: [outc, inc]
        ### phase:   [outc, inc]
        phase = mrr_tr_to_out_phase(p_weight, a=self.mrr_a, r=self.mrr_r, onesided=True) - np.pi/2
        ### add phase shift induced by waveguide
        ### wg_phase: [outc, inc, max_col]
        # print(phase.size(),phase.device, self.wg_phase.size(), self.wg_phase.device)
        phase = phase.unsqueeze(-1) + self.wg_phase # [outc, inc, max_col]
        p_weight = polar_to_complex(p_weight.unsqueeze(-1), phase) # [outc, inc, max_col, 2]
        return p_weight

    @lru_cache(maxsize=16)
    def get_zero_padding(self, x_size, padding, dim):
        size = list(x_size)
        size[dim] = padding
        zero_padding = torch.zeros(size, device=self.device, dtype=torch.float32)
        return zero_padding

    def pointwise_conv2d(self, x, weight):
        ### HPEngine: propagate through directional coupler
        ### x: [bs, inc, h', w', 2]
        ### weight: [outc, inc, max_col, 2]
        bs, inc, h_out, w_out, _ = x.size()
        x = x.permute(0,4,1,2,3).contiguous().view(bs*2, inc, h_out, w_out)
        _, x, _, _ = im2col_2d(W=None, X=x, stride=1, padding=0, w_size=(weight.size(0),weight.size(1),1,1)) # [inc, h'*w'*bs*2]
        x = x.view(inc, h_out*w_out*bs, 2) # [inc, h'*w'*bs, 2]
        max_col = weight.size(2)
        n_seg = int((x.size(1) + max_col - 1) / max_col)
        padding = n_seg * weight.size(2) - x.size(1)
        if(padding > 0):
            zero_padding = self.get_zero_padding(tuple(x.size()), padding, dim=1)
            x = torch.cat([x, zero_padding], dim=1)

        x = x.view(1, inc, n_seg, weight.size(2), 2) #[1, inc, n_seg, max_col, 2]
        weight = weight.unsqueeze(2) # [outc, inc, 1, max_col, 2]
        port1_real = x[..., 0] - weight[..., 1]
        port1_imag = x[..., 1] + weight[..., 0]
        port2_real = weight[..., 0] - x[..., 1]
        port2_imag = x[..., 0] + weight[..., 1]
        port1 = torch.stack([port1_real, port1_imag], dim=-1) # [outc, inc, n_seg, max_col, 2]
        port2 = torch.stack([port2_real, port2_imag], dim=-1) # [outc, inc, n_seg, max_col, 2]
        out = (get_complex_energy(port1).sum(dim=1) - get_complex_energy(port2).sum(dim=1)).view(weight.size(0), -1) # [outc, h'w'bs+padding]
        if(padding > 0):
            out = out[..., :-padding]
        # out [outc, h'w'bs] -> [bs, outc, h', w']
        out = out.view(weight.size(0), h_out, w_out, bs).permute(3,0,1,2).contiguous()
        return out

    def forward(self, x):
        if(self.in_bit < 16):
            x = self.input_quantizer(x)
        d_weight = self.weight_quantizer(self.d_weight)
        p_weight = self.weight_quantizer(self.p_weight)
        p_weight = self.build_complex_pointwise_weight(p_weight)
        x = self.depthwise_conv2d(x, d_weight) # [bs, inc, h, w] -> [bs, inc, h', w', 2]
        out = self.pointwise_conv2d(x, p_weight) #[bs, inc, h', w', 2] -> [bs, outc, h', w']

        if(self.bias is not None):
            out = out + self.bias.unsqueeze(0)

        return out

if __name__ == "__main__":
    device = 'cuda'
    layer = MORRSeparableConv2d(
        in_channel=4,
        out_channel=4,
        kernel_size=3,
        stride=1,
        padding=1,
        mini_block=4,
        bias=False,
        S_trainable=True,
        S_scale=3,
        mode="weight",
        v_max=10.8,
        v_pi=4.36,
        w_bit=16,
        in_bit=16,
        photodetect=True,
        mrr_a=0.8682,
        mrr_r=0.8602,
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
        )
    x = torch.randn(3, 4, 8, 8, device=device)
    y1 = layer(x)

    print(y1)


