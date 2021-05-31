

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


class MZILinear(nn.Module):
    '''
    description: SVD-based Linear layer.
    '''
    def __init__(self,
        in_channel,
        out_channel,
        bias=False,
        S_trainable=True,
        S_scale=3,
        mode="weight",
        v_max=10.8,
        v_pi=4.36,
        w_bit=16,
        in_bit=16,
        photodetect=True,
        device=torch.device("cuda")
    ):
        super(MZILinear, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.S_trainable = S_trainable
        self.mode = mode
        assert mode in {"weight", "usv", "phase", "voltage"}, logging.error(f"Mode not supported. Expected one from (weight, usv, phase, voltage) but got {mode}.")
        self.v_max = v_max
        self.v_pi = v_pi
        self.gamma = np.pi / self.v_pi**2
        self.w_bit = w_bit
        self.in_bit = in_bit
        self.photodetect = photodetect
        self.device = device

        ### allocate parameters
        self.weight = None
        self.U = None
        self.S = None
        self.V = None
        self.delta_list_U = None
        self.delta_list_V = None
        self.phase_U = None
        self.phase_S = None
        self.S_scale = S_scale
        self.phase_V = None
        self.voltage_U = None
        self.voltage_S = None
        self.voltage_V = None
        ### build trainable parameters
        self.build_parameters(mode)
        ### unitary parametrization tool
        self.decomposer = RealUnitaryDecomposerBatch()
        ### quantization tool
        self.input_quantizer = PACT_Act(self.in_bit, device=self.device)
        self.unitary_quantizer = voltage_quantize_prune_with_gamma_noise_of_unitary_fn(self.w_bit, self.v_pi, self.v_max, device=self.device)
        self.weight_quantizer = voltage_quantize_prune_with_gamma_noise_of_matrix_fn(self.w_bit, self.v_pi, self.v_max, device=self.device)
        self.diag_quantizer = voltage_quantize_prune_with_gamma_noise_of_diag_fn(self.w_bit, self.v_pi, self.v_max, S_scale=self.S_scale, gamma_noise_std=0, device=self.device)
        self.voltage_quantizer = voltage_quantize_fn(self.w_bit, self.v_pi ,self.v_max)
        self.phase_quantizer = phase_quantize_fn(self.w_bit, self.v_pi ,self.v_max, gamma_noise_std=0)

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
        self.weight = torch.Tensor(self.out_channel, self.in_channel).to(self.device).float()
        ## usv mode
        self.U = torch.Tensor(self.out_channel, self.out_channel).to(self.device).float()
        self.S = torch.Tensor(min(self.out_channel, self.in_channel)).to(self.device).float()
        self.V = torch.Tensor(self.in_channel, self.in_channel).to(self.device).float()
        ## phase mode
        self.delta_list_U = torch.Tensor(self.out_channel).to(self.device).float()
        self.phase_U = torch.Tensor(self.out_channel*(self.out_channel-1)//2).to(self.device).float()
        self.phase_S = torch.Tensor(min(self.out_channel, self.in_channel)).to(self.device).float()
        self.delta_list_V = torch.Tensor(self.in_channel).to(self.device).float()
        self.phase_V = torch.Tensor(self.in_channel*(self.in_channel-1)//2).to(self.device).float()
        ## voltage mode
        self.voltage_U = torch.Tensor(self.out_channel*(self.out_channel-1)//2).to(self.device).float()
        self.voltage_S = torch.Tensor(min(self.out_channel, self.in_channel)).to(self.device).float()
        self.voltage_V = torch.Tensor(self.in_channel*(self.in_channel-1)//2).to(self.device).float()

        if(mode == 'weight'):
            self.weight = Parameter(self.weight)
        elif(mode == "usv"):
            self.U = Parameter(self.U)
            self.S = Parameter(self.S, requires_grad=self.S_trainable)
            self.V = Parameter(self.V)
        elif(mode == "phase"):
            self.phase_U = Parameter(self.phase_U)
            self.phase_S = Parameter(self.phase_S, requires_grad=self.S_trainable)
            self.phase_V = Parameter(self.phase_V)
        elif(mode == "voltage"):
            self.voltage_U = Parameter(self.voltage_U)
            self.voltage_S = Parameter(self.voltage_S, requires_grad=self.S_trainable)
            self.voltage_V = Parameter(self.voltage_V)
        else:
            raise NotImplementedError

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
        self.unitary_quantizer.set_gamma_noise(noise_std, random_state)
        self.diag_quantizer.set_gamma_noise(noise_std, random_state)
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

    def forward(self, x):
        if(self.in_bit < 16):
            x = self.input_quantizer(x)
        if(not self.fast_forward_flag or self.weight is None):
            weight = self.build_weight()
        else:
            weight = self.weight
        out = F.linear(x, weight, bias=None)
        if(self.photodetect):
            out = out.square()

        if(self.bias is not None):
            out = out + self.bias.unsqueeze(0)

        return out


class MZIBlockLinear(nn.Module):
    '''
    description: SVD-based Linear layer using blocking matrix multiplication.
    '''
    def __init__(self,
        in_channel,
        out_channel,
        miniblock=16, ### 16x16 MZI array
        bias=False,
        S_trainable=True,
        S_scale=3,
        mode="weight",
        v_max=10.8,
        v_pi=4.36,
        w_bit=16,
        in_bit=16,
        photodetect=True,
        device=torch.device("cuda")
    ):
        super(MZILinear, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.miniblock = miniblock
        self.S_trainable = S_trainable
        self.mode = mode
        assert mode in {"weight", "usv", "phase", "voltage"}, logging.error(f"Mode not supported. Expected one from (weight, usv, phase, voltage) but got {mode}.")
        self.v_max = v_max
        self.v_pi = v_pi
        self.gamma = np.pi / self.v_pi**2
        self.w_bit = w_bit
        self.in_bit = in_bit
        self.photodetect = photodetect
        self.device = device

        ### allocate parameters
        self.weight = None
        self.U = None
        self.S = None
        self.V = None
        self.delta_list_U = None
        self.delta_list_V = None
        self.phase_U = None
        self.phase_S = None
        self.S_scale = S_scale
        self.phase_V = None
        self.voltage_U = None
        self.voltage_S = None
        self.voltage_V = None
        ### build trainable parameters
        self.build_parameters(mode)
        ### unitary parametrization tool
        self.decomposer = RealUnitaryDecomposerBatch(alg="francis")
        ### quantization tool
        self.input_quantizer = PACT_Act(self.in_bit, device=self.device)
        self.unitary_quantizer = voltage_quantize_prune_with_gamma_noise_of_unitary_fn(self.w_bit, self.v_pi, self.v_max, device=self.device)
        self.weight_quantizer = voltage_quantize_prune_with_gamma_noise_of_matrix_fn(self.w_bit, self.v_pi, self.v_max, device=self.device)
        self.diag_quantizer = voltage_quantize_prune_with_gamma_noise_of_diag_fn(self.w_bit, self.v_pi, self.v_max, S_scale=self.S_scale, gamma_noise_std=0, device=self.device)
        self.voltage_quantizer = voltage_quantize_fn(self.w_bit, self.v_pi ,self.v_max)
        self.phase_quantizer = phase_quantize_fn(self.w_bit, self.v_pi ,self.v_max, gamma_noise_std=0)

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
        self.weight = torch.Tensor(self.out_channel, self.in_channel).to(self.device).float()
        ## usv mode
        self.U = torch.Tensor(self.out_channel, self.out_channel).to(self.device).float()
        self.S = torch.Tensor(min(self.out_channel, self.in_channel)).to(self.device).float()
        self.V = torch.Tensor(self.in_channel, self.in_channel).to(self.device).float()
        ## phase mode
        self.delta_list_U = torch.Tensor(self.out_channel).to(self.device).float()
        self.phase_U = torch.Tensor(self.out_channel*(self.out_channel-1)//2).to(self.device).float()
        self.phase_S = torch.Tensor(min(self.out_channel, self.in_channel)).to(self.device).float()
        self.delta_list_V = torch.Tensor(self.in_channel).to(self.device).float()
        self.phase_V = torch.Tensor(self.in_channel*(self.in_channel-1)//2).to(self.device).float()
        ## voltage mode
        self.voltage_U = torch.Tensor(self.out_channel*(self.out_channel-1)//2).to(self.device).float()
        self.voltage_S = torch.Tensor(min(self.out_channel, self.in_channel)).to(self.device).float()
        self.voltage_V = torch.Tensor(self.in_channel*(self.in_channel-1)//2).to(self.device).float()

        if(mode == 'weight'):
            self.weight = Parameter(self.weight)
        elif(mode == "usv"):
            self.U = Parameter(self.U)
            self.S = Parameter(self.S, requires_grad=self.S_trainable)
            self.V = Parameter(self.V)
        elif(mode == "phase"):
            self.phase_U = Parameter(self.phase_U)
            self.phase_S = Parameter(self.phase_S, requires_grad=self.S_trainable)
            self.phase_V = Parameter(self.phase_V)
        elif(mode == "voltage"):
            self.voltage_U = Parameter(self.voltage_U)
            self.voltage_S = Parameter(self.voltage_S, requires_grad=self.S_trainable)
            self.voltage_V = Parameter(self.voltage_V)
        else:
            raise NotImplementedError

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
        self.unitary_quantizer.set_gamma_noise(noise_std, random_state)
        self.diag_quantizer.set_gamma_noise(noise_std, random_state)
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

    def forward(self, x):
        if(self.in_bit < 16):
            x = self.input_quantizer(x)
        if(not self.fast_forward_flag or self.weight is None):
            weight = self.build_weight()
        else:
            weight = self.weight
        out = F.linear(x, weight, bias=None)
        if(self.photodetect):
            out = out.square()

        if(self.bias is not None):
            out = out + self.bias.unsqueeze(0)

        return out

if __name__ == "__main__":
    device = 'cuda'
    layer = MZILinear(4,4, mode="usv", device=device)
    x = torch.randn(1, 4, device=device)
    y1 = layer(x)
    layer.sync_parameters(src="usv")
    layer.switch_mode_to("phase")
    layer.set_gamma_noise(1.1e-5)
    y2 = layer(x)
    print(y1)
    print(y2)

