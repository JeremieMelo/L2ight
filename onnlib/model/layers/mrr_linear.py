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
from .device import *

sys.path.pop(0)

__all__ = ["AddDropMRRLinear"]

class AddDropMRRLinear(nn.Module):
    '''
    description: Add-drop MRR Linear layer with weight generation
    '''
    def __init__(self,
        in_channel,
        out_channel,
        bias=False,
        miniblock=4,
        mode="weight",
        v_max=10.8,
        v_pi=4.36,
        w_bit=16,
        in_bit=16,
        ### mrr parameter
        mrr_a=MORRConfig_20um_MQ.attenuation_factor,
        mrr_r=MORRConfig_20um_MQ.coupling_factor,
        device=torch.device("cuda")
    ):
        super(AddDropMRRLinear, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.mode = mode
        self.miniblock = miniblock
        assert mode in {"weight", "weight_reduce", "phase", "voltage", "dsconv", "bsconv", "circconv", "penni", "dwconv"}, logging.error(f"Mode not supported. Expected one from (weight, weight_reduce, phase, voltage, dsconv, bsconv, circconv, penni, dwconv) but got {mode}.")
        if(mode in {"dsconv", "bsconv", "penni", "dwconv"}):
            self.mode = mode = "weight" ### does not support dsconv mode
        self.v_max = v_max
        self.v_pi = v_pi
        self.gamma = np.pi / self.v_pi**2
        self.w_bit = w_bit
        self.in_bit = in_bit
        self.mrr_a = mrr_a
        self.mrr_r = mrr_r
        self.device = device

        ### allocate parameters
        self.weight = None
        ### build trainable parameters
        self.build_parameters(mode)

        ### quantization tool
        # self.input_quantizer = PACT_Act(self.in_bit, device=self.device)
        self.input_quantizer = input_quantize_fn(self.in_bit, alg="normal", device=self.device)

        self.weight_quantizer = weight_quantize_fn(self.w_bit, alg="dorefa_sym")

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
        ### default disable dynamic weight generation
        self.disable_dynamic_weight()
        self.eye_b = None
        self.eye_v = None


        if(bias):
            self.bias = Parameter(torch.Tensor(out_channel).to(self.device))
        else:
            self.register_parameter('bias', None)

        # self.reset_parameters()

    def build_parameters(self, mode="weight"):
        ## weight mode
        ## phase mode for round-trip phase shift
        # self.phase = torch.Tensor(self.out_channel, int((self.in_channel + self.miniblock - 1)/self.miniblock), self.miniblock).to(self.device).float()
        # ## voltage mode
        # self.voltage = torch.Tensor(self.out_channel, int((self.in_channel + self.miniblock - 1)/self.miniblock), self.miniblock).to(self.device).float()

        if(mode in {'weight', "weight_reduce"}):
            self.weight = Parameter(torch.Tensor(self.out_channel, self.in_channel).to(self.device).float())
        elif(mode in "circconv"):
            self.weight = Parameter(torch.Tensor((self.out_channel + self.miniblock - 1)//self.miniblock, int((self.in_channel + self.miniblock - 1)/self.miniblock), self.miniblock).to(self.device).float())
        elif(mode == "phase"):
            raise NotImplementedError
            self.phase = Parameter(self.phase)
        elif(mode == "voltage"):
            raise NotImplementedError
            self.voltage = Parameter(self.voltage)
        else:
            raise NotImplementedError

    def reset_parameters(self):
        if(self.mode in {"weight", "weight_reduce", "circconv"}):
            init.kaiming_normal_(self.weight.data)
        elif(self.mode == "phase"):
           raise NotImplementedError
        elif(self.mode == "voltage"):
           raise NotImplementedError
        else:
            raise NotImplementedError

        if self.bias is not None:
            # init.uniform_(self.bias, 0, 0)
            fan_in = self.in_channel
            bound = 1 / np.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def sync_parameters(self, src="weight"):
        '''
        description: synchronize all parameters from the source parameters
        '''

        raise NotImplementedError

    def build_weight(self):
        if(self.mode in {"weight", "weight_reduce"}):
            if(self.w_bit < 16):
                ### differentiable quantizer based on STE to enable QAT (Dorefa-Net, arXiv 2016)
                weight = self.weight_quantizer(self.weight)
                if(self.dynamic_weight_flag):
                    self.basis = weight[:self.base_out, :self.base_in, ...] # [base_out, base_in, k]
                    if(self.coeff_in is not None):
                        coeff_in = self.weight_quantizer(self.coeff_in)
                    else:
                        coeff_in = None
                    if(self.coeff_out is not None):
                        coeff_out = self.weight_quantizer(self.coeff_out)
                    else:
                        coeff_out = None
                    weight = self.weight_generation(self.basis, coeff_in, coeff_out)
                else:
                    weight = weight.view(self.out_channel, -1)[:, :self.in_channel]
            else:
                weight = self.weight
                if(self.dynamic_weight_flag):
                    self.basis = weight[:self.base_out, :self.base_in, ...] # [base_out, base_in, k]
                    weight = self.weight_generation(self.basis, self.coeff_in, self.coeff_out)
                else:
                    weight = self.weight.view(self.out_channel, -1)[:, :self.in_channel]
        elif(self.mode in {"circconv"}):
            if(self.w_bit < 16):
                weight = self.weight_quantizer(self.weight)
            else:
                weight = self.weight
            p, q, k = weight.size()
            weight = circulant(weight) # [p, q, k] -> [p, q, k, k]
            weight = weight.permute([0, 2, 1, 3]).contiguous().view(p*k, q*k)[:self.out_channel, :self.in_channel]
        elif(self.mode == "phase"):
            ### not differentiable
            raise NotImplementedError
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
            raise NotImplementedError
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
        # self.phase_quantizer.set_gamma_noise(noise_std, random_state)

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
        # if(self.mode == "phase"):
        #     self.build_weight(update_list=param_dict)

    def gen_mixedtraining_mask(self, sparsity, prefer_small=False, random_state=None, enable=True):
        '''
        description: generate sparsity masks for mixed training \\
        param sparsity {float scalar} fixed parameter ratio, valid range: (0,1]
        prefer_small {bool scalar} True if select phases from small unitary first
        return mask {dict} a dict with all masks for trainable parameters in the current mode. 1/True is for trainable, 0/False is fixed.
        '''
        raise NotImplementedError
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
        raise NotImplementedError
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

    def enable_dynamic_weight(self, base_in, base_out, relu=True, nonlinear=True):
        ### multi-level weight generation
        self.base_in = base_in # input channel base
        self.base_out = base_out # output channel base
        if(base_out == 0): ## disable cross-kernel generation
            self.base_out = self.out_channel
        elif(min(self.out_channel, self.in_channel) > self.base_out > 0):
            ### enable generation
            self.base_out = base_out
        else:
            ### base_out is too large, cannot save param, then disable it
            self.base_out = self.out_channel

        ### TODO temporarily never turn on intra-kernel generation.
        if(int((self.in_channel + self.miniblock - 1)/self.miniblock) > self.base_in > 0):
            if(self.mode == "weight_reduce"):
                self.coeff_in = Parameter(torch.Tensor(self.base_out, int((self.in_channel + self.miniblock - 1)/self.miniblock) - self.base_in, self.base_in).to(self.device))
            elif(self.mode == "weight"):
                self.coeff_in = Parameter(torch.Tensor(self.base_out, int((self.in_channel + self.miniblock - 1)/self.miniblock), self.base_in).to(self.device))

            # init.xavier_normal_(self.coeff_in)
            init.kaiming_normal_(self.coeff_in, mode='fan_out', nonlinearity='relu')
        else:
            ### input channel is already very small
            self.coeff_in = None
        ### only when base_out < min(out_channel, in_channel), will cross_out_channel generation save #params.
        if(min(self.out_channel, self.in_channel) > self.base_out > 0):
            if(self.mode == "weight_reduce"):
                self.coeff_out = Parameter(torch.Tensor(self.out_channel - self.base_out, self.base_out).to(self.device))
            elif(self.mode == "weight"):
                self.coeff_out = Parameter(torch.Tensor(self.out_channel, self.base_out).to(self.device))
            # init.xavier_normal_(self.coeff_out)
            init.kaiming_normal_(self.coeff_out, mode='fan_out', nonlinearity='relu')
        else:
            self.coeff_out = None
        self.dynamic_weight_flag = True if self.coeff_in is not None or self.coeff_out is not None else False
        self.dynamic_weight_relu_flag = relu
        self.dynamic_weight_nonlinear_flag = nonlinear
        self.dynamic_preserve_prec = True
        if(self.dynamic_weight_flag):
            if(self.coeff_in is not None):
                self.basis = self.weight[:self.base_out, :self.base_in, ...]
            else:
                self.basis = self.weight[:self.base_out, ...]
        else:
            self.basis = None

    def disable_dynamic_weight(self):
        self.dynamic_weight_flag = False

    def weight_generation(self, basis, coeff_in, coeff_out):
        ### Level 1
        if(coeff_in is not None):
            # weight_1 [base_out, inc, blocksize]
            # coeff_in x basis = [bo, inc-bi, bi] x [bo, bi, blocksize]
            # basis = basis.view(basis.size(0), basis.size(1), -1)
            # print("basis", basis.requires_grad)
            # print_stat(basis)
            weight_1 = torch.matmul(coeff_in, basis)
            if(self.mode == "weight_reduce"):
                weight_1 = torch.cat([weight_1, basis], dim=1)
            # print("matmul 1", weight_1.requires_grad)
            # thresholding by relu
            # print_stat(weight_1)
            if(self.dynamic_weight_relu_flag):
                weight_1 = F.relu(weight_1)
            # modulate the input MRR with nonlinearity
            # first amplify and calculate the round-trip phase shift
            # print("relu", weight_1.requires_grad)
            # print_stat(weight_1)
            if(self.dynamic_weight_nonlinear_flag):
                weight_1 = (weight_1 * 1).square()
                # then change to through port intensity transmission
                weight_1 = mrr_roundtrip_phase_to_tr(rt_phi=weight_1, a=self.mrr_a, r=self.mrr_r, intensity=True) / 2
            # print_stat(weight_1)
        else:
            weight_1 = basis


        ### Level 2
        if(coeff_out is not None):
            ### TODO not supported yet
            # if(not self.dynamic_preserve_prec):
            #     weight_1 = self.coeff_out_quantizer(weight_1)
            # propogate through add-drop MRR weight bank
            # weight_2 [outc, inc]
            weight_1 = weight_1.view(weight_1.size(0),-1)[:, :self.in_channel]
            # print_stat(coeff_out)
            weight_2 = torch.matmul(coeff_out, weight_1)
            if(self.mode == "weight_reduce"):
                weight_2 = torch.cat([weight_2, weight_1], dim=0)
            # weight_2 = weight_2
            ### assume this weight can be achieved by weight bank with proper MRR conversion.
            # print_stat(weight_2)
        else:
            weight_2 = weight_1.view(weight_1.size(0),-1)[:, :self.in_channel]

        return weight_2

    def get_num_params(self, fullrank=False):
        if((self.dynamic_weight_flag==True) and (fullrank == False)):
            total = self.basis.numel()
            if(self.coeff_in is not None):
                total += self.coeff_in.numel()
            if(self.coeff_out is not None):
                total += self.coeff_out.numel()
        else:
            total = self.out_channel * self.in_channel
        if(self.bias is not None):
            total += self.bias.numel()

        return total

    def get_param_size(self, fullrank=False, fullprec=False):
        if((self.dynamic_weight_flag==True) and (fullrank == False)):
            total = self.basis.numel() * self.w_bit / 8
            if(self.coeff_in is not None):
                total += self.coeff_in.numel() * self.w_bit / 8
            if(self.coeff_out is not None):
                total += self.coeff_out.numel() * self.w_bit / 8
        else:
            if(fullprec):
                total = (self.out_channel * self.in_channel) * 4
            else:
                total = (self.out_channel * self.in_channel) * self.w_bit / 8
        if(self.bias is not None):
            total += self.bias.numel() * 4
        return total

    def get_ortho_loss(self):
        ### we want row vectors in the basis to be orthonormal
        if(self.dynamic_weight_flag):
            if(self.coeff_in is not None):
                basis = self.basis.view(self.basis.size(0), self.basis.size(1), -1) # [bo, bi, k^2]
                ## basis
                dot_b = torch.matmul(basis, basis.permute([0,2,1])) # [bo, bi, k^2] x [bo, k^2, bi] = [bo, bi, bi]

                ## U
                coeff_in = self.coeff_in / (self.coeff_in.data.norm(p=2, dim=1, keepdim=True) + 1e-8)# normalization
                dot_u = torch.matmul(coeff_in.permute(0,2,1), coeff_in) # [bo, bi, ci-bi] x [bo, ci-bi, bi] = [bo, bi, bi]
            else:
                pass

            if(self.coeff_out is not None):
                # V
                coeff_out = self.coeff_out / (self.coeff_out.data.norm(p=2, dim=0, keepdim=True) + 1e-8) # normalization
                dot_v = torch.matmul(coeff_out.t(), coeff_out) # [bo, co-bo] x [co-bo, bo] = [bo, bo]
            else:
                pass

            if(self.coeff_in is not None and self.eye_b is None):
                self.eye_b = torch.eye(dot_b.size(-1), dtype=dot_b.dtype, device=dot_b.device).unsqueeze(0).repeat(basis.size(0), 1, 1)
            if(self.coeff_out is not None and self.eye_v is None):
                self.eye_v = torch.eye(dot_v.size(-1), dtype=dot_v.dtype, device=dot_v.device)
            loss = 0
            if(self.coeff_in is not None):
                loss = loss + F.mse_loss(dot_b, self.eye_b) + F.mse_loss(dot_u, self.eye_b)
            if(self.coeff_out is not None):
                loss = loss + F.mse_loss(dot_v, self.eye_v)
        else:
            loss = 0

        return loss

    def forward(self, x):
        if(self.in_bit < 16):
            x = self.input_quantizer(x)
        if(not self.fast_forward_flag or self.weight is None):
            weight = self.build_weight()
        else:
            weight = self.weight #.view(self.out_channel, -1)[:, :self.in_channel]

        ### record weight_2
        self.weight_2 = weight
        out = F.linear(x, weight, bias=self.bias)

        # if(self.bias is not None):
        #     out = out + self.bias.unsqueeze(0)

        return out

    def extra_repr(self):
        s = ('{in_channel}, {out_channel}, wb={w_bit}, ib={in_bit}')
        if self.bias is None:
            s += ', bias=False'
        return s.format(**self.__dict__)

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

