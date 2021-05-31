
import logging
import os
import sys
import math

from torch import random
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', ".."))

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Parameter, init
from tqdm import tqdm

from pyutils import *
try:
    import universal_cuda
except:
    logging.warning(f"Import universal_cuda fail")
try:
    import hadamard_cuda
except:
    logging.warning(f"Import hadamard_cuda fail")
sys.path.pop(0)


class FOLinear(nn.Module):
    def __init__(self,
        in_channel,
        out_channel,
        mini_block=4,
        bias=False,
        S_scale=3,
        mode="phase",
        v_max=10.8,
        v_pi=4.36,
        w_bit=16,
        in_bit=16,
        transform="trainable",
        photodetect=False,
        device=torch.device("cuda")
    ):
        super(FOLinear, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.mini_block = mini_block

        self.grid_dim_y = out_channel // mini_block
        self.grid_dim_x = in_channel // mini_block
        self.grid_dim_y_pad = int(np.ceil(out_channel / mini_block))
        self.grid_dim_x_pad = int(np.ceil(in_channel / mini_block))
        self.in_channel_pad = self.grid_dim_x_pad * self.mini_block
        self.out_channel_pad = self.grid_dim_y_pad * self.mini_block
        self.mode = mode
        assert mode in {"weight", "usv", "phase", "voltage"}, logging.error(f"Mode not supported. Expected one from (weight, usv, phase, voltage) but got {mode}.")
        self.v_max = v_max
        self.v_pi = v_pi
        self.gamma = np.pi / self.v_pi**2
        self.w_bit = w_bit
        self.in_bit = in_bit
        self.transform = transform
        self.photodetect = photodetect
        self.device = device

        ### allocate parameters
        self.weight = None
        self.T = None
        self.Tr = None
        self.phase_U = None
        self.phase_S = None
        self.S_scale = S_scale
        self.phase_V = None
        self.voltage_U = None
        self.voltage_S = None
        self.voltage_V = None
        self.unitary_base = torch.cat([
                torch.stack([
                    torch.eye(mini_block,
                              device=self.device),
                    torch.zeros(mini_block, mini_block, device=self.device)], dim=-1),
                torch.stack([
                    torch.zeros(
                        mini_block, mini_block, device=self.device),
                    torch.eye(mini_block, device=self.device)], dim=-1)], 0)
        self.complex_eye = self.unitary_base[:self.mini_block]
        ### build trainable parameters
        self.build_parameters(mode, transform)
        ### unitary parametrization tool
        self.decomposer = RealUnitaryDecomposerBatch()
        ### quantization tool
        self.input_quantizer = PACT_Act(self.in_bit, device=self.device)
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

        self.reset_parameters()

    def build_parameters(self, mode="phase", transform="trainable"):
        ## weight mode (complex-valued)
        self.weight = torch.Tensor(self.grid_dim_y_pad, self.grid_dim_x_pad, self.mini_block, self.mini_block, 2).to(self.device).float()
        ## usv mode (complex-valued)
        ## U and V are shared in this layer
        self.U = torch.Tensor(1, 1, self.mini_block, self.mini_block, 2).to(self.device).float()
        self.S = torch.Tensor(self.grid_dim_y_pad, self.grid_dim_x_pad, self.mini_block, 2).to(self.device).float()
        self.V = torch.Tensor(1, 1, self.mini_block, self.mini_block, 2).to(self.device).float()
        ## phase mode (MZI-implemented magnitude and PS-implemented phase)
        self.T = TrainableButterfly(
                        length=self.mini_block,
                        wbit=self.w_bit,
                        mode="full",
                        bit_reversal=True,
                        enable_last_level_phase_shifter=True,
                        device=self.device)
        self.Tr = TrainableButterfly(
                        length=self.mini_block,
                        wbit=self.w_bit,
                        mode="full_reverse",
                        bit_reversal=True,
                        enable_last_level_phase_shifter=True,
                        device=self.device)
        self.phase_U = self.Tr.phases
        self.phase_S = torch.Tensor(self.grid_dim_y_pad, self.grid_dim_x_pad, self.mini_block, 2).to(self.device).float()
        self.phase_V = self.T.phases
        ## voltage mode
        self.voltage_U = self.Tr.phases.data.clone()
        self.voltage_S = torch.ones_like(self.phase_S)
        self.voltage_V = self.T.phases.data.clone()

        if(mode == 'weight'):
            ### do not know how to train weights in the subspace
            raise NotImplementedError

            # self.weight = Parameter(self.weight)
        elif(mode == "usv"):
            # self.U = Parameter(self.U)
            self.S = Parameter(self.S)
            # self.V = Parameter(self.V)
        elif(mode == "phase"):
            self.phase_S = Parameter(self.phase_S)
        elif(mode == "voltage"):
            self.voltage_U = Parameter(self.voltage_U)
            self.voltage_S = Parameter(self.voltage_S)
            self.voltage_V = Parameter(self.voltage_V)
        else:
            raise NotImplementedError

    def reset_parameters(self):
        if(self.mode == "weight"):
            raise NotImplementedError
        elif(self.mode == "usv"):
            if(self.transform == "trainable_zero"):
                self.set_zero(self.T)
                self.set_zero(self.Tr)
            elif(self.transform == "trainable_fft"):
                self.train_fft(self.T)
                self.train_fft(self.Tr, inverse=True)
                self.phase_U.requires_grad = False
                self.phase_V.requires_grad = False
            else:
                pass
            nn.init.kaiming_normal_(self.S.data)

        elif(self.mode == "phase"):
            if(self.transform == "trainable_zero"):
                self.set_zero(self.T)
                self.set_zero(self.Tr)
            elif(self.transform == "trainable_fft"):
                self.train_fft(self.T)
                self.train_fft(self.Tr, inverse=True)
                self.phase_U.requires_grad = False
                self.phase_V.requires_grad = False
            else:
                pass
            # self.phase_S.data.uniform_(0, 2 * np.pi)
            self.phase_S.data.copy_(nn.init.kaiming_normal_(self.phase_S.data) + np.pi)
            # mag = self.phase_S.norm(p=2, dim=-1).clamp(-1, 1)
            # self.phase_S.data[..., 0].copy_(mag.acos())
            # self.phase_S.data[..., 1].copy_(torch.angle(torch.view_as_complex(self.phase_S.data)))

        elif(self.mode == "voltage"):
            if(self.transform == "trainable_zero"):
                self.set_zero(self.T)
                self.set_zero(self.Tr)
            elif(self.transform == "trainable_fft"):
                self.train_fft(self.T)
                self.train_fft(self.Tr, inverse=True)
                self.phase_U.requires_grad = False
                self.phase_V.requires_grad = False
                self.voltage_U.data.copy_(phase_to_voltage(self.phase_U.data))
                self.voltage_V.data.copy_(phase_to_voltage(self.phase_V.data))
            else:
                pass

            self.voltage_S.data[..., 0].copy_(phase_to_voltage(torch.ones_like(self.voltage_S.data[..., 0]).mul_(1/self.S_scale).acos()))
            self.voltage_S.data[..., 1].zero_()
        else:
            raise NotImplementedError

        if self.bias is not None:
            init.uniform_(self.bias, 0, 0)

    def set_zero(self, model):
        model.phases.data.zero_()
        model.phases.requires_grad = False

    def train_fft(self, model, inverse=False):
        logging.info(f"[I] Start training {'FFT' if inverse == False else 'IFFT'}")
        optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=2e-3)
        x = self.unitary_base
        for step in tqdm(range(2000)):
            if(inverse == False):
                y = torch.ifft(model(x), signal_ndim=1, normalized=True)
            else:
                y = model(torch.fft(x, signal_ndim=1, normalized=True))
            loss = torch.nn.functional.mse_loss(x, y)
            optimizer.zero_grad()
            model.zero_grad()
            loss.backward()
            optimizer.step()
            # metric = ((model.W2.data.matmul(model.W1.data) - torch.eye(N, device=device))**2).mean()
            # metric = ((complex_matmul(model.W2.data, model.W1.data) - torch.eye(N, device=device))**2).mean()
        print(f"[I] Start training {'FFT' if inverse == False else 'IFFT'}, loss={loss.data.item()}")

    def build_weight_from_usv(self, U, S, V):
        ### differentiable feature is gauranteed
        # S = S.clamp(min=-self.S_scale, max=self.S_scale)
        weight = complex_matmul(U, complex_mult(S.unsqueeze(-2),V))
        self.weight = weight
        return weight

    def build_weight_from_phase(self, phase_U, phase_V, phase_S):
        self.U = self.Tr(self.complex_eye, phase_U).unsqueeze(0).unsqueeze(0)
        self.V = self.T(self.complex_eye, phase_V).unsqueeze(0).unsqueeze(0)
        mag = phase_S[..., 0].cos().mul_(self.S_scale)
        if(isinstance(self.S, nn.Parameter)):
            S = torch.stack([mag * phase_S[..., 1].cos(), mag * phase_S[..., 1].sin()], dim=-1)
            self.S.data.copy_(S.data)
        else:
            self.S = torch.stack([mag * phase_S[..., 1].cos(), mag * phase_S[..., 1].sin()], dim=-1)
            S = self.S

        return self.build_weight_from_usv(self.U, S, self.V)

    def build_weight_from_voltage(self, voltage_U, voltage_V, voltage_S, gamma_U, gamma_V, gamma_S):
        self.phase_U = voltage_to_phase(voltage_U, gamma_U)
        self.phase_V = voltage_to_phase(voltage_V, gamma_V)
        self.phase_S = voltage_to_phase(voltage_S, gamma_S)
        return self.build_weight_from_phase(self.phase_U, self.phase_V, self.phase_S)

    def build_phase_from_usv(self, U, S, V):
        raise NotImplementedError

    def build_phase_S_from_S(self, S):
        phase_S = torch.stack([
            S.norm(p=2, dim=-1).clamp(min=-self.S_scale, max=self.S_scale).mul(1/self.S_scale).acos(),
            torch.angle(torch.view_as_complex(S))],dim=-1)
        return phase_S

    def build_S_from_phase_S(self, phase_S):
        S = polar_to_complex(phase_S[..., 0].cos().mul(self.S_scale), phase_S[..., 1])
        return S

    def build_usv_from_weight(self, weight):
        raise NotImplementedError

    def build_phase_from_weight(self, weight):
        raise NotImplementedError

    def build_voltage_from_phase(self, phase_U, phase_V, phase_S):
        self.voltage_U.data.copy_(phase_to_voltage(phase_U, self.gamma))
        self.voltage_S.data.copy_(phase_to_voltage(phase_S, self.gamma))
        self.voltage_V.data.copy_(phase_to_voltage(phase_V, self.gamma))

        return self.voltage_U, self.voltage_V, self.voltage_S

    def build_voltage_from_usv(self, U, S, V):
        raise NotImplementedError

    def build_voltage_from_weight(self, weight):
        raise NotImplementedError

    def sync_parameters(self, src="phase"):
        '''
        description: synchronize all parameters from the source parameters
        '''
        if(src == "weight"):
            self.build_voltage_from_weight(self.weight)
        elif(src== "usv"):
            self.phase_S.data.copy_(self.build_phase_S_from_S(self.S.data))
            self.build_weight_from_phase(self.phase_U, self.phase_V, self.phase_S)
            self.build_voltage_from_phase(self.phase_U, self.phase_V, self.phase_S)
        elif(src == "phase"):
            self.build_weight_from_phase(self.phase_U, self.phase_V, self.phase_S)
            self.build_voltage_from_phase(self.phase_U, self.phase_V, self.phase_S)
        elif(src == "voltage"):
            self.build_weight_from_voltage(self.voltage_U, self.voltage_V, self.voltage_S)
        else:
            raise NotImplementedError

    def build_weight(self):
        if(self.mode == "weight"):
            weight = self.weight
        elif(self.mode == "usv"):
            if(self.w_bit < 16 or self.gamma_noise_std > 1e-5 or self.crosstalk_factor > 1e-5):
                phase_U = self.phase_quantizer(self.phase_U, self.mixedtraining_mask["phase_U"] if self.mixedtraining_mask is not None else None, mode="butterfly")
                phase_V = self.phase_quantizer(self.phase_V, self.mixedtraining_mask["phase_V"] if self.mixedtraining_mask is not None else None, mode="butterfly")
                self.phase_S = self.build_phase_S_from_S(self.S)
                phase_S = self.phase_quantizer(self.phase_S, self.mixedtraining_mask["phase_S"] if self.mixedtraining_mask is not None else None, mode="diagonal")
                S = self.build_S_from_phase_S(phase_S)
            else:
                phase_U = self.phase_U
                phase_V = self.phase_V
                S = absclamp(self.S, min=-self.S_scale, max=self.S_scale)
            self.U = self.Tr(self.complex_eye, phase_U).unsqueeze(0).unsqueeze(0)
            self.V = self.T(self.complex_eye, phase_V).unsqueeze(0).unsqueeze(0)

            weight = self.build_weight_from_usv(self.U, S, self.V)

        elif(self.mode == "phase"):
            if(self.w_bit < 16 or self.gamma_noise_std > 1e-5 or self.crosstalk_factor > 1e-5):
                phase_U = self.phase_quantizer(self.phase_U, self.mixedtraining_mask["phase_U"] if self.mixedtraining_mask is not None else None, mode="butterfly")
                phase_S = self.phase_quantizer(self.phase_S, self.mixedtraining_mask["phase_S"] if self.mixedtraining_mask is not None else None, mode="diagonal")
                phase_V = self.phase_quantizer(self.phase_V, self.mixedtraining_mask["phase_V"] if self.mixedtraining_mask is not None else None, mode="butterfly")
            else:
                phase_U = self.phase_U
                phase_S = self.phase_S
                phase_V = self.phase_V

            weight = self.build_weight_from_phase(phase_U, phase_V, phase_S)
        elif(self.mode == "voltage"):
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
            weight = self.build_weight_from_voltage(voltage_U,voltage_V, voltage_S, gamma_U, gamma_V, gamma_S)
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
            old_param = getattr(self, name)
            old_param.data.copy_(param.view_as(old_param))
        if(self.mode == "phase"):
            self.build_weight()

    def gen_mixedtraining_mask(self, sparsity, random_state=None, enable=True):
        '''
        description: generate sparsity masks for mixed training \\
        param sparsity {float scalar} fixed parameter ratio, valid range: (0,1]
        return mask {dict} a dict with all masks for trainable parameters in the current mode. 1/True is for trainable, 0/False is fixed.
        '''
        if(self.mode == "weight"):
            raise NotImplementedError
            out = {"weight":self.weight.data > percentile(self.weight.data, 100*sparsity)}
        elif(self.mode == "usv"):
            raise NotImplementedError
            ## S is forced with 0 sparsity
            out = {"U": self.U.data > percentile(self.U.data, sparsity*100), "S": torch.ones_like(self.S.data, dtype=torch.bool), "V": self.V.data > percentile(self.V.data, sparsity*100)}
        elif(self.mode == "phase"):
            ## phase_U and phase_V are shared, so tuning every phase is affordable
            ## block selection is based on group Lasso of |phase_S|
            mag = self.phase_S[..., 0].cos().abs()

            mask_U = torch.ones_like(self.phase_U.data, dtype=torch.bool)
            mask_V = torch.ones_like(self.phase_V.data, dtype=torch.bool)
            ### avoid a lot of same value
            mag_n = mag + torch.randn_like(mag)*0.0001
            mask_S = mag_n >= percentile(mag_n, 100*sparsity)
            mask_S = torch.stack([mask_S, mask_S], dim=-1)
            out = {"phase_U": mask_U, "phase_S": mask_S, "phase_V": mask_V}
        elif(self.mode == "voltage"):
            ## selection is based on block-level group lasso
            mag = voltage_to_phase(self.voltage_S[..., 0], self.gamma).cos().abs()
            ### avoid a lot of same value
            mag_n = mag + torch.randn_like(mag)*0.0001
            mask_U = torch.ones_like(self.phase_U.data, dtype=torch.bool)
            mask_V = torch.ones_like(self.phase_V.data, dtype=torch.bool)
            mask_S = mag >= percentile(mag, 100*sparsity)
            mask_S = torch.stack([mask_S, mask_S], dim=-1)

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
        ### input quantization is not enabled
        # if(self.in_bit < 16):
        #     x = self.input_quantizer(x)
        if(not self.fast_forward_flag or self.weight is None):
            weight = self.build_weight()
        else:
            weight = self.weight
        ### complex matrix multiplication
        ###[bs, inc, 2] x [outc, inc, 2]
        weight = merge_chunks(weight, complex=True)[:self.out_channel, :self.in_channel, :] # [outc, inc, 2]
        out = complex_matmul(x, weight.permute(1,0,2))

        if(self.photodetect):
            out = get_complex_energy(out)

        if(self.bias is not None):
            out = out + self.bias.unsqueeze(0)

        return out


class FWHT1D_CPU(torch.autograd.Function):
    @staticmethod
    def transform(cls, tensor):
        """ Simple implementation of FWHT, receiving as input a torch Tensor. """
        bit = length = len(tensor)
        result = tensor.detach().numpy()  # transform to numpy

        for _ in range(int(np.log2(length))):
            bit >>= 1
            for i in range(length):
                if i & bit == 0:
                    j = i | bit
                    temp = result[i]  # this copies by value
                    result[i] += result[j]
                    result[j] = temp - result[j]

        result /= np.sqrt(length)
        return torch.from_numpy(result)  # transform back to torch

    @staticmethod
    def forward(ctx, input):
        return FWHT1D_CPU.transform(input)

    @staticmethod
    def backward(ctx, grad_output):
        return FWHT1D_CPU.transform(grad_output)


class FWHT1D_CUDA(torch.autograd.Function):
    '''Unitary 1D hadamard transform implemented with customized CUDA kernel. Normalization factor is 1/sqrt(N). N is a power of 2
    https://github.com/HazyResearch/structured-nets/blob/master/pytorch/structure/hadamard_cuda'''
    @staticmethod
    def forward(ctx, input):
        return hadamard_cuda.hadamard_transform(input) / np.sqrt(input.size(-1))

    @staticmethod
    def backward(ctx, grad_output):
        return hadamard_cuda.hadamard_transform(grad_output) / np.sqrt(grad_output.size(-1))


class UFT1D_CUDA(torch.autograd.Function):
    '''Unitary 1D universal frequency transform implemented with customized CUDA kernel. Normalization factor is 1/sqrt(N). N is a power of 2
    '''
    @staticmethod
    def forward(ctx, input):
        factor = np.sqrt(input.size(-2))
        res = universal_cuda.universal_transform(input) / factor
        return res

    @staticmethod
    def backward(ctx, grad_output):
        factor = np.sqrt(grad_output.size(-2))
        res = universal_cuda.inverse_universal_transform(grad_output) / factor
        return res


class IUFT1D_CUDA(torch.autograd.Function):
    '''Unitary 1D universal frequency transform implemented with customized CUDA kernel. Normalization factor is 1/sqrt(N). N is a power of 2
    '''
    @staticmethod
    def forward(ctx, input):
        factor = np.sqrt(input.size(-2))
        res = universal_cuda.inverse_universal_transform(input) / factor
        return res

    @staticmethod
    def backward(ctx, grad_output):
        factor = np.sqrt(grad_output.size(-2))
        res = universal_cuda.universal_transform(grad_output) / factor
        return res


class TrainableButterfly(nn.Module):
    def __init__(self,
        length,
        wbit=32,
        mode="full",
        shared_phases=None,
        bit_reversal=True,
        enable_last_level_phase_shifter=True,
        coupler_transmission_factor_t=np.sqrt(2)/2,
        coupler_insertion_loss=0,
        crossing_transmission_factor=1,
        crossing_phase_shift=0,
        phase_noise_std=0,
        device=torch.device("cuda:0")):
        super(TrainableButterfly, self).__init__()
        self.length = length
        self.wbit = wbit
        self.n_level = int(np.log2(length))
        self.coupler_transmission_factor_t = coupler_transmission_factor_t
        self.coupler_insertion_loss = coupler_insertion_loss
        self.phase_noise_std = phase_noise_std
        self.mode = mode
        assert mode in {"reduced", "reduced_reverse", "full", "full_reverse",
                        "full_reverse_enhance", "full_nonideal", "full_reverse_nonideal"}, "[E] Only support reduced, reduced_reverse, full and full_reverse, full_reverse_enhance, full_nonideal, full_reverse_nonideal"
        self.build_coupler_transfer_function = {"reduced": self.build_coupler_transfer_function_1,
                                                "reduced_reverse": self.build_coupler_transfer_function_1,
                                                "full": self.build_coupler_transfer_function_2,
                                                # no need to reverse coupler, reverse is mainly for permutation
                                                "full_reverse": self.build_coupler_transfer_function_2,
                                                "full_reverse_enhance": self.build_coupler_transfer_function_2,
                                                "full_nonideal": self.build_coupler_transfer_function_2_nonideal,
                                                "full_reverse_nonideal": self.build_coupler_transfer_function_2_nonideal}[mode]
        self.bit_reversal = bit_reversal
        self.enable_last_level_phase_shifter = enable_last_level_phase_shifter
        self.crossing_transmission_factor = crossing_transmission_factor
        self.crossing_phase_shift = crossing_phase_shift

        self.device = device
        self.phases = nn.Parameter(torch.zeros(self.n_level+1, length // 2, 2, dtype=torch.float,
                                               device=device)) if shared_phases is None else shared_phases.data
        if("reduce" in mode):
            self.ones = torch.ones(self.n_level, length // 2,
                                   dtype=torch.float, device=device)
            self.zeros = torch.zeros(
                self.n_level, length // 2, dtype=torch.float, device=device)
        self.permutations = ButterflyPermutation(
            length,
            crossing_transmission_factor=crossing_transmission_factor,
            crossing_phase_shift=crossing_phase_shift,
            device=device)
        self.permutation_inverse = {"reduced": False,
                                    "reduced_reverse": True,
                                    "full": False,
                                    "full_reverse": True,
                                    "full_reverse_enhance": True,
                                    "full_nonideal": False,
                                    "full_reverse_nonideal":True}[mode]

        self.reset_parameters()

    def reset_parameters(self):
        # nn.init.uniform_(self.phases, a=0, b=2*np.pi)
        nn.init.uniform_(self.phases, a=-np.pi/2, b=-np.pi/2)
        # nn.init.uniform_(self.phases, a=0, b=0)

    def build_coupler_transfer_function_1(self, phases):
        # phases [n_level, length // 2, 2]
        # phases = phases.data
        cos_phases = torch.cos(phases)  # cos phi1 and cos phi2
        sin_phases = torch.sin(phases)  # sin phi1 and sin phi2
        # phi1 + phi2 [n_level, length // 2]
        phases_sum = torch.sum(phases, dim=-1)
        # [n_level, length // 2]
        cos_phases_1, cos_phases_2 = cos_phases[..., 0], cos_phases[..., 1]
        # [n_level, length // 2]
        sin_phases_1, sin_phases_2 = sin_phases[..., 0], sin_phases[..., 1]
        cos_phases_sum = torch.cos(phases_sum)  # [n_level, length // 2]
        sin_phases_sum = torch.sin(phases_sum)  # [n_level, length // 2]

        weight_real = torch.stack([self.ones, -sin_phases_2, -sin_phases_1, cos_phases_sum], dim=-1).view(
            self.n_level, self.length // 2, 2, 2)  # [n_level, length // 2, 2, 2]
        weight_imag = torch.stack([self.zeros, cos_phases_2, cos_phases_1, sin_phases_sum], dim=-1).view(
            self.n_level, self.length // 2, 2, 2)  # [n_level, length // 2, 2, 2] 2x2 transfer function
        # [n_level, length // 2, 2, 2, comp]
        weights = torch.stack([weight_real, weight_imag], dim=-1)

        return weights

    def build_coupler_transfer_function_2(self, phases):
        # phases [n_level-1, length // 2, 2]
        cos_phases = torch.cos(phases)  # cos phi1 and cos phi2
        sin_phases = torch.sin(phases)  # sin phi1 and sin phi2
        # [n_level-1, length // 2]
        cos_phases_1, cos_phases_2 = cos_phases[..., 0], cos_phases[..., 1]
        # [n_level-1, length // 2]
        sin_phases_1, sin_phases_2 = sin_phases[..., 0], sin_phases[..., 1]
        weight_real = torch.stack([cos_phases_1, -sin_phases_2, -sin_phases_1, cos_phases_2], dim=-1).view(
            self.n_level, self.length // 2, 2, 2)  # [n_level-1, length // 2, 2, 2]
        weight_imag = torch.stack([sin_phases_1, cos_phases_2, cos_phases_1, sin_phases_2], dim=-1).view(
            self.n_level, self.length // 2, 2, 2)  # [n_level-1, length // 2, 2, 2] 2x2 transfer function
        # [n_level-1, length // 2, 2, 2, comp]
        weights = torch.stack([weight_real, weight_imag], dim=-1)
        return weights

    def build_coupler_transfer_function_2_nonideal(self, phases):
        t = self.coupler_transmission_factor_t
        insertion_loss = self.coupler_insertion_loss
        assert 0 <= self.coupler_insertion_loss <= 1, logging.error(f"Insertion loss of coupler should be within [0, 1], but got {self.coupler_insertion_loss}")

        assert 1 - self.coupler_insertion_loss - t**2 >= 0, logging.error(f"Impossible transmission factor of coupler, requires t^2 + k^2 = 1 - insertion_loss, but got t={t}, insertion loss={insertion_loss}")
        k = np.sqrt(1 - insertion_loss - t**2)

        # phases [n_level-1, length // 2, 2]
        cos_phases = torch.cos(phases)  # cos phi1 and cos phi2
        sin_phases = torch.sin(phases)  # sin phi1 and sin phi2
        # [n_level-1, length // 2]
        cos_phases_1, cos_phases_2 = cos_phases[..., 0], cos_phases[..., 1]
        # [n_level-1, length // 2]
        sin_phases_1, sin_phases_2 = sin_phases[..., 0], sin_phases[..., 1]
        weight_real = torch.stack([t*cos_phases_1, -k*sin_phases_2, -k*sin_phases_1, t*cos_phases_2], dim=-1).view(
            self.n_level, self.length // 2, 2, 2)  # [n_level-1, length // 2, 2, 2]
        weight_imag = torch.stack([t*sin_phases_1, k*cos_phases_2, k*cos_phases_1, t*sin_phases_2], dim=-1).view(
            self.n_level, self.length // 2, 2, 2)  # [n_level-1, length // 2, 2, 2] 2x2 transfer function
        # [n_level-1, length // 2, 2, 2, comp]
        weights = torch.stack([weight_real, weight_imag], dim=-1)
        return weights

    def build_coupler_transfer_function_2_reverse(self, phases):
        # phases [n_level, length // 2, 2]
        cos_phases = torch.cos(phases)  # cos phi1 and cos phi2
        sin_phases = torch.sin(phases)  # sin phi1 and sin phi2
        # [n_level, length // 2]
        cos_phases_1, cos_phases_2 = cos_phases[..., 0], cos_phases[..., 1]
        # [n_level, length // 2]
        sin_phases_1, sin_phases_2 = sin_phases[..., 0], sin_phases[..., 1]
        weight_real = torch.stack([cos_phases_1, -sin_phases_1, -sin_phases_2, cos_phases_2], dim=-1).view(
            self.n_level, self.length // 2, 2, 2)  # [n_level, length // 2, 2, 2]
        weight_imag = torch.stack([sin_phases_1, cos_phases_1, cos_phases_2, sin_phases_2], dim=-1).view(
            self.n_level, self.length // 2, 2, 2)  # [n_level, length // 2, 2, 2] 2x2 transfer function
        # [n_level, length // 2, 2, 2, comp]
        weights = torch.stack([weight_real, weight_imag], dim=-1)
        return weights

    def build_extra_phase_shifter_transfer_function(self, phases):
        # phases [1, length //2, 2] -> [1, length]
        phases = phases.view(-1, self.length)
        cos_phases = torch.cos(phases)  # cos phi
        sin_phases = torch.sin(phases)  # sin phi
        weights = torch.stack([cos_phases, sin_phases],
                              dim=-1)  # [1, length, 2]
        return weights

    def propagate_coupler(self, weight, x):
        # weight: [1,       length // 2, 2, 2, comp]
        # x:      [batch,   length // 2, 1, 2, comp]
        x = x.unsqueeze(-3)
        output = complex_mult(weight, x)  # [batch, length // 2, 2, 2, comp]
        output = output.sum(dim=-2)  # [batch, length // 2, 2, comp]
        return output

    def propagate_extra_phase_shifter(self, weight, x):
        # weights [1, length, 2]
        # x       [batch, length, 2]
        output = complex_mult(weight, x)
        return output

    def propagate_butterfly(self, weights, ps_weights, x):
        if(self.bit_reversal):
            x = self.permutations(x, level=-1)
        for level in range(self.n_level):
            # [batch, length // 2, 2, comp]
            x = x.contiguous().view(-1, self.length // 2, 2, 2)
            x = self.propagate_coupler(weights[level:level+1, ...], x)
            # [batch, length, comp]
            x = x.contiguous().view(-1, self.length, 2)
            if(level < self.n_level - 1):
                x = self.permutations(
                    x, level, inverse=self.permutation_inverse)
        if(self.enable_last_level_phase_shifter):
            x = self.propagate_extra_phase_shifter(ps_weights, x)
        if(self.bit_reversal):
            x = self.permutations(x, level=self.n_level - 1)

        return x

    def inject_phase_noise(self, phases, phase_noise_std):
        noise = phases.data.clone().normal_(0, phase_noise_std).clamp_(-0.15, 0.15)
        phases_n = phases + noise
        return phases_n

    def forward(self, x, phases=None):
        shape = x.size()  # [..., length, comp]
        x = x.view(-1, self.length, 2)  # [batch, length, comp]
        phases = self.phases if phases is None else phases

        if(self.phase_noise_std > 1e-5):
            phases = self.inject_phase_noise(phases, self.phase_noise_std)
        weights = self.build_coupler_transfer_function(phases[:-1, ...])
        ps_weights = self.build_extra_phase_shifter_transfer_function(phases[-1:, ...])
        output = self.propagate_butterfly(weights, ps_weights, x)
        if(self.mode in {"full_nonideal", "full_reverse_nonideal"}):
            output = output.contiguous().view(shape)
        else:
            output = output.contiguous().view(shape) / np.sqrt(self.length)
        return output


class ButterflyPermutation(nn.Module):
    def __init__(self,
                length,
                crossing_transmission_factor=1,
                crossing_phase_shift=0,
                device=torch.device("cuda:0")):
        super(ButterflyPermutation, self).__init__()
        self.length = length
        self.crossing_transmission_factor = crossing_transmission_factor
        assert 0 <= crossing_transmission_factor <= 1, logging.error(f"Transmission factor for waveguide crossings must be within [0, 1], but got {crossing_transmission_factor}")
        self.crossing_phase_shift = crossing_phase_shift
        self.n_level = int(np.log2(self.length)) - 1
        self.device = device

        self.forward_indices, self.backward_indices = self.gen_permutation_indices()
        self.bit_reversal_indices = bitreversal_permutation(self.length)
        self.num_crossings = self.calc_num_crossings(self.forward_indices)
        self.crossings = self.gen_crossings(self.num_crossings)

    def gen_permutation_indices(self):
        # forward indices  [1,2,3,4,5,6,7,8] -> [1,5,2,6,3,7,4,8]
        # barkward indices [1,2,3,4,5,6,7,8] -> [1,3,5,7,2,4,6,8]

        forward_indices, backward_indices = [], []
        initial_indices = torch.arange(
            0, self.length, dtype=torch.long, device=self.device)

        for level in range(self.n_level):
            block_size = 2 ** (level + 2)
            indices = initial_indices.view(-1, self.length // block_size, 2,
                                           block_size // 2).transpose(dim0=-2, dim1=-1).contiguous().view(-1)
            forward_indices.append(indices)

            indices = initial_indices.view(-1,
                                           self.length // block_size, block_size)
            indices = torch.cat(
                [indices[..., ::2], indices[..., 1::2]], dim=-1).contiguous().view(-1)
            backward_indices.append(indices)
        return forward_indices, backward_indices

    def calc_num_crossings(self, forward_indices):
        ### num crossings are related to forward indices
        ### for example
        ### from: 0 4 1 5 2 6 3 7
        ### to  : 0 1 2 3 4 5 6 7
        ### get : 0 3 1 2 2 1 3 0
        return [(indices - torch.arange(self.length, device=indices.device)).abs() for indices in forward_indices]

    def gen_crossings(self, num_crossings):
        '''
        @description: transfer matrix of cascaded crossings, modeling its insertion loss and phase shift
        @param num_crossings {list of torch.Tensor} number of crossings for all waveguides [length] * n_level
        @return: crossings {list of torch.Tensor} cascaded crossing transfer function [length, 2] * n_level
        '''
        ### cascaded crossings (t^n)*(e^(n*phi))
        crossings = []
        for n_cross in num_crossings:
            n_cross = n_cross.float()
            mag = self.crossing_transmission_factor ** n_cross
            phase = n_cross * self.crossing_phase_shift
            crossings.append(torch.stack([mag * phase.cos(), mag * phase.sin()], dim=-1))
        return crossings

    def forward(self, x, level, inverse=False):
        if(level == -1 or level == self.n_level):
            output = ButterflyPermutationFunction.apply(
                x, self.bit_reversal_indices)
        else:
            if(inverse == False):
                # output = ButterflyPermutationFunction.apply(x, self.forward_indices[level], self.backward_indices[level])
                output = ButterflyPermutationFunction.apply(
                    x, self.forward_indices[level])
                ## in the original transform, crossings are added after permutation
                output = complex_mult(self.crossings[level][(None,)*(output.dim()-2)], output)

            else:
                # output = ButterflyPermutationFunction.apply(x, self.backward_indices[self.n_level-level-1], self.forward_indices[self.n_level-level-1])
                ## in the reversed transform, crossings are added before permutation
                x = complex_mult(self.crossings[level][(None,)*(x.dim()-2)], x)
                output = ButterflyPermutationFunction.apply(
                    x, self.backward_indices[self.n_level-level-1])

        return output


def bitreversal_permutation(n, device=torch.device("cuda:0")):
    """Return the bit reversal permutation used in FFT.
    Parameter:
        n: integer, must be a power of 2.
    Return:
        perm: bit reversal permutation, numpy array of size n
    """
    m = int(math.log2(n))
    assert n == 1 << m, 'n must be a power of 2'
    perm = np.arange(n).reshape(n, 1)
    for i in range(m):
        n1 = perm.shape[0] // 2
        perm = np.hstack((perm[:n1], perm[n1:]))
    return torch.from_numpy(perm.squeeze(0)).to(device)


class ButterflyPermutationFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, forward_indices):
        ctx.forward_indices = forward_indices
        output = input[..., forward_indices, :]
        return output

    @staticmethod
    def backward(ctx, grad_output):
        forward_indices = ctx.forward_indices
        grad_input = grad_output.clone()
        grad_input[..., forward_indices, :] = grad_input
        return grad_input, None


if __name__ == "__main__":
    device = 'cuda'
    layer = FOLinear(4,4,4, mode="phase", transform="trainable",device=device)
    x = torch.randn(1, 4, 2, device=device)

    y1 = layer(x)
    print(layer.weight, layer.S)
    layer.set_gamma_noise(1.1e-5)
    y2 = layer(x)
    print(y1)
    print(y2)
    res = complex_matmul(layer.U, layer.V)
    print(res)
    res = complex_matmul(layer.U, layer.U.permute(0, 1, 3,2,4))
    print(res)

