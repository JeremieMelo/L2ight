
import logging
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', ".."))

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Parameter, init

from pyutils.mzi_op import \
    voltage_quantize_prune_with_gamma_noise_of_unitary_fn
from .mzi_linear import MZILinear
from pyutils import *
sys.path.pop(0)


__all__ = ["MZIConv2d"]
class MZIConv2d(MZILinear):
    '''
    description: SVD-based 2D convolution using im2col.
    '''
    def __init__(self,
        in_channel,
        out_channel,
        kernel_size=3,
        stride=1,
        padding=1,
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
        super(MZIConv2d, self).__init__(
            in_channel * kernel_size * kernel_size,
            out_channel,
            bias=bias,
            S_trainable=S_trainable,
            S_scale=S_scale,
            mode=mode,
            v_max=v_max,
            v_pi=v_pi,
            w_bit=w_bit,
            in_bit=in_bit,
            photodetect=photodetect,
            device=device
        )
        self.in_channel_conv = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def get_output_dim(self, img_height, img_width):
        h_out = (img_height - self.kernel_size + 2 * self.padding) / self.stride + 1
        w_out = (img_width - self.kernel_size + 2 * self.padding) / self.stride + 1
        return (int(h_out), int(w_out))

    def forward(self, x):
        if(self.in_bit < 16):
            x = self.input_quantizer(x)
        if(not self.fast_forward_flag or self.weight is None):
            weight = self.build_weight() # [outc, inc * ks * ks]
        else:
            weight = self.weight # [outc, inc * ks * ks]
        weight = weight.view(self.out_channel, self.in_channel_conv, self.kernel_size, self.kernel_size)

        out = F.conv2d(x, weight, bias=None, stride=self.stride, padding=self.padding)
        if(self.photodetect):
            out = out.square()
        if(self.bias is not None):
            out = out + self.bias.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        return out

class MZIConv2dDeprecated(nn.Module):

    '''
    description: SVD-based 2D convolution using im2col.
    '''
    def __init__(self,
        in_channel,
        out_channel,
        kernel_size=3,
        stride=1,
        padding=1,
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
        super(MZIConv2d, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.in_channel_unroll = self.in_channel*self.kernel_size*self.kernel_size
        self.S_trainable = S_trainable
        self.S_scale = S_scale
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
        self.phase_V = None
        self.phase_S = None
        self.S_scale = S_scale
        self.voltage_U = None
        self.voltage_V = None
        self.voltage_S = None
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

        if(bias):
            self.bias = Parameter(torch.Tensor(out_channel).to(self.device))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def build_parameters(self, mode="weight"):
        if(mode == 'weight'):
            self.weight = Parameter(torch.Tensor(self.out_channel, self.in_channel_unroll).to(self.device))
        elif(mode == "usv"):
            self.U = Parameter(torch.Tensor(self.out_channel, self.out_channel).to(
                self.device).to(torch.float32))
            self.S = Parameter(torch.Tensor(min(self.out_channel, self.in_channel_unroll)).to(
                self.device).to(torch.float32), requires_grad=self.S_trainable)
            self.V = Parameter(torch.Tensor(self.in_channel_unroll, self.in_channel_unroll).to(
                self.device).to(torch.float32))
        elif(mode == "phase"):
            ### phases are on CPU
            self.delta_list_U = torch.Tensor(self.out_channel)
            self.phase_U = Parameter(torch.Tensor(self.out_channel*(self.out_channel-1)//2))
            self.phase_S = Parameter(torch.Tensor(min(self.out_channel, self.in_channel_unroll)).to(
                self.device).to(torch.float32), requires_grad=self.S_trainable)
            self.delta_list_V = torch.Tensor(self.in_channel_unroll)
            self.phase_V = Parameter(torch.Tensor(self.in_channel_unroll*(self.in_channel_unroll-1)//2))
        elif(mode == "voltage"):
            ### voltages are on CPU
            self.delta_list_U = torch.Tensor(self.out_channel)
            self.voltage_U = Parameter(torch.Tensor(self.out_channel*(self.out_channel-1)//2))
            self.voltage_S = Parameter(torch.Tensor(min(self.out_channel, self.in_channel_unroll)).to(
                self.device).to(torch.float32), requires_grad=self.S_trainable)
            self.delta_list_V = torch.Tensor(self.in_channel_unroll)
            self.voltage_V = Parameter(torch.Tensor(self.in_channel_unroll*(self.in_channel_unroll-1)//2))
        else:
            raise NotImplementedError

    def reset_parameters(self):
        if(self.mode == "weight"):
            init.kaiming_normal_(self.weight.data)
        elif(self.mode == "usv"):
            W = init.kaiming_normal_(torch.empty(self.out_channel, self.in_channel_unroll)).numpy().astype(np.float64)
            U, S, V = np.linalg.svd(W, full_matrices=True)
            self.U.data.copy_(torch.from_numpy(U).float().to(self.device))
            self.V.data.copy_(torch.from_numpy(V).float().to(self.device))
            self.S.data.copy_(torch.from_numpy(S).float().to(self.device))
        elif(self.mode == "phase"):
            W = init.kaiming_normal_(torch.empty(self.out_channel, self.in_channel_unroll)).numpy().astype(np.float64)
            U, S, V = np.linalg.svd(W, full_matrices=True)
            delta_list, phi_mat = self.decomposer.decompose(U)
            self.delta_list_U = torch.from_numpy(delta_list)
            self.phase_U.data.copy_(torch.from_numpy(upper_triangle_to_vector_cpu(phi_mat)).float())
            delta_list, phi_mat = self.decomposer.decompose(V)
            self.delta_list_V = torch.from_numpy(delta_list)
            self.phase_V.data.copy_(torch.from_numpy(upper_triangle_to_vector_cpu(phi_mat)).float())
            self.phase_S.data.copy_(torch.from_numpy(S).float().to(self.device).clamp_(max=self.S_scale).mul_(1/self.S_scale).acos())
        elif(self.mode == "voltage"):
            W = init.kaiming_normal_(torch.empty(self.out_channel, self.in_channel_unroll)).numpy().astype(np.float64)
            U, S, V = np.linalg.svd(W, full_matrices=True)
            delta_list, phi_mat = self.decomposer.decompose(U)
            self.delta_list_U = torch.from_numpy(delta_list)
            self.voltage_U.data.copy_(torch.from_numpy(phase_to_voltage_cpu(upper_triangle_to_vector_cpu(phi_mat), self.gamma)).float())
            delta_list, phi_mat = self.decomposer.decompose(V)
            self.delta_list_V = torch.from_numpy(delta_list)
            self.voltage_V.data.copy_(torch.from_numpy(phase_to_voltage_cpu(upper_triangle_to_vector_cpu(phi_mat), self.gamma)).float())
            self.voltage_S.data.copy_(phase_to_voltage(torch.from_numpy(S).float().to(self.device).clamp_(max=self.S_scale).mul_(1/self.S_scale).arccos()))
        else:
            raise NotImplementedError

        if self.bias is not None:
            init.uniform_(self.bias, 0, 0)

    def build_weight_from_usv(self, U, S, V):
        if(self.out_channel < self.in_channel_unroll):
            self.weight = torch.mm(U*S.unsqueeze(0), V[:self.out_channel, :])
        elif(self.out_channel > self.in_channel_unroll):
            self.weight = torch.mm(U[:, :self.in_channel_unroll], S.unsqueeze(1)*V)
        else:
            self.weight = torch.mm(U, S.unsqueeze(1)*V)

        return self.weight

    def build_weight_from_phase(self, delta_list_U, phase_U, delta_list_V, phase_V, phase_S):
        self.U = torch.from_numpy(self.decomposer.reconstruct_2(delta_list_U.data.cpu().numpy(), vector_to_upper_triangle_cpu(phase_U.data.cpu().numpy()))).float().to(self.device)
        self.V = torch.from_numpy(self.decomposer.reconstruct_2(delta_list_V.data.cpu().numpy(), vector_to_upper_triangle_cpu(phase_V.data.cpu().numpy()))).float().to(self.device)
        self.S = phase_S.cos().mul_(self.S_scale)
        return self.build_weight_from_usv(self.U, self.S, self.V)

    def build_weight_from_voltage(self, delta_list_U, voltage_U, delta_list_V, voltage_V, voltage_S, gamma_U, gamma_V, gamma_S):
        self.phase_U = voltage_to_phase(voltage_U, gamma_U)
        self.phase_V = voltage_to_phase(voltage_V, gamma_V)
        self.phase_S = voltage_to_phase(voltage_S, gamma_S)
        return self.build_weight_from_phase(delta_list_U, self.phase_U, delta_list_V, self.phase_V, self.phase_S)

    def build_weight(self):
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
            if(self.w_bit < 16 or self.gamma_noise_std > 1e-5):
                phase_U = self.phase_quantizer(self.phase_U)
                phase_S = self.phase_quantizer(self.phase_S)
                phase_V = self.phase_quantizer(self.phase_V)
            else:
                phase_U = self.phase_U
                phase_S = self.phase_S
                phase_V = self.phase_V

            weight = self.build_weight_from_phase(self.delta_list_U, phase_U, self.delta_list_V, phase_V, phase_S)
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
        return weight.view(-1, self.in_channel, self.kernel_size, self.kernel_size)

    def get_output_dim(self, img_height, img_width):
        h_out = (img_height - self.kernel_size + 2 * self.padding) / self.stride + 1
        w_out = (img_width - self.kernel_size + 2 * self.padding) / self.stride + 1
        return (int(h_out), int(w_out))

    def enable_fast_forward(self):
        self.fast_forward_flag = True

    def disable_fast_forward(self):
        self.fast_forward_flag = False

    def set_gamma_noise(self, noise_std):
        self.gamma_noise_std = noise_std
        self.unitary_quantizer.gamma_noise_std = noise_std
        self.phase_quantizer.gamma_noise_std = noise_std
        self.diag_quantizer.gamma_noise_std = noise_std
        self.voltage_quantizer.gamma_noise_std = noise_std

    def load_parameters(self, param_dict):
        for name, param in param_dict.items():
            getattr(self, name).data.copy_(param)

    def forward(self, x):
        if(self.in_bit < 16):
            x = self.input_quantizer(x)
        if(not self.fast_forward_flag or self.weight is None):
            self.weight = self.build_weight()
        out = F.conv2d(x, self.weight, bias=None, stride=self.stride, padding=self.padding)
        if(self.photodetect):
            out = out.square()
        if(self.bias is not None):
            out = out + self.bias.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

        return out
