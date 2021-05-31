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
import torch.fft

from pyutils import *
from .device import *

sys.path.pop(0)

__all__ = ["MOLGCirculantLinear", "AllPassMORRCirculantLinear"]


class MOLGCirculantLinear(nn.Module):
    '''MORR-based circulant linear layer used in [Gu+, DATE'21] SqueezeLight'''
    def __init__(self,
        in_channel,
        out_channel,
        mini_block,
        bias=None,
        in_bit=32,
        w_bit=32,
        balance_weight=False,
        add_drop=True,
        crosstalk_factor=0.01,
        device=torch.device("cuda")
    ):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.mini_block = mini_block
        self.in_channel_pad = int(np.ceil(in_channel / mini_block).item() * mini_block)
        self.out_channel_pad = int(np.ceil(out_channel / mini_block).item() * mini_block)
        # assert in_channel % mini_block == 0, logging.error(f"in_channel must be multiples of mini block size, but got in_channel={in_channel}, mini_blocm={mini_block}")
        # assert out_channel % mini_block == 0, logging.error(f"out_channel must be multiples of mini block size, but got in_channel={in_channel}, mini_blocm={mini_block}")

        self.in_bit = in_bit
        self.w_bit = w_bit
        self.input_quantizer = input_quantize_fn(in_bit)
        self.weight_quantizer = weight_quantize_fn(w_bit)

        self.balance_weight = balance_weight
        self.add_drop = add_drop
        self.crosstalk_factor = crosstalk_factor
        self.device = device

        self.grid_dim_y = self.out_channel_pad // mini_block
        self.grid_dim_x = self.in_channel_pad // mini_block
        self.x_zero_pad = None

        ### dim0 is the eigens of the circulant matrix, dim1 is the row-wise scaling factor
        self.weight = Parameter(torch.empty(self.grid_dim_y, self.grid_dim_x, self.mini_block, device=self.device))

        # self.sign = torch.nn.Parameter(torch.randn(1, self.grid_dim_y, self.grid_dim_x, 1, device=self.device))
        # self.sign_quantizer = uniform_quantize(1, gradient_clip=True)
        self.scale = torch.nn.Parameter(torch.randn(1, 1, max(1,self.grid_dim_x//2), mini_block, device=self.device))
        self.scale_pad = torch.nn.Parameter(torch.ones(1, 1, 1, mini_block, device=self.device))

        if(bias):
            self.bias = Parameter(torch.Tensor(self.out_channel).to(self.device))
        else:
            self.register_parameter('bias', None)

        self.molg_lambda_to_mag_curve_coeff_1 = torch.tensor([0.2874, -0.4361, 0.2449, 0.02252], device=self.device) # [0,2] very flat
        self.molg_lambda_to_mag_curve_coeff_2 = torch.tensor([-3.115, 5.873, -2.562, 0.6916, 0.1116], device=self.device)
        self.molg_lambda_to_mag_curve_coeff_2_n = torch.tensor([-3.115, 6.586, -3.631, -0.7289, 1], device=self.device)
        self.molg_lambda_to_mag_curve_coeff_3 = torch.tensor([0.9611, -0.01226, 0.04576, 0.07918], device=self.device) # [0, 1] flat
        self.molg_lambda_to_mag_curve_coeff_4 = torch.tensor([[0.6345, 0.5001, 0.1213],[0.3235, 0.5001, 0.3733]], device=self.device) # [0, 2] gaussian full curve
        self.molg_lambda_to_mag_curve_coeff_5 = torch.tensor([0.2127, -1.5, 2.232, -0.06193], device=self.device) # [0, 1] all_pass, increasing curve, near sqrt
        self.molg_lambda_to_mag_curve_coeff_half = torch.tensor([0.02403, -0.2699, 1.17, -2.454, 2.523, -0.08003], device=self.device) # fit in [0, 3.56] all_pass, increasing curve, near sqrt in [0,1]
        self.molg_lambda_to_mag_curve_coeff_half_grad = torch.tensor([0.02403, -0.2699, 1.17, -2.454, 2.523], device=self.device)*torch.tensor([5,4,3,2,1], device=self.device) # fit in [0, 3.56] all_pass, increasing curve, near sqrt in [0,1]
        ### thermal cross coupling matrix
        ### circulant matrix, its Fourier transform can be precomputed, applied to weight matrix
        # self.crosstalk_coupling_matrix = circulant(torch.tensor([1] + [self.crosstalk_factor]*(self.mini_block-1), dtype=torch.float32, device=self.device))
        # self.crosstalk_coupling_matrix_f = torch.rfft(self.crosstalk_coupling_matrix[0], signal_ndim=1, normalized=True, onesided=True)
        self.enable_thermal_crosstalk = False

        ## phase variations
        self.enable_phase_noise = False
        self.phase_noise_std = 0


    def init_weights(self):
        init.kaiming_normal_(self.weight.data)
        # if(self.out_channel_pad > self.out_channel):
        #     self.weight.data[-1, :, self.out_channel-self.out_channel_pad:, :].zero_()
        # if(self.in_channel_pad > self.in_channel):
        #     self.weight.data[:, -1, self.in_channel-self.in_channel_pad:, :].zero_()

        if self.bias is not None:
            init.uniform_(self.bias, 0, 0)

    def get_group_lasso(self):
        ### only consider magnitude as the metric for pruning
        # if(self.weight_real_imag is None):
        #     weight = self.weight
        #     t = self.coupler_transmission_factor_t
        #     k = np.sqrt(1 - self.coupler_insertion_loss - t**2)
        #     weight_real = (-weight[..., 0].sin() - weight[..., 1].sin()) * t * k
        #     weight_imag = (weight[..., 0].cos() + weight[..., 1].cos()) * t * k
        #     self.weight_real_imag = torch.stack([weight_real, weight_imag], dim=-1) # [p, q, k, 2]
        # lasso = torch.norm(get_complex_magnitude(self.weight_real_imag), p=2, dim=2) / self.mini_block
        lasso = torch.norm(self.weight, p=2, dim=2) / self.mini_block
        return lasso

    def input_modulator(self, x):
        ### voltage to power, which is proportional to the phase shift
        return x*x

    def set_crosstalk_coupling_matrix(self, coupling_factor, drop_perc=0):
        assert 0 <= coupling_factor <= 1, logging.error(f"Coupling factor must in [0,1], but got {coupling_factor}")

        self.crosstalk_factor = 1 + max(3, (self.mini_block * (1 - drop_perc) - 1)) * coupling_factor
        # if(crosstalk_noise_std < 1e-5):
        #     self.crosstalk_coupling_matrix = torch.tensor([1] + [self.crosstalk_factor]*(self.mini_block-1), dtype=torch.float32, device=self.device)
        # else:
        #     self.crosstalk_coupling_matrix = torch.zeros(self.mini_block, dtype=torch.float32, device=self.device).normal_(coupling_factor, crosstalk_noise_std).clamp_(coupling_factor-3*crosstalk_noise_std,coupling_factor+3*crosstalk_noise_std)
        #     self.crosstalk_coupling_matrix[0] = 1
        # self.crosstalk_coupling_matrix_f = torch.rfft(self.crosstalk_coupling_matrix, signal_ndim=1, normalized=True, onesided=True)

    def enable_crosstalk(self):
        self.enable_thermal_crosstalk = True

    def disable_crosstalk(self):
        self.enable_thermal_crosstalk = False

    def set_phase_variation(self, phase_noise_std=0):
        self.phase_noise_std = phase_noise_std

    def enable_phase_variation(self):
        self.enable_phase_noise = True

    def disable_phase_variation(self):
        self.enable_phase_noise = False

    def propagate_molg(self, weight, x):
        '''
        @description: propagate through the analytically calculated transfer matrix of molg
        @param weight {torch.Tensor} two phase shifters in the MZI-based attenuators
        @param x {torch.Tensor} complex-valued input
        @return: y {torch.Tensor} output of attenuators
        '''
        ### x : [bs, q, k]
        ### weights: [p, q, k]
        # print("before")
        # print_stat(x)
        x = torch.rfft(x, signal_ndim=1, normalized=True, onesided=True)
        ### normalize weight
        # weight = weight / weight.norm(p=2,dim=-1, keepdim=True)
        if(self.enable_thermal_crosstalk and self.crosstalk_factor > 1):
            weight = weight * self.crosstalk_factor
        weight = torch.rfft(weight, signal_ndim=1, normalized=False, onesided=True)
        x = complex_mult(weight.unsqueeze(0), x.unsqueeze(1))
        x = torch.irfft(x, signal_ndim=1, normalized=True, onesided=True, signal_sizes=[self.mini_block])
        # print("mid")
        # print_stat(x)

        # x = polynomial(x.clamp(0, 1), self.molg_lambda_to_mag_curve_coeff_2)
        # print_stat(x)


        if(self.enable_phase_noise and self.phase_noise_std > 1e-5):
            x = x + torch.zeros_like(x).normal_(0, self.phase_noise_std)

        ### phase: [bs, p, q, k], cannot be larger than 1, which is the valid range of molg lambda
        phase = x
        # x = polynomial(x, self.molg_lambda_to_mag_curve_coeff_2_n)
        x = polynomial(x, self.molg_lambda_to_mag_curve_coeff_half).clamp(0,1)
        # share scale between positive and negative neurons
        scale = self.scale# / self.scale.norm(p=2, dim=2, keepdim=True)
        if(self.grid_dim_x % 2 == 0): #even blocks
            x = x * torch.cat([scale, scale], dim=2)
        else:# odd blocks
            if(self.grid_dim_x > 1):
                x = x * torch.cat([scale, scale, self.scale_pad], dim=2)
            else:
                x = x * self.scale_pad
        # x = F.tanh(x)
        # x = gaussian(x, self.molg_lambda_to_mag_curve_coeff_4)
        # x_n = polynomial(x[:,:,:x.size(0)//2,:], self.molg_lambda_to_mag_curve_coeff_2_n)
        # x_p = polynomial(x[:,:,x.size(0)//2:,:], self.molg_lambda_to_mag_curve_coeff_5)
        # x = torch.cat([x_n, x_p], dim=2)
        # print("after")
        # print_stat(x)

        return x, phase

    def propagate_photodetection(self, x):
        '''
        @description: photodetection, from light to analog voltages
        @param x {torch.Tensor} light signals before photodetection, should be complex-valued
        @param oe_factor {scalar} linear scaling factor between light energy to analog voltage
        @return: y {torch.Tensor} real-valued voltage outputs
        '''
        ### x: [bs, p, q, k] -> [bs, outc]
        if(self.add_drop):
            x = 2 * x - 1
        if(self.balance_weight):
            x = -x[:,:,:x.size(2)//2,:].sum(dim=2) + x[:,:,x.size(2)//2:,:].sum(dim=2)
            # x = x * self.sign_quantizer(self.sign).data
            # x = x.sum(dim=2)
        else:
            if(self.add_drop):
                x = x.sum(dim=2)
            else:
                x = x.sum(dim=2) - x.size(2)//2
        x = x.view(x.size(0), -1)# / self.grid_dim_x
        # x = x.sum(dim=2).view(x.size(0), -1)
        # print("PD")
        # print_stat(x)
        # exit(1)

        return x

    def block_accumulation(self, x, complex=False):
        '''
        @description: accumulation of block-circulant matrix multiplication. Performed after photodetection, thus input will be real-valued.
        @param x {torch.Tensor} real-valued signals after photo-detection [batch_size, grid_dim_y, grid_dim_x, mini_block]
        @return: y {torch.Tensor} linear layer final outputs [batch_size, out_channel]
        '''
        ### x: [bs, p, q, k] -> [bs, ouc]
        if(self.balance_weight and x.size(2) > 1):
            y = (-x[:,:,:x.size(2)//2,...].sum(dim=2)+x[:,:,x.size(2)//2:,...].sum(dim=2))
        else:
            y = x.sum(dim=2).view(x.size(0), -1)
        if(complex):
            y = y.view(y.size(0), -1, 2)
        else:
            y = y.view(y.size(0), -1)
        return y

    def postprocessing(self, x):
        return x

    def forward(self, x):
        '''
        @description: forward a layer of FFT-ONN
        @param x {torch.Tensor} real-valued inputs, which is assumed to be the ideal magnitude modulation within [0, 1]
        @return: out {torch.Tensor} real-valued outputs after photodetection
        '''
        ### partitioning to grid_dim_x segments
        ### x: [bs, inc] -> [bs, q, k]
        # print(x.size())
        x = self.input_quantizer(x)
        weight = self.weight_quantizer(self.weight)
        if(self.in_channel_pad > self.in_channel):
            if(self.x_zero_pad is None or self.x_zero_pad.size(0) != x.size(0)):
                self.x_zero_pad = torch.zeros(x.size(0), self.in_channel_pad - self.in_channel, device=x.device, dtype=x.dtype)
            x = torch.cat([x, self.x_zero_pad], dim=1)

        x = x.view(-1, self.grid_dim_x, self.mini_block)
        # print(x.size())
        ### modulation
        ### assume the real input is the magnitude of the modulator output with fixed phase response
        ### x: [bs, q, k] -> [bs, q, k, 2]
        x = self.input_modulator(x)

        ### propagate through attenuator (weight)
        ### x: [bs, q, k, 2] -> [bs, p, q, k, 2]
        x, phase = self.propagate_molg(weight, x)
        # print(x.size())


        ### propagate through photodetection, from optics to voltages
        ### x: [bs, p, q, k, 2] -> [bs, p, q, k]
        x = self.propagate_photodetection(x)
        # print(x.size())

        ### perform partial product accumulation in electrical domain
        ### x: [bs, p, q, k] -> [bs, outc]
        # x = self.block_accumulation(x, complex=False)

        ### postprocessing before activation
        ### x: [bs, outc] -> [bs, outc]
        # out = self.postprocessing(x)
        if(self.out_channel <self.out_channel_pad):
            x = x[..., :self.out_channel]
        if(self.bias is not None):
            x = x + self.bias.unsqueeze(0)

        return x, phase


class AllPassMORRCirculantLinear(nn.Module):
    '''
    description: All-pass MORR Linear layer, assumes (1) block-circulant matrix (2) differential rails (3) learnable balancing factors. Used for SqueezeLight TCAD extension.
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
        MORRConfig=MORRConfig_20um_MQ,
        ### trainable MORR nonlinearity
        trainable_morr_bias=False,
        trainable_morr_scale=False,
        device=torch.device("cuda")
    ):
        super(AllPassMORRCirculantLinear, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.mode = mode
        self.miniblock = miniblock
        assert mode in {"weight", "phase", "voltage"}, logging.error(f"Mode not supported. Expected one from (weight, phase, voltage) but got {mode}.")

        self.v_max = v_max
        self.v_pi = v_pi
        self.gamma = np.pi / self.v_pi**2
        self.w_bit = w_bit
        self.in_bit = in_bit
        self.MORRConfig = MORRConfig
        self.mrr_a = MORRConfig.attenuation_factor
        self.mrr_r = MORRConfig.coupling_factor
        self.device = device
        self.trainable_morr_bias = trainable_morr_bias
        self.trainable_morr_scale = trainable_morr_scale
        ### calculate FWHM (rad)
        self.morr_fwhm = -4 * np.pi**2 * MORRConfig.radius * MORRConfig.effective_index * (1/MORRConfig.resonance_wavelength-1/(MORRConfig.resonance_wavelength - MORRConfig.bandwidth/2))



        ### allocate parameters
        self.weight = None
        self.x_zero_pad = None
        self.morr_output_scale = None ## learnable balancing factors implelemt by MRRs
        self.morr_input_bias = None ## round-trip phase shift bias within MORR
        self.morr_input_scale = None ## scaling factor for the round-trip phase shift within MORR
        self.morr_gain = (100/(self.in_channel//self.miniblock))**0.5 ## TIA gain, calculated such that output variance is around 1
        ### build trainable parameters
        self.build_parameters(mode)

        ### quantization tool
        # self.input_quantizer = PACT_Act(self.in_bit, device=self.device)
        self.input_quantizer = input_quantize_fn(self.in_bit, device=self.device)
        self.weight_quantizer = weight_quantize_fn(self.w_bit, alg="dorefa_pos") ## [0-1] positive only, maintain the original scale
        self.morr_output_scale_quantizer = weight_quantize_fn(self.w_bit, alg="dorefa_sym") ## [-1,1] full-range [TCAD'21]

        self.voltage_quantizer = voltage_quantize_fn(self.w_bit, self.v_pi ,self.v_max)
        self.phase_quantizer = phase_quantize_fn(self.w_bit, self.v_pi ,self.v_max, gamma_noise_std=0)
        self.mrr_roundtrip_phase_to_tr = mrr_roundtrip_phase_to_tr_func(a=self.mrr_a, r=self.mrr_r, intensity=True)

        ### default set to slow forward
        self.disable_fast_forward()
        ### default set no gamma noise
        self.set_gamma_noise(0)
        ### default set no crosstalk
        self.disable_crosstalk()
        ### default set no phase variation
        self.disable_phase_variation()

        # ### default disable mixed training
        # self.disable_mixedtraining()
        # ### default disable dynamic weight generation
        # self.disable_dynamic_weight()
        # self.eye_b = None
        # self.eye_v = None


        if(bias):
            self.bias = Parameter(torch.Tensor(out_channel).to(self.device))
        else:
            self.register_parameter('bias', None)

        # self.reset_parameters()

    def build_parameters(self, mode="weight"):
        ## weight mode
        self.in_channel_pad = int(np.ceil(self.in_channel / self.miniblock).item() * self.miniblock)
        self.out_channel_pad = int(np.ceil(self.out_channel / self.miniblock).item() * self.miniblock)
        self.grid_dim_y = self.out_channel_pad // self.miniblock
        self.grid_dim_x = self.in_channel_pad // self.miniblock

        if(mode in {'weight'}):
            self.weight = Parameter(torch.ones(self.grid_dim_y, self.grid_dim_x, self.miniblock, device=self.device, dtype=torch.float))
            ### learnable balancing factor (morr_output_scale)
            ### (1) shared between different rows
            ### (2) shared between positive and negative rails
            ### (3) extra 1 padding if each row has odd number of miniblocks
            ### (4) different scaling factors are allowed for each cirulant block [Gu+, DATE'21] (too complicated)
            ### (4.1) In the TCAD version, we use a single scaling factor for each block
            # self.morr_output_scale = Parameter(torch.randn(1, 1, max(1,self.grid_dim_x//2) + 1, self.mini_block, device=self.device)) # deprecated
            self.morr_output_scale = Parameter(torch.randn(1, 1, max(1,self.grid_dim_x//2) + 1, 1, device=self.device))
            if(self.trainable_morr_bias):
                ### initialize with the finest-granularity, i.e., per mini-block [TCAD'21]
                self.morr_input_bias = Parameter(torch.zeros(self.grid_dim_y, self.grid_dim_x, device=self.device, dtype=torch.float))
            if(self.trainable_morr_scale):
                ### initialize with the finest-granularity, i.e., per mini-block [TCAD'21]
                self.morr_input_scale = Parameter(torch.zeros(self.grid_dim_y, self.grid_dim_x, device=self.device, dtype=torch.float))
        elif(mode == "phase"):
            raise NotImplementedError
            self.phase = Parameter(self.phase)
        elif(mode == "voltage"):
            raise NotImplementedError
            self.voltage = Parameter(self.voltage)
        else:
            raise NotImplementedError

    def reset_parameters(self):
        if(self.mode in {"weight"}):
            # init.kaiming_normal_(self.weight.data)
            ### nonlinear curve aware initialization
            morr_uniform_(self.weight, MORRConfig=self.MORRConfig, n_op=self.miniblock, biased=self.w_bit >= 16, gain=2 if self.in_bit < 16 else 1) # quantization needs zero-center
            # print("initialized weight:", end=", ")
            # print_stat(self.weight)
            ### output distribution aware initialization to output scaling factor
            # init.normal_(self.morr_output_scale.data)
            # init.uniform_(self.morr_output_scale, -(100/(self.in_channel//self.miniblock))**0.5, (100/(self.in_channel//self.miniblock))**0.5)
            # init.uniform_(self.morr_output_scale, -1, 1) ## scaling need to performed after quantization
            t1 = mrr_roundtrip_phase_to_tr_fused(torch.tensor([0]).float(), a=self.mrr_a, r=self.mrr_r, intensity=True)
            t2 = mrr_roundtrip_phase_to_tr_fused(torch.tensor([self.morr_fwhm*2.4]).float(), a=self.mrr_a, r=self.mrr_r, intensity=True)
            g = ((t2 - t1) / (2.4*self.morr_fwhm)).item() ## 0~2.4 FWHM slope as a linear approximation
            # print(self.morr_fwhm, g)

            self.sigma_out_scale = 4/(3*self.grid_dim_x**0.5*g*self.morr_fwhm)
            self.sigma_out_scale_quant_gain = None
            # print("calculated output scale std:", self.sigma_out_scale)
            init.normal_(self.morr_output_scale, 0, self.sigma_out_scale)

            if(self.morr_input_bias is not None):
                ### after sigmoid, it is close to 0 bias, but still need randomness
                init.normal_(self.morr_input_bias.data, -4, 0.1)
            if(self.morr_input_scale is not None):
                ### after sigmoid, it cooresponds to 1 scale
                init.zeros_(self.morr_input_scale.data)
        elif(self.mode == "phase"):
           raise NotImplementedError
        elif(self.mode == "voltage"):
           raise NotImplementedError
        else:
            raise NotImplementedError

        if self.bias is not None:
            init.uniform_(self.bias, 0, 0)

    def sync_parameters(self, src="weight"):
        '''
        description: synchronize all parameters from the source parameters
        '''

        raise NotImplementedError

    def build_weight(self):
        if(self.mode in {"weight"}):
            if(self.w_bit < 16):
                ### differentiable quantizer based on STE to enable QAT (Dorefa-Net, arXiv 2016)
                weight = self.weight_quantizer(self.weight)
            else:
                weight = self.weight
        elif(self.mode == "phase"):
            ### not differentiable
            raise NotImplementedError
        elif(self.mode == "voltage"):
            raise NotImplementedError
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

    def input_modulator(self, x):
        ### voltage to power, which is proportional to the phase shift
        return x*x

    def set_crosstalk_coupling_matrix(self, coupling_factor, drop_perc=0):
        ### crosstalk coupling matrix is a symmetric matrix, but the intra-MORR crosstalk can be taken as a round-trip phase shift scaling factor, which is proportional to the number of segments after pruned.
        ### drop-perc is the pruning percentage.
        assert 0 <= coupling_factor <= 1, logging.error(f"Coupling factor must in [0,1], but got {coupling_factor}")

        self.crosstalk_factor = 1 + max(3, (self.mini_block * (1 - drop_perc) - 1)) * coupling_factor

    def enable_crosstalk(self):
        self.enable_thermal_crosstalk = True

    def disable_crosstalk(self):
        self.enable_thermal_crosstalk = False

    def set_phase_variation(self, phase_noise_std=0):
        self.phase_noise_std = phase_noise_std

    def enable_phase_variation(self):
        self.enable_phase_noise = True

    def disable_phase_variation(self):
        self.enable_phase_noise = False

    def enable_trainable_morr_scale(self):
        self.trainable_morr_scale = True

    def disable_trainable_morr_scale(self):
        self.trainable_morr_scale = False

    def enable_trainable_morr_bias(self):
        self.trainable_morr_bias = True

    def disable_trainable_morr_bias(self):
        self.trainable_morr_bias = False

    def propagate_morr(self, weight, x):
        '''
        @description: propagate through the analytically calculated transfer matrix of molg. We implement circulant matrix multiplication using FFT
        @param weight {torch.Tensor} two phase shifters in the MZI-based attenuators
        @param x {torch.Tensor} complex-valued input
        @return: y {torch.Tensor} output of attenuators
        '''
        ### x : [bs, q, k]
        ### weights: [p, q, k]
        # print("weight:", weight)
        # print("input", x)
        # circmult_gt = x.matmul(merge_chunks(circulant(weight)).t())


        # x = torch.rfft(x, signal_ndim=1, normalized=True, onesided=True)
        x = torch.fft.rfft(x, n=self.miniblock, dim=-1, norm="ortho")
        ### normalize weight
        # crosstalk on the weights are much cheaper to compute than on the phase shift
        if(self.enable_thermal_crosstalk and self.crosstalk_factor > 1):
            weight = weight * self.crosstalk_factor
        # weight = torch.rfft(weight, signal_ndim=1, normalized=False, onesided=True)
        weight = torch.fft.rfft(weight, n=self.miniblock, dim=-1, norm="backward")
        # x = complex_mult(weight.unsqueeze(0), x.unsqueeze(1))
        x = weight.unsqueeze(0) * x.unsqueeze(1)
        # x = torch.irfft(x, signal_ndim=1, normalized=True, onesided=True, signal_sizes=[self.miniblock])
        x = torch.fft.irfft(x, n=self.miniblock, dim=-1, norm="ortho")

        # print("circmult", x)
        # print("circmult_gt:", circmult_gt)


        if(self.enable_phase_noise and self.phase_noise_std > 1e-5):
            x = x + torch.zeros_like(x).normal_(0, self.phase_noise_std)


        ### Deprecated: Use polynomial for fast computation and gradient-based sensitivity-regularization [Gu+, DATE'21] (not suitable in TCAD)
        # x = polynomial(x, self.molg_lambda_to_mag_curve_coeff_half).clamp(0,1)

        ### input scaling [TCAD'21], must have valid ranges. too small will have dead neuron and not enough nonlinearity; too large will have larger power, cross-channel crosstalk. [0.1 - 1.1] will be suitable
        if(self.trainable_morr_scale):
            x = x * (torch.sigmoid(self.morr_input_scale.unsqueeze(0).unsqueeze(-1)) + 0.1)
            # print("input scale:", self.morr_input_scale.data)
            # print("scaled PS:", x)
        ### input biasing [TCAD'21], must have valid ranges. too large will have power issue and cross-channel crosstalk. [-2FWHM ~ 0]
        if(self.trainable_morr_bias):
            x = x - 2 * self.morr_fwhm * torch.sigmoid(self.morr_input_bias.unsqueeze(0).unsqueeze(-1))
            # print("input bias:", self.morr_input_bias.data)
            # print("biased PS:", x)
        ### Use theoretical transmission function for trainable MORR nonlinearity [TCAD'21]
        ### x is the phase detuning, x=0 means on-resonance
        ### phase: [bs, p, q, k]
        x = mrr_roundtrip_phase_to_tr_fused(rt_phi=x, a=self.mrr_a, r=self.mrr_r, intensity=True)
        # print("transmission: ", x)
        ### output scaling
        if(self.w_bit < 16):
            morr_output_scale = self.morr_output_scale_quantizer(self.morr_output_scale)
            if(self.sigma_out_scale_quant_gain is None):
                self.sigma_out_scale_quant_gain = self.sigma_out_scale / morr_output_scale.data.std().item()
            morr_output_scale = morr_output_scale.mul(self.sigma_out_scale_quant_gain)### gain factor from Tanh used in quantization
        else:
            morr_output_scale = self.morr_output_scale
        # morr_output_scale = morr_output_scale * self.morr_gain
        scale = morr_output_scale[..., :-1, :]
        scale_pad = morr_output_scale[..., -1:, :]
        # print("output scale: ", scale)
        if(self.grid_dim_x % 2 == 0):
            #even blocks
            x = x * torch.cat([scale, scale], dim=2)
        else:
            # odd blocks
            if(self.grid_dim_x > 1):
                x = x * torch.cat([morr_output_scale, scale], dim=2)
            else:
                x = x * scale_pad

        return x

    def propagate_morr_efficient(self, weight, x):
        '''
        @description: propagate through the analytically calculated transfer matrix of molg. We implement circulant matrix multiplication using fast circ matmul
        @param weight {torch.Tensor} two phase shifters in the MZI-based attenuators
        @param x {torch.Tensor} complex-valued input
        @return: y {torch.Tensor} output of attenuators
        '''
        ### x : [bs, q, k]
        ### weights: [p, q, k]

        ## build circulant weight matrix
        # crosstalk on the weights are much cheaper to compute than on the phase shift
        if(self.enable_thermal_crosstalk and self.crosstalk_factor > 1):
            weight = weight * self.crosstalk_factor
        weight = toeplitz(weight).unsqueeze(0) # [1,  p, q, k, k]
        x = x.unsqueeze(1).unsqueeze(-1)       # [bs, 1, q, k, 1]
        x = weight.matmul(x).squeeze(-1)       # [bs, p, q, k]

        if(self.enable_phase_noise and self.phase_noise_std > 1e-5):
            x = x + torch.zeros_like(x).normal_(0, self.phase_noise_std)

        # print("pre-biased phase:", end=", ")
        # print_stat(x)

        ### input scaling [TCAD'21], must have valid ranges. too small will have dead neuron and not enough nonlinearity; too large will have larger power, cross-channel crosstalk. [0.1 - 1.1] will be suitable
        if(self.trainable_morr_scale):
            x = x * (torch.sigmoid(self.morr_input_scale.unsqueeze(0).unsqueeze(-1)) + 0.1)

        ### input biasing [TCAD'21], must have valid ranges. too large will have power issue and cross-channel crosstalk. [-2FWHM ~ 0]
        if(self.trainable_morr_bias):
            x = x - 2 * self.morr_fwhm * torch.sigmoid(self.morr_input_bias.unsqueeze(0).unsqueeze(-1))

        ### Use theoretical transmission function for trainable MORR nonlinearity [TCAD'21]
        ### x is the phase detuning, x=0 means on-resonance
        ### phase: [bs, p, q, k]
        # print("post-biased phase:", end=", ")
        # print_stat(phase)
        x = self.mrr_roundtrip_phase_to_tr(x) # 3x faster than autograd
        # print("morr transmission:", end=", ")
        # print_stat(x)

        ## implement balancing factor as dot-product
        # print("output scale:", end=", ")
        # print_stat(self.morr_output_scale[..., :-1,:])
        if(self.w_bit < 16):
            morr_output_scale = self.morr_output_scale_quantizer(self.morr_output_scale)
            if(self.sigma_out_scale_quant_gain is None):
                self.sigma_out_scale_quant_gain = self.sigma_out_scale / morr_output_scale.data.std().item()
            morr_output_scale = morr_output_scale.mul(self.sigma_out_scale_quant_gain)### gain factor from Tanh used in quantization
        else:
            morr_output_scale = self.morr_output_scale
        # morr_output_scale = morr_output_scale * self.morr_gain
        scale = morr_output_scale[..., :-1, :]
        scale_pad = morr_output_scale[..., -1:, :]

        # print("morr diff transmission:", end=", ")
        # diff = x[..., :x.size(2)//2,:]-x[..., x.size(2)//2:,:]
        # print_stat(diff)
        if(self.grid_dim_x % 2 == 0):
            #even blocks
            scale = torch.cat([scale, -scale], dim=2) # [1, 1, q, 1]
        else:
            # odd blocks
            if(self.grid_dim_x > 1):
                scale = torch.cat([morr_output_scale, -scale], dim=2) # [1, 1, q, 1]
            else:
                scale = scale_pad # [1, 1, q, 1]
        scale = scale.squeeze(-1).squeeze(0) # [1 ,1, 1, q]
        # print("output scale Q:", end=", ")
        # print_stat(scale[..., :scale.size(-1)//2])
        x = scale.matmul(x) # [1, 1, 1, q] x [bs, p, q, k] = [bs, p, 1, k]
        # x2 = scale[..., :scale.size(-1)//2].matmul(diff)
        # print("output diff:", end=", ")
        # print_stat(x2)
        x = x.view(x.size(0), -1) # [bs, p*k]
        return x

    def propagate_photodetection(self, x):
        '''
        @description: photodetection, from light to analog voltages
        @param x {torch.Tensor} light signals before photodetection, should be complex-valued
        @param oe_factor {scalar} linear scaling factor between light energy to analog voltage
        @return: y {torch.Tensor} real-valued voltage outputs
        '''
        ### x: [bs, p, q, k] -> [bs, outc]
        ### by default we use differential rails with balanced photo-detection. first half is negative, the second half is positive
        ### add-drop MORR is not considered. we only use all-pass MORR
        x = x[:,:,:int(np.ceil(x.size(2)/2)),:].sum(dim=2) - x[:,:,int(np.ceil(x.size(2)/2)):,:].sum(dim=2)

        x = x.view(x.size(0), -1)

        return x

    def block_accumulation(self, x, complex=False):
        '''
        @description: accumulation of block-circulant matrix multiplication. Performed after photodetection, thus input will be real-valued.
        @param x {torch.Tensor} real-valued signals after photo-detection [batch_size, grid_dim_y, grid_dim_x, mini_block]
        @return: y {torch.Tensor} linear layer final outputs [batch_size, out_channel]
        '''
        ### x: [bs, p, q, k] -> [bs, ouc]
        if(self.balance_weight and x.size(2) > 1):
            y = (-x[:,:,:x.size(2)//2,...].sum(dim=2)+x[:,:,x.size(2)//2:,...].sum(dim=2))
        else:
            y = x.sum(dim=2).view(x.size(0), -1)
        if(complex):
            y = y.view(y.size(0), -1, 2)
        else:
            y = y.view(y.size(0), -1)
        return y

    def postprocessing(self, x):
        return x

    def forward_slow(self, x):
        assert x.size(-1) == self.in_channel, f"[E] Input dimension does not match the weight size {self.out_channel, self.in_channel}, but got input size ({tuple(x.size())}))"
        if(self.in_bit < 16):
            x = self.input_quantizer(x)
        # if(not self.fast_forward_flag or self.weight is None):
        #     weight = self.build_weight()
        # else:
        #     weight = self.weight #.view(self.out_channel, -1)[:, :self.in_channel]

        weight = self.build_weight()
        if(self.in_channel_pad > self.in_channel):
            if(self.x_zero_pad is None or self.x_zero_pad.size(0) != x.size(0)):
                self.x_zero_pad = torch.zeros(x.size(0), self.in_channel_pad - self.in_channel, device=x.device, dtype=x.dtype)
            x = torch.cat([x, self.x_zero_pad], dim=1)

        x = x.view(-1, self.grid_dim_x, self.miniblock)
        # print(x.size())
        ### modulation
        ### assume the real input is the magnitude of the modulator output with fixed phase response
        ### x: [bs, q, k] -> [bs, q, k, 2]
        x = self.input_modulator(x)

        ### propagate through attenuator (weight)
        ### x: [bs, q, k, 2] -> [bs, p, q, k, 2]
        x = self.propagate_morr(weight, x)
        # print(x.size())

        ### propagate through photodetection, from optics to voltages
        ### x: [bs, p, q, k, 2] -> [bs, p, q, k]
        x = self.propagate_photodetection(x)
        # print(x.size())

        ### postprocessing before activation
        ### x: [bs, outc] -> [bs, outc]
        # out = self.postprocessing(x)

        if(self.out_channel < self.out_channel_pad):
            x = x[..., :self.out_channel]
        if(self.bias is not None):
            x = x + self.bias.unsqueeze(0)

        return x

    def forward(self, x):
        # ~20x faster than forward slow
        assert x.size(-1) == self.in_channel, f"[E] Input dimension does not match the weight size {self.out_channel, self.in_channel}, but got input size ({tuple(x.size())}))"
        if(self.in_bit < 16):
            x = self.input_quantizer(x)
        # print("input Q:", end=", ")
        # print_stat(x)

        weight = self.build_weight()
        if(self.in_channel_pad > self.in_channel):
            if(self.x_zero_pad is None or self.x_zero_pad.size(0) != x.size(0)):
                self.x_zero_pad = torch.zeros(x.size(0), self.in_channel_pad - self.in_channel, device=x.device, dtype=x.dtype)
            x = torch.cat([x, self.x_zero_pad], dim=1)

        x = x.view(-1, self.grid_dim_x, self.miniblock)

        ### modulation
        ### assume the real input is the magnitude of the modulator output with fixed phase response
        ### x: [bs, q, k] -> [bs, q, k, 2]
        x = self.input_modulator(x)

        ### propagate through morr array (weight)
        ### x: [bs, q, k] -> [bs, p*k]
        x = self.propagate_morr_efficient(weight, x)

        if(self.out_channel < self.out_channel_pad):
            x = x[..., :self.out_channel]
        if(self.bias is not None):
            x = x + self.bias.unsqueeze(0)

        return x
