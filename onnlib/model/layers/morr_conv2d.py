
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

__all__ = ["MOLGCirculantConv2d", "AllPassMORRCirculantConv2d"]


class MOLGCirculantConv2d(nn.Module):
    '''MORR-based circulant conv2d layer used in [Gu+, DATE'21] SqueezeLight'''
    def __init__(
        self,
        input_channel,
        output_channel,
        kernel_size,
        miniblock,
        stride=1,
        padding=0,
        in_bit=16,
        w_bit=16,
        bias=False,
        balance_weight=False,
        add_drop=True,
        crosstalk_factor=0.01,
        device=torch.device("cuda:0")
    ):
        super().__init__()
        self.stride = stride
        self.padding = padding
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.kernel_size = kernel_size
        self.miniblock = miniblock
        self.in_bit = in_bit
        self.w_bit = w_bit
        self.balance_weight = balance_weight
        self.add_drop = add_drop
        self.crosstalk_factor = crosstalk_factor
        self.device = device
        self.input_quantizer = input_quantize_fn(in_bit)
        self.weight_quantizer = weight_quantize_fn(w_bit)

        self.output_channel_pad = int(np.ceil(output_channel / miniblock).item() * miniblock)
        self.weight_input_channel = kernel_size * kernel_size * input_channel
        self.weight_input_channel_pad = int (np.ceil(self.weight_input_channel / miniblock).item()) * miniblock
        self.grid_dim_y = self.output_channel_pad // miniblock
        self.grid_dim_x = self.weight_input_channel_pad // miniblock
        self.x_zero_pad = None

        ## unrolled weight with padding
        self.weight = torch.nn.Parameter(torch.zeros(self.grid_dim_y, self.grid_dim_x, miniblock, device=self.device))
        # self.sign = torch.nn.Parameter(torch.randn(1, self.grid_dim_y, self.grid_dim_x, 1, device=self.device))
        # self.sign_quantizer = uniform_quantize(1, gradient_clip=True)
        self.scale = torch.nn.Parameter(torch.randn(1, 1, max(1, self.grid_dim_x//2), miniblock, device=self.device))
        self.scale_pad = torch.nn.Parameter(torch.ones(1, 1, 1, miniblock, device=self.device))
        self.bias = bias

        if bias:
            self.b = torch.nn.Parameter(torch.zeros(self.output_channel, device=self.device))
        else:
            self.b = None
        self.init_weights()
        self.molg_lambda_to_mag_curve_coeff = torch.tensor([0.2874, -0.4361, 0.2449, 0.02252], device=self.device) # [0,2] very flat
        self.molg_lambda_to_mag_curve_coeff_2 = torch.tensor([-3.115, 5.873, -2.562, 0.6916, 0.1116], device=self.device)
        self.molg_lambda_to_mag_curve_coeff_2_n = torch.tensor([-3.115, 6.586, -3.631, -0.7289, 1], device=self.device)
        self.molg_lambda_to_mag_curve_coeff_3 = torch.tensor([0.9611, -0.01226, 0.04576, 0.07918], device=self.device) # [0, 1] flat
        self.molg_lambda_to_mag_curve_coeff_4 = torch.tensor([[0.6345, 0.5001, 0.1213],[0.3235, 0.5001, 0.3733]], device=self.device) # [0, 2] gaussian full curve
        self.molg_lambda_to_mag_curve_coeff_5 = torch.tensor([0.2127, -1.5, 2.232, -0.06193], device=self.device) # [0, 1] all_pass, increasing curve, near sqrt
        self.molg_lambda_to_mag_curve_coeff_half = torch.tensor([0.02403, -0.2699, 1.17, -2.454, 2.523, -0.08003], device=self.device) # fit in [0, 3.56] all_pass, increasing curve, near sqrt in [0,1]
        self.molg_lambda_to_mag_curve_coeff_half_grad = torch.tensor([0.02403, -0.2699, 1.17, -2.454, 2.523], device=self.device)*torch.tensor([5,4,3,2,1], device=self.device) # fit in [0, 3.56] all_pass, increasing curve, near sqrt in [0,1]

        ### thermal cross coupling matrix
        ### circulant matrix, its Fourier transform can be precomputed, applied to weight matrix
        # self.crosstalk_coupling_matrix = torch.tensor([1] + [self.crosstalk_factor]*(self.miniblock-1), dtype=torch.float32, device=self.device)
        # self.crosstalk_coupling_matrix_f = torch.rfft(self.crosstalk_coupling_matrix, signal_ndim=1, normalized=True, onesided=True)
        self.enable_thermal_crosstalk = False

        ## phase variations
        self.enable_phase_noise = False
        self.phase_noise_std = 0

    def init_weights(self):
        set_torch_deterministic()
        nn.init.kaiming_normal_(self.weight)
        # _, self.scale = quant_kaiming_uniform_(self.weight, self.w_bit, beta=1.5)
        if self.bias:
            nn.init.uniform_(self.b)

    def assign_engines(self, out_par=1, image_par=1, phase_noise_std=0, disk_noise_std=0, deterministic=False):
        if(phase_noise_std > 1e-4):
            if(deterministic):
                set_torch_deterministic()
            self.phases = torch.cat([torch.zeros(out_par, image_par, self.input_channel//2, self.kernel_size, self.kernel_size, device=self.device).normal_(mean=-np.pi/2, std=phase_noise_std), torch.zeros(out_par, image_par, self.input_channel-self.input_channel//2, self.kernel_size, self.kernel_size, device=self.device).normal_(mean=np.pi/2, std=phase_noise_std)], dim=2)
        else:
            self.phases = torch.cat([torch.zeros(1, 1, self.input_channel//2, self.kernel_size, self.kernel_size, device=self.device) + 3 * np.pi / 2, torch.zeros(1, 1, self.input_channel-self.input_channel//2, self.kernel_size, self.kernel_size, device=self.device) + np.pi / 2],dim=2)
        if(disk_noise_std > 1e-5):
            if(deterministic):
                set_torch_deterministic()
            self.disks = 1 - torch.zeros(out_par, image_par, self.input_channel, self.kernel_size, self.kernel_size, 2, device=self.device).normal_(mean=0, std=disk_noise_std).abs()
        else:
            self.disks = torch.ones(1, 1, self.input_channel, self.kernel_size, self.kernel_size, 2, device=self.device)
        self.assigned = True

    def deassign_engines(self):
        self.phases = torch.cat([torch.zeros(1, 1, input_channel//2, kernel_size, kernel_size, device=self.device) + 3 * np.pi / 2, torch.zeros(1, 1, input_channel-input_channel//2, kernel_size, kernel_size, device=self.device) + np.pi / 2],dim=2)
        self.disks = torch.ones(1, 1, self.input_channel, self.kernel_size, self.kernel_size, 2, device=self.device)
        self.assigned = False

    def static_pre_calibration(self):
        coeff = 0.25
        N = self.phases.size(2)*self.phases.size(3)*self.phases.size(4)
        sin_phi = self.phases.sin()
        self.beta0 = self.disks[...,0].sum(dim=(2,3,4)) # [out_par, image_par]
        self.beta1 = self.disks[...,1].sum(dim=(2,3,4)) # [out_par, image_par]
        self.beta2 = (sin_phi*(self.disks[...,0]+self.disks[...,1])/2).mean(dim=(2,3,4)) # [out_par, image_par]
        if(self.mode == "oadder"):
            self.correction = 2 * N * coeff * (1 - self.beta2)
        elif(self.mode == "oconv"):
            self.correction_disk_0 = self.disks[...,0].mean(dim=(2,3,4))
            self.correction_disk_1 = self.disks[...,1].mean(dim=(2,3,4))
            self.correction_phi = self.beta2 / self.disks.mean(dim=(2,3,4,5))
        else:
            raise NotImplementedError
        # return self.correction # [out_par, image_par]

    def enable_calibration(self):
        self.calibration = True

    def disable_calibration(self):
        self.calibration = False

    def apply_calibration(self, x):
        # x [outc, h*w*bs]

        x_size = x.size()
        bs = x.size(0)
        n_out_tile, out_tile_size = self.phases.size(0), x.size(0) // self.phases.size(0)
        n_image_tile, image_tile_size = self.phases.size(1), x.size(1) // self.phases.size(1)
        # correction = self.correction.unsqueeze(1).unsqueeze(-1) # [out_par, 1, image_par, 1]
        x = x.view(n_out_tile, out_tile_size, n_image_tile, image_tile_size)
        if(self.mode == "oconv"):
            correction_disk_0 = self.correction_disk_0.unsqueeze(1).unsqueeze(-1) # [out_par, 1, image_par, 1]
            correction_disk_1 = self.correction_disk_1.unsqueeze(1).unsqueeze(-1) # [out_par, 1, image_par, 1]
            correction_phi = self.correction_phi.unsqueeze(1).unsqueeze(-1) # [out_par, 1, image_par, 1]

        elif(self.mode == "oadder"):
            x = x + correction
        else:
            raise NotImplementedError
        x = x.view(x_size)
        return x

    def tiling_matmul(self, W, X, sin_phases, disks):
        n_out_tile, out_tile_size = sin_phases.size(0), W.size(0) // sin_phases.size(0)
        n_image_tile, image_tile_size = sin_phases.size(1), X.size(1) // sin_phases.size(1)
        W = W.unsqueeze(1) # [outc, 1, in_c*kernel_size*kernel_size]
        W = W.view(n_out_tile, out_tile_size, 1, -1) # [n_out_tile,out_tile_size,1, in_c*kernel_size*kernel_size]
        X = X.view(X.size(0), n_image_tile, image_tile_size).permute(1, 2, 0).contiguous().unsqueeze(0) # [1, n_image_tile, image_tile_size, inc*kernel_size*kernel_size]

        factor_mul = sin_phases * (disks[...,0]+disks[...,1]) / 2
        factor_mul = factor_mul.view(n_out_tile, n_image_tile, -1).unsqueeze(1) # [n_out_tile,1,n_image_tile, in_c*kernel_size*kernel_size]

        # inject additive error
        factor_add = (disks[..., 0] - disks[..., 1]) / 4
        factor_add = factor_add.view(n_out_tile, n_image_tile, -1).unsqueeze(1) # [n_out_tile,1,n_image_tile, in_c*kernel_size*kernel_size]
        additive_error = ((W * W) * factor_add).sum(dim=-1).unsqueeze(3) # [n_out_tile, out_tile_size, n_image_tile, 1]
        factor_add = factor_add.squeeze(1).unsqueeze(2) # [n_out_tile, n_image_tile, 1, in_c*kernel_size*kernel_size]
        additive_error = additive_error + ((X * X) * factor_add).sum(dim=-1).unsqueeze(1) # [n_out_tile, out_tile_size, n_image_tile, 1]+[n_out_tile, 1, n_image_tile, image_tile_size]->[n_out_tile, out_tile_size, n_image_tile, image_tile_size]
        additive_error = additive_error.view(self.output_channel, -1) # [outc, h*w*bs]

        W = (W * factor_mul).view(self.output_channel, n_image_tile, -1).permute(1,0,2).contiguous() # # [n_image_tile,outc,in_c*kernel_size*kernel_size]
        X = X.squeeze(0).permute(0,2,1).contiguous() # X [n_image_tile, in_c*kernel_size*kernel_size, image_tile_size]

        out = W.bmm(X).transpose(0,1).contiguous().view(self.output_channel, -1) # [outc, h*w*bs]
        out = out + additive_error
        return out

    def tiling_matmul_with_oconv_calibration(self, W, X, sin_phases, disks):
        n_out_tile, out_tile_size = sin_phases.size(0), W.size(0) // sin_phases.size(0)
        n_image_tile, image_tile_size = sin_phases.size(1), X.size(1) // sin_phases.size(1)
        disks = self.disks.view(n_out_tile, n_image_tile, -1 , 2).unsqueeze(1) # [n_out_tile, 1, n_image_tile, inc*ks*ks, 2]
        W = W.unsqueeze(1) # [outc, 1, in_c*kernel_size*kernel_size]
        W = W.view(n_out_tile, out_tile_size, 1, -1) # [n_out_tile,out_tile_size,1, in_c*kernel_size*kernel_size]
        # W_sq = (W * W).unsqueeze(-1) # [n_out_tile,out_tile_size,1, in_c*kernel_size*kernel_size, 1]
        W_sq_sum = ((W * W).unsqueeze(-1)*disks).sum(dim=-2).unsqueeze(3) # [n_out_tile, out_tile_size, n_image_tile, 1, 2] for two rails


        X = X.view(X.size(0), n_image_tile, image_tile_size).permute(1, 2, 0).contiguous().unsqueeze(0) # [1, n_image_tile, image_tile_size, inc*kernel_size*kernel_size]
        # X_sq = (X * X).unsqueeze(-1)
        # disks = disks.transpose(1,2).contiguous() # [n_out_tile, n_image_tile, 1, inc*ks*ks, 2]
        X_sq_sum = ((X * X).unsqueeze(-1) * disks.transpose(1,2).contiguous()).sum(dim=-2).unsqueeze(1) # [n_out_tile,1, n_image_tile, image_tile_size, 2]

        factor_mul = sin_phases.unsqueeze(-1) * self.disks
        factor_mul = factor_mul.view(n_out_tile, n_image_tile, -1, 2).unsqueeze(1) # [n_out_tile,1,n_image_tile, in_c*kernel_size*kernel_size, 2]

        W = (W.unsqueeze(-1) * factor_mul).view(self.output_channel, n_image_tile, -1, 2).permute(1,0,2,3).contiguous() # # [n_image_tile,outc,in_c*kernel_size*kernel_size, 2]
        X = X.squeeze(0).permute(0,2,1).contiguous() # X [n_image_tile, in_c*kernel_size*kernel_size, image_tile_size]
        two_X_W_sin_phi_alpha_0 = 2*W[...,0].bmm(X).view(n_image_tile, n_out_tile, out_tile_size, image_tile_size).permute(1,2,0,3).contiguous()
        two_X_W_sin_phi_alpha_1 = 2*W[...,1].bmm(X).view(n_image_tile, n_out_tile, out_tile_size, image_tile_size).permute(1,2,0,3).contiguous() #  [n_out_tile, out_tile_size, n_image_tile, image_tile_size]
        W_sq_plus_X_sq_sum = W_sq_sum + X_sq_sum # [n_out_tile, out_tile_size, n_image_tile, image_tile_size, 2]
        rail_0 = (W_sq_plus_X_sq_sum[..., 0] + two_X_W_sin_phi_alpha_0) / (4 * self.correction_disk_0.unsqueeze(1).unsqueeze(-1))
        rail_1 = (W_sq_plus_X_sq_sum[..., 1] - two_X_W_sin_phi_alpha_1) / (4 * self.correction_disk_1.unsqueeze(1).unsqueeze(-1))
        # print(rail_0.size(), rail_1.size())
        out = (rail_0 - rail_1) / self.correction_phi.unsqueeze(1).unsqueeze(-1)
        out = out.view(self.output_channel, -1)

        return out

    def adder2d_function(self, X, W, stride=1, padding=0):
        n_filters, d_filter, h_filter, w_filter = W.size()
        n_x, d_x, h_x, w_x = X.size()

        h_out = (h_x - h_filter + 2 * padding) / stride + 1
        w_out = (w_x - w_filter + 2 * padding) / stride + 1

        h_out, w_out = int(h_out), int(w_out)
        X_col = torch.nn.functional.unfold(X.view(
            1, -1, h_x, w_x), h_filter, dilation=1, padding=padding, stride=stride).view(n_x, -1, h_out*w_out)
        X_col = X_col.permute(1, 2, 0).contiguous().view(X_col.size(1), -1)
        W_col = W.view(n_filters, -1)

        out = -torch.cdist(W_col, X_col.transpose(0, 1), 1)

        out = out.view(n_filters, h_out, w_out, n_x)
        out = out.permute(3, 0, 1, 2).contiguous()

        return out

    def optical_adder2d_function(self, X, W, stride=1, padding=0):
        n_filters, d_filter, h_filter, w_filter = W.size()
        n_x, d_x, h_x, w_x = X.size()

        h_out = (h_x - h_filter + 2 * padding) / stride + 1
        w_out = (w_x - w_filter + 2 * padding) / stride + 1

        h_out, w_out = int(h_out), int(w_out)
        X_col = torch.nn.functional.unfold(X.view(
            1, -1, h_x, w_x), h_filter, dilation=1, padding=padding, stride=stride).view(n_x, -1, h_out*w_out)
        X_col = X_col.permute(1, 2, 0).contiguous().view(X_col.size(1), -1)
        W_col = W.view(n_filters, -1) # [out_c, in_c*kernel_size*kernel_size]

        W_col_sq = (W_col * W_col).sum(dim=1, keepdim=True) # [outc, 1]
        X_col_sq = (X_col * X_col).sum(dim=0, keepdim=True) # [1, ...]
        W_by_X_by_sin_phi = -2*(W_col * self.phases.sin().view(1, -1).data).matmul(X_col) # [outc, inc*kernel_size*kernel_size] * [1, inc*kernel_size*kernel_size] @ [inc*ks*ks, ...] = [outc, ...]
        out = -(W_col_sq + X_col_sq + W_by_X_by_sin_phi)

        out = out.view(n_filters, h_out, w_out, n_x)
        out = out.permute(3, 0, 1, 2).contiguous()
        if(self.calibration):
            # [bs, outc, h, w]
            out = self.apply_calibration(out)

        return out

    def optical_conv2d_function(self, X, W, stride=1, padding=0):
        if(self.assigned == False):
            ## half kernels are negative
            W = torch.cat([-W[:, 0:self.input_channel//2,...], W[:, self.input_channel//2:,...]], dim=1)
            return F.conv2d(X, W, stride=stride, padding=padding)
        n_x = X.size(0)
        n_filters, d_filter, h_filter, w_filter = W.size()
        W_col, X_col, h_out, w_out = im2col_2d(W, X, stride, padding)
        # matmul with phase consideration
        # real = W_col.matmul(X_col)
        if(self.calibration):
            out = self.tiling_matmul_with_oconv_calibration(W_col, X_col, self.phases.sin(), self.disks)
        else:
            out = self.tiling_matmul(W_col, X_col, self.phases.sin(), self.disks)
        # print(1, F.mse_loss(real, out).data.item())
        # print("real:", real[0,100])
        # print("noisy:", out[0,100])
        # [outc, h*w*bs]

        # print("calib:", out[0,100])
        # print(2, F.mse_loss(real, out).data.item())
        out = out.view(n_filters, h_out, w_out, n_x)
        out = out.permute(3, 0, 1, 2).contiguous()

        return out

    def set_crosstalk_coupling_matrix(self, coupling_factor, drop_perc):
        assert 0 <= coupling_factor <= 1, logging.error(f"Coupling factor must in [0,1], but got {coupling_factor}")
        self.crosstalk_factor = 1 + max(3, (self.miniblock * (1 - drop_perc) - 1)) * coupling_factor
        # if(crosstalk_noise_std < 1e-5):
        #     self.crosstalk_coupling_matrix = torch.tensor([1] + [self.crosstalk_factor]*(self.miniblock-1), dtype=torch.float32, device=self.device)
        # else:
        #     self.crosstalk_coupling_matrix = torch.zeros(self.miniblock, dtype=torch.float32, device=self.device).normal_(coupling_factor, crosstalk_noise_std).clamp_(coupling_factor-3*crosstalk_noise_std,coupling_factor+3*crosstalk_noise_std)
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

    def inject_thermal_crosstalk(self, W, coupling_matrix_f):
        ### W: [p, q, k]
        ###coupling_matrix_f in the Fourier domain: [k//2]
        ### return weight in the Fourier domain
        W = torch.rfft(W, signal_ndim=1, normalized=False, onesided=True)
        W = complex_mult(W, coupling_matrix_f.unsqueeze(0).unsqueeze(0))
        return W

    def propagate_molg(self, W, X):
        ### W: [p, q, k]
        ### X: [ks*ks*inc, h_out*w_out*bs]
        X = X.t().contiguous() # [ks*ks*inc, h_out*w_out*bs] -> [h_out*w_out*bs, ks*ks*inc]
        X = X.view(X.size(0), self.grid_dim_x, self.miniblock) # [h_out*w_out*bs, ks*ks*inc] -> [h_out*w_out*bs, q, k]
        X = torch.rfft(X, signal_ndim=1, normalized=True, onesided=True)
        ### normalize weight
        # W = W / W.norm(p=2,dim=-1, keepdim=True)
        if(self.enable_thermal_crosstalk and self.crosstalk_factor > 1):
            W = W * self.crosstalk_factor
        W = torch.rfft(W, signal_ndim=1, normalized=False, onesided=True)
        X = complex_mult(W.unsqueeze(0), X.unsqueeze(1))
        X = torch.irfft(X, signal_ndim=1, normalized=True, onesided=True, signal_sizes=[self.miniblock]) # [h_out*w_out*bs, p, q, k]
        # X = X * 1.6
        if(self.enable_phase_noise and self.phase_noise_std > 1e-5):
            X = X + torch.zeros_like(X).normal_(0, self.phase_noise_std)
        phase = X
        # X_n = polynomial(X[:,:,:X.size(2)//2,:], self.molg_lambda_to_mag_curve_coeff_2_n)
        # X_p = polynomial(X[:,:,X.size(2)//2:,:], self.molg_lambda_to_mag_curve_coeff_5)
        # X = torch.cat([X_n, X_p], dim=2)
        # X = polynomial(X, self.molg_lambda_to_mag_curve_coeff_2_n)
        X = polynomial(X, self.molg_lambda_to_mag_curve_coeff_half).clamp(0,1)
        scale = self.scale# / self.scale.norm(p=2, dim=2, keepdim=True)
        if(self.grid_dim_x % 2 == 0): #even blocks
            X = X * torch.cat([scale, scale], dim=2)
        else:# odd blocks
            if(self.grid_dim_x > 1):
                X = X * torch.cat([scale, scale, self.scale_pad], dim=2)
            else:
                X = X * self.scale_pad
        # X = X * self.scale.abs()
        # X = F.tanh(X)
        # print_stat(phase)
        # X = gaussian(X, self.molg_lambda_to_mag_curve_coeff_4)

        if(self.add_drop):
            X = 2 * X - 1
        ### X [h_out*w_out*bs, p, q, k]
        if(self.balance_weight):
            neg_X = -X[:,:,:X.size(2)//2,:].sum(dim=2)
            pos_X = X[:,:,X.size(2)//2:,:].sum(dim=2)
            # print_stat(neg_X)
            # print_stat(pos_X)
            X = neg_X + pos_X
            # X = X * self.sign_quantizer(self.sign).data
            # X = X.sum(dim=2)
        else:
            if(self.add_drop):
                X = X.sum(dim=2) # [outc, h_out*w_out*bs]
            else:
                X = X.sum(dim=2) - X.size(2) // 2
        X = X.view(X.size(0), -1).t().contiguous()
        if(self.output_channel_pad > self.output_channel):
            X = X[:self.output_channel, :]

        return X, phase

    def molg_conv2d(self, X, W, stride=1, padding=0):
        ### W : [p, q, k]
        n_x = X.size(0)

        _, X_col, h_out, w_out = im2col_2d(None, X, stride, padding, w_size=(self.output_channel, self.input_channel, self.kernel_size, self.kernel_size))
        ## zero-padding X_col
        if(self.weight_input_channel_pad > self.weight_input_channel):
            if(self.x_zero_pad is None or self.x_zero_pad.size(1) != X_col.size(1)):
                self.x_zero_pad = torch.zeros(self.weight_input_channel_pad-self.weight_input_channel, X_col.size(1), dtype=torch.float32, device=self.device)

            X_col = torch.cat([X_col, self.x_zero_pad], dim=0)
        # matmul with phase consideration
        # real = W_col.matmul(X_col)
        out, phase = self.propagate_molg(W, X_col) # [outc, w_out]

        out = out.view(self.output_channel, h_out, w_out, n_x)
        out = out.permute(3, 0, 1, 2).contiguous()

        return out, phase

    def get_output_dim(self, img_height, img_width):
        h_out = (img_height - self.kernel_size + 2 * self.padding) / self.stride + 1
        w_out = (img_width - self.kernel_size + 2 * self.padding) / self.stride + 1
        return (int(h_out), int(w_out))

    def forward(self, x):

        x = self.input_quantizer(x)
        ### weight quantization, within range of [0,1]
        weight = self.weight_quantizer(self.weight)

        ### squared x to transform voltage to power, whichis proportional to phase shift
        output, phase = self.molg_conv2d(x*x, weight, self.stride, self.padding)

        if self.bias:
            output += self.b.unsqueeze(0).unsqueeze(2).unsqueeze(3)

        return output, phase

class AllPassMORRCirculantConv2d(nn.Module):
    '''
    description: All-pass MORR Conv2d layer, assumes (1) block-circulant matrix (2) differential rails (3) learnable balancing factors. Used for SqueezeLight TCAD extension.
    '''
    def __init__(self,
        in_channel,
        out_channel,
        kernel_size,
        bias=False,
        miniblock=4,
        stride=1,
        padding=0,
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
        super(AllPassMORRCirculantConv2d, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.mode = mode
        self.kernel_size = kernel_size
        self.miniblock = miniblock
        self.stride = stride
        self.padding = padding
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
        self.morr_gain = (100/(self.in_channel*self.kernel_size**2//self.miniblock))**0.5 ## TIA gain, calculated such that output variance is around 1
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
        self.out_channel_pad = int(np.ceil(self.out_channel / self.miniblock).item() * self.miniblock)
        self.weight_in_channel = self.kernel_size * self.kernel_size * self.in_channel
        self.weight_in_channel_pad = int (np.ceil(self.weight_in_channel / self.miniblock).item()) * self.miniblock
        self.grid_dim_y = self.out_channel_pad // self.miniblock
        self.grid_dim_x = self.weight_in_channel_pad // self.miniblock

        if(mode in {'weight'}):
            self.weight = Parameter(torch.ones(self.grid_dim_y, self.grid_dim_x, self.miniblock, device=self.device, dtype=torch.float))
            ### learnable balancing factor (morr_output_scale)
            ### (1) shared between different rows
            ### (2) shared between positive and negative rails
            ### (3) extra 1 padding if each row has odd number of miniblocks
            ### (4) different scaling factors are allowed for each cirulant block [Gu+, DATE'21] (too complicated)
            ### (4.1) In the TCAD version, we use a single scaling factor for each block
            # self.morr_output_scale = Parameter(torch.randn(1, 1, max(1,self.grid_dim_x//2) + 1, self.miniblock, device=self.device)) # deprecated
            self.morr_output_scale = Parameter(torch.zeros(max(1,self.grid_dim_x//2) + 1, device=self.device))
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
            morr_uniform_(self.weight, MORRConfig=self.MORRConfig, n_op=self.miniblock, biased=self.w_bit >= 16, gain=2 if self.in_bit < 16 else 1)
            ### output distribution aware initialization to output scaling factor
            # init.normal_(self.morr_output_scale.data)
            t1 = mrr_roundtrip_phase_to_tr_fused(torch.tensor([0]).float(), a=self.mrr_a, r=self.mrr_r, intensity=True)
            t2 = mrr_roundtrip_phase_to_tr_fused(torch.tensor([self.morr_fwhm*2.4]).float(), a=self.mrr_a, r=self.mrr_r, intensity=True)
            g = ((t2 - t1) / (2.4*self.morr_fwhm)).item() ## 0~2.4 FWHM slope as a linear approximation
            # print(self.morr_fwhm, g)

            self.sigma_out_scale = 4/(3*self.grid_dim_x**0.5*g*self.morr_fwhm)
            self.sigma_out_scale_quant_gain = None
            init.normal_(self.morr_output_scale, 0, self.sigma_out_scale)
            if(self.morr_input_bias is not None):
                ### after sigmoid, it corresponds to 0 bias
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

        self.crosstalk_factor = 1 + max(3, (self.miniblock * (1 - drop_perc) - 1)) * coupling_factor

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
        ### weights: [p, q, k]
        ### X: [ks*ks*inc, h_out*w_out*bs]
        # x = x.t().contiguous() # [ks*ks*inc, h_out*w_out*bs] -> [h_out*w_out*bs, ks*ks*inc]
        # x = x.view(x.size(0), self.grid_dim_x, self.miniblock) # [h_out*w_out*bs, ks*ks*inc] -> [h_out*w_out*bs, q, k]
        x = x.view(self.grid_dim_x, self.miniblock, x.size(-1)) # [q, k, h_out*w_out*bs]
        # x = torch.rfft(x, signal_ndim=1, normalized=True, onesided=True) # old API for pytorch 1.6
        x = torch.fft.rfft(x, n=self.miniblock, dim=1, norm="ortho") # new API for pytorch 1.7
        ### normalize weight
        # W = W / W.norm(p=2,dim=-1, keepdim=True)
        if(self.enable_thermal_crosstalk and self.crosstalk_factor > 1):
            weight = weight * self.crosstalk_factor
        # weight = torch.rfft(weight, signal_ndim=1, normalized=False, onesided=True) # old API for pytorch 1.6
        weight = torch.fft.rfft(weight, n=self.miniblock, dim=-1, norm="backward")  # new API for pytorch 1.7
        # x = complex_mult(weight.unsqueeze(0), x.unsqueeze(1))
        x = weight.unsqueeze(-1) * x.unsqueeze(0) # [p, q, k, 1]x[1, q, k, :]=[p, q, k, h_out*w_out*bs]
        # x = torch.irfft(x, signal_ndim=1, normalized=True, onesided=True, signal_sizes=[self.miniblock]) # [h_out*w_out*bs, p, q, k]
        x = torch.fft.irfft(x, n=self.miniblock, dim=2, norm="ortho")# [p, q, k, h_out*w_out*bs]
        # X = X * 1.6
        if(self.enable_phase_noise and self.phase_noise_std > 1e-5):
            x = x + torch.zeros_like(x).normal_(0, self.phase_noise_std)

        ### Deprecated: Use polynomial for fast computation and gradient-based sensitivity-regularization [Gu+, DATE'21] (not suitable in TCAD)
        # x = polynomial(x, self.molg_lambda_to_mag_curve_coeff_half).clamp(0,1)

        ### input scaling [TCAD'21], must have valid ranges. too small will have dead neuron and not enough nonlinearity; too large will have larger power, cross-channel crosstalk. [0.1 - 1.1] will be suitable
        if(self.trainable_morr_scale): ## not quite useful, can be learned in weight distribution
            # x = x * (torch.sigmoid(self.morr_input_scale.unsqueeze(0).unsqueeze(-1)) + 0.1)
            x = x * (torch.sigmoid(self.morr_input_scale.unsqueeze(-1).unsqueeze(-1)) + 0.1)
        ### input biasing [TCAD'21], must have valid ranges. too large will have power issue and cross-channel crosstalk. [-2FWHM ~ 0]
        if(self.trainable_morr_bias):
            # x = x - 2 * self.morr_fwhm * torch.sigmoid(self.morr_input_bias.unsqueeze(0).unsqueeze(-1))
            x = x - 2 * self.morr_fwhm * torch.sigmoid(self.morr_input_bias.unsqueeze(-1).unsqueeze(-1))
        ### Use theoretical transmission function for trainable MORR nonlinearity [TCAD'21]
        ### x is the phase detuning, x=0 means on-resonance
        ### phase: [p, q, k, h_out*w_out*bs]

        x = mrr_roundtrip_phase_to_tr_fused(rt_phi=x, a=self.mrr_a, r=self.mrr_r, intensity=True)

        ### output scaling
        if(self.w_bit < 16):
            morr_output_scale = self.morr_output_scale_quantizer(self.morr_output_scale)
            if(self.sigma_out_scale_quant_gain is None):
                self.sigma_out_scale_quant_gain = self.sigma_out_scale / morr_output_scale.data.std().item()
            morr_output_scale = morr_output_scale.mul(self.sigma_out_scale_quant_gain)### gain factor from Tanh used in quantization
        else:
            morr_output_scale = self.morr_output_scale
        # print("scale:", end=", ")
        # print_stat(morr_output_scale)
        # morr_output_scale = morr_output_scale * self.morr_gain
        scale = morr_output_scale[:-1]
        scale_pad = morr_output_scale[-1:]
        if(self.grid_dim_x % 2 == 0):
            #even blocks
            scale = torch.cat([scale, scale], dim=0)
        else:
            # odd blocks
            if(self.grid_dim_x > 1):
                scale = torch.cat([morr_output_scale, scale], dim=0)
            else:
                scale = scale_pad
        scale = scale.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

        x = x * scale

        return x

    def propagate_photodetection(self, x):
        '''
        @description: photodetection, from light to analog voltages
        @param x {torch.Tensor} light signals before photodetection, should be complex-valued
        @param oe_factor {scalar} linear scaling factor between light energy to analog voltage
        @return: y {torch.Tensor} real-valued voltage outputs
        '''
        ### x: [h_out*w_out*bs, p, q, k] -> [outc, h_out*w_out*bs] deprecated
        ### x: [p, q, k, h_out*w_out*bs] -> [outc, h_out*w_out*bs]
        ### by default we use differential rails with balanced photo-detection. first half is negative, the second half is positive
        ### add-drop MORR is not considered. we only use all-pass MORR
        # x = -x[:,:,:x.size(2)//2,:].sum(dim=2) + x[:,:,x.size(2)//2:,:].sum(dim=2)

        # x = x.view(x.size(0), -1).t().contiguous()

        x = x[:,:int(np.ceil(x.size(1)/2)),...].sum(dim=1) - x[:,int(np.ceil(x.size(1)/2)):,...].sum(dim=1)
        x = x.view(-1, x.size(-1))
        if(self.out_channel_pad > self.out_channel):
            x = x[:self.out_channel, :]

        return x

    def propagate_morr_efficient(self, weight, x):
        '''
        @description: propagate through the analytically calculated transfer matrix of molg. We implement circulant matrix multiplication using circulant matrix mul
        @param weight {torch.Tensor} two phase shifters in the MZI-based attenuators
        @param x {torch.Tensor} complex-valued input
        @return: y {torch.Tensor} output of attenuators
        '''
        ### weights: [p, q, k]
        ### x: [ks*ks*inc, h_out*w_out*bs]

        x = x.t() #[h_out*w_out*bs, ks*ks*inc]
        x = x.view(x.size(0), self.grid_dim_x, self.miniblock) # [h_out*w_out*bs, q, k]

        if(self.enable_thermal_crosstalk and self.crosstalk_factor > 1):
            weight = weight * self.crosstalk_factor
        weight = toeplitz(weight).unsqueeze(0) # [1, p, q, k, k]
        x = x.unsqueeze(1).unsqueeze(-1)       # [h*w*bs, 1, q, k, 1]
        x = weight.matmul(x).squeeze(-1)       # [h*w*bs, p, q, k]

        if(self.enable_phase_noise and self.phase_noise_std > 1e-5):
            x = x + torch.zeros_like(x).normal_(0, self.phase_noise_std) # [h*w*bs, p, q, k]

        ### input scaling [TCAD'21], must have valid ranges. too small will have dead neuron and not enough nonlinearity; too large will have larger power, cross-channel crosstalk. [0.1 - 1.1] will be suitable
        if(self.trainable_morr_scale): ## not quite useful, can be learned in weight distribution
            x = x * (torch.sigmoid(self.morr_input_scale.unsqueeze(0).unsqueeze(-1)) + 0.1) # [h*w*bs, p, q, k]
        ### input biasing [TCAD'21], must have valid ranges. too large will have power issue and cross-channel crosstalk. [-2FWHM ~ 0]
        if(self.trainable_morr_bias):
            x = x - 2 * self.morr_fwhm * torch.sigmoid(self.morr_input_bias.unsqueeze(0).unsqueeze(-1)) # [h*w*bs, p, q, k]
        ### Use theoretical transmission function for trainable MORR nonlinearity [TCAD'21]
        ### x is the phase detuning, x=0 means on-resonance
        ### phase: [h_out*w_out*bs, p, q, k]
        x = self.mrr_roundtrip_phase_to_tr(x)

        ### output scaling
        if(self.w_bit < 16):
            morr_output_scale = self.morr_output_scale_quantizer(self.morr_output_scale)
            if(self.sigma_out_scale_quant_gain is None):
                self.sigma_out_scale_quant_gain = self.sigma_out_scale / morr_output_scale.data.std().item()
            morr_output_scale = morr_output_scale.mul(self.sigma_out_scale_quant_gain)### gain factor from Tanh used in quantization
        else:
            morr_output_scale = self.morr_output_scale
        # morr_output_scale = morr_output_scale * self.morr_gain

        scale = morr_output_scale[:-1]
        scale_pad = morr_output_scale[-1:]
        if(self.grid_dim_x % 2 == 0):
            #even blocks
            scale = torch.cat([scale, -scale], dim=0)
        else:
            # odd blocks
            if(self.grid_dim_x > 1):
                scale = torch.cat([morr_output_scale, -scale], dim=0)
            else:
                scale = scale_pad
        scale = scale.unsqueeze(0).unsqueeze(0).unsqueeze(0) # [1, 1, 1, q]

        x = scale.matmul(x) #[1,1,1,q]x[h_out*w_out*bs, p, q, k]=[h_out*w_out*bs, p, 1, k]
        x = x.view(x.size(0), -1).t() # [p*k, h_out*w_out*bs]
        if(self.out_channel_pad > self.out_channel):
            x = x[:self.out_channel, :] # [outc, h_out*w_out*bs]
        return x

    def block_accumulation(self, x, complex=False):
        '''
        @description: accumulation of block-circulant matrix multiplication. Performed after photodetection, thus input will be real-valued.
        @param x {torch.Tensor} real-valued signals after photo-detection [batch_size, grid_dim_y, grid_dim_x, miniblock]
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

    def morr_conv2d(self, X, W, stride=1, padding=0):
        ### W : [p, q, k]
        n_x = X.size(0)

        _, X_col, h_out, w_out = im2col_2d(None, X, stride, padding, w_size=(self.out_channel, self.in_channel, self.kernel_size, self.kernel_size))
        ## zero-padding X_col
        if(self.weight_in_channel_pad > self.weight_in_channel):
            if(self.x_zero_pad is None or self.x_zero_pad.size(1) != X_col.size(1)):
                self.x_zero_pad = torch.zeros(self.weight_in_channel_pad-self.weight_in_channel, X_col.size(1), dtype=torch.float32, device=self.device)

            X_col = torch.cat([X_col, self.x_zero_pad], dim=0)
        # matmul
        # print(W.size(), X_col.size())
        out = self.propagate_morr(W, X_col) # [outc, w_out]
        out = self.propagate_photodetection(out)
        out = out.view(self.out_channel, h_out, w_out, n_x)
        out = out.permute(3, 0, 1, 2).contiguous()

        return out

    def morr_conv2d_efficient(self, X, W, stride=1, padding=0):
        ### W : [p, q, k]
        n_x = X.size(0)

        _, X_col, h_out, w_out = im2col_2d(None, X, stride, padding, w_size=(self.out_channel, self.in_channel, self.kernel_size, self.kernel_size))
        ## zero-padding X_col
        if(self.weight_in_channel_pad > self.weight_in_channel):
            if(self.x_zero_pad is None or self.x_zero_pad.size(1) != X_col.size(1)):
                self.x_zero_pad = torch.zeros(self.weight_in_channel_pad-self.weight_in_channel, X_col.size(1), dtype=torch.float32, device=self.device)

            X_col = torch.cat([X_col, self.x_zero_pad], dim=0)
        # matmul
        # print(W.size(), X_col.size())
        out = self.propagate_morr_efficient(W, X_col) # [outc, w_out]
        out = out.view(self.out_channel, h_out, w_out, n_x)
        out = out.permute(3, 0, 1, 2).contiguous()

        return out

    def forward_slow(self, x):
        if(self.in_bit < 16):
            x = self.input_quantizer(x)
        # if(not self.fast_forward_flag or self.weight is None):
        #     weight = self.build_weight()
        # else:
        #     weight = self.weight #.view(self.out_channel, -1)[:, :self.in_channel]
        weight = self.build_weight()
        x = self.input_modulator(x)
        x = self.morr_conv2d(x, weight, stride=self.stride, padding=self.padding)

        if(self.bias is not None):
            x = x + self.b.unsqueeze(0).unsqueeze(2).unsqueeze(3)

        return x

    def forward(self, x):
        ### ~7x faster than forward slow
        if(self.in_bit < 16):
            x = self.input_quantizer(x)
        # if(not self.fast_forward_flag or self.weight is None):
        #     weight = self.build_weight()
        # else:
        #     weight = self.weight #.view(self.out_channel, -1)[:, :self.in_channel]
        weight = self.build_weight()
        x = self.input_modulator(x)
        x = self.morr_conv2d_efficient(x, weight, stride=self.stride, padding=self.padding)

        if(self.bias is not None):
            x = x + self.b.unsqueeze(0).unsqueeze(2).unsqueeze(3)

        return x




