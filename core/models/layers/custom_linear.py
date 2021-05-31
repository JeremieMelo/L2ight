
import logging
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from pyutils.compute import gen_boolean_mask, gen_gaussian_noise, percentile
from pyutils.matrix_parametrization import RealUnitaryDecomposerBatch
from pyutils.mzi_op import (checkerboard_to_vector, phase_to_voltage,
                            vector_to_checkerboard, voltage_quantize_fn,
                            voltage_to_phase)
from pyutils.quantize import input_quantize_fn
from pyutils.torch_train import set_torch_deterministic
from torch import Tensor
from torch.nn import Parameter, init
from torch.types import Device

from .utils import LearningProfiler, PhaseQuantizer, LinearFeatureSampler, FeedbackSampler, SingularValueGradientSampler

__all__ = ["MZIBlockLinear"]


def dfa_linear(x, weight, feedback_matrix, grad_final):
    class DFALinearFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, weight, feedback_matrix, grad_final):
            # grad_final is a buffer to create a dummy path in order to propagate the final error back
            with torch.no_grad():
                out = F.linear(x, weight)
            ctx.save_for_backward(x, feedback_matrix)
            return out, grad_final

        @staticmethod
        def backward(ctx, grad_output, grad_final):
            ## grad_output [bs, outc]
            x, feedback_matrix = ctx.saved_tensors
            # [bs, outL] x [outL, inc] = [bs, inc]
            grad_input = grad_final.matmul(feedback_matrix)
            # [outc, bs] x [bs, inc] = [outc, inc]
            grad_weight = grad_output.t().matmul(x)
            return grad_input, grad_weight, None, grad_final

    return DFALinearFunction.apply(x, weight, feedback_matrix, grad_final)


class DFALinear(torch.nn.Linear):
    def __init__(self, in_channel, out_channel, n_class, bias=False, dfa=True, device=torch.device("cuda")) -> None:
        super().__init__(in_channel, out_channel, bias)
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.n_class = n_class
        self.dfa = dfa
        self.device = device

    def reset_feedback_matrix(self, sparsity=0, base_matrix=None):
        if(base_matrix is not None):
            self.feedback_matrix = base_matrix[:, :self.in_channel]
        else:
            self.feedback_matrix = torch.nn.init.kaiming_normal_(
                torch.empty(self.n_class, self.in_channel, device=self.device))
            if(sparsity > 0):
                feedback_matrix_abs = self.feedback_matrix.abs()
                mask = feedback_matrix_abs < torch.quantile(
                    feedback_matrix_abs, sparsity)
                self.feedback_matrix.masked_fill_(mask, 0)

    def forward(self, x, grad_final):

        x, grad_final = dfa_linear(
            x, self.weight, self.feedback_matrix if self.dfa else self.weight.data, grad_final)
        return x, grad_final


def dfa_fuse_logits(x, grad_final):
    class DFAFuseLogitsFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, grad_final):
            # grad_final is a buffer to create a dummy path in order to propagate the final error back
            return x

        @staticmethod
        def backward(ctx, grad_output):
            return grad_output, grad_output
    return DFAFuseLogitsFunction.apply(x, grad_final)


def dfa_relu(x, grad_final):
    class DFAReLUFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, grad_final):
            # grad_final is a buffer to create a dummy path in order to propagate the final error back
            with torch.no_grad():
                x = torch.relu(x)
            ctx.save_for_backward(x < 0)
            return x, grad_final

        @staticmethod
        def backward(ctx, grad_output, grad_final):
            mask, = ctx.saved_tensors
            grad_input = grad_output.clone().masked_fill_(mask, 0)
            return grad_input, grad_final
    return DFAReLUFunction.apply(x, grad_final)


def sparse_bp_linear(x: Tensor, u: Tensor, s: Tensor, v: Tensor, I_U: Optional[Tensor] = None, I_V: Optional[Tensor] = None, feedback_sampler: FeedbackSampler = None, feature_sampler: LinearFeatureSampler = None, rank_sampler: SingularValueGradientSampler = None, profiler: LinearFeatureSampler = None) -> Tensor:
    """Blocking linear function. SVD-based matrix multiplication. Support block-level structured sparsity in feedback matrix and input features [Randomized Automatic Differentiation, ICLR'21]. Only gradient w.r.t. S matrix is calculated. Support low-rank backpropagation given the low-rank property of singular values.

    Args:
        x (torch.Tensor): padded input, batched and blocked vector [bs, q*k]
        u (torch.Tensor): U matrices. [p, q, k, k]
        s (torch.Tensor): Sigma matrices. [p, q, k]
        v (torch.Tensor): V* matrices. [p, q, k, k]
        I_U (torch.Tensor): Pseudo-I for U. [p, q, k, k]
        I_V (torch.Tensor): Pseudo-I for V. [p, q, k, k]
        bp_feedback_mask (torch.Tensor, optional): p x q Boolean Mask to prune feedback matrix. Designed to speedup the feedback accumulation. False represents pruned. Can be fixed mask. Defaults to None.
        bp_input_mask (torch.Tensor, optional): bs x q Boolean Mask to prune input features. Each example must not share masks in fully-connected layers, otherwise the weights will not be updated in this interation. Cannot be fixed mask, must uniformly sample from the inputs to gaurantee unbiased estimation. Designed to speedup the gradient calculation and save memory footprint. False represents pruned. Defaults to None.
        bp_rank (int, optional): Only calculate the gradients to the first bp_rank singular values. Will not dynamically calculate largest bp_rank values. Ranges [1, k]. Defaults to 8.

    Returns:
        [torch.Tensor]: [bs, p*k] output tensor
    """
    class SparseBPLinearFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x: Tensor, u: Tensor, s: Tensor, v: Tensor) -> Tensor:
            p, q, k = u.size(0), u.size(1), u.size(2)
            outc = p*k
            inc = q*k
            with torch.no_grad():
                weight = u.matmul(s.unsqueeze(-1).mul(v))
                if(feedback_sampler is not None and feedback_sampler.forward_sparsity > 1e-9):
                    # if this happen, this must be reused in backward, otherwise it won't work
                    feedback_sampler.sample_(weight, forward=True)

                out = F.linear(x, weight.permute([0, 2, 1, 3]).contiguous().view(
                    outc, inc), bias=None)
                if(profiler is not None):
                    profiler.update_forward(x, weight, out, feedback_sampler)
                if(feature_sampler is not None):
                    x, _ = feature_sampler.sample(x)
                ctx.save_for_backward(x, u, v, weight)

            return out

        @staticmethod
        def backward(ctx, grad_output: Tensor) -> Tuple[Tensor, ...]:
            ## grad_output [bs, outc] = [bs, p*k]
            grad_input = grad_sigma = None
            x, u, v, weight = ctx.saved_tensors
            p, q, k = u.size(0), u.size(1), u.size(2)
            outc = p*k
            inc = q*k
            # sparse matrix as feedback matrix [p, q, k, k] x [p, q, 1, 1] = [p, q, k, k]
            if(ctx.needs_input_grad[0]):
                if(feedback_sampler is not None and feedback_sampler.forward_sparsity < 1e-9):
                    # only sample when there is no forward sampling. If there is forward sampling, we directly use the saved weight.
                    feedback_sampler.sample_(weight, forward=False)
                # [bs, outc] x [outc, inc] = [bs, inc]
                grad_input = grad_output.matmul(weight.permute(
                    [0, 2, 1, 3]).contiguous().view(outc, inc))

            # we calculate the gradient to Sigma in an memory-efficient way, we first calculate dW, then calculate dS. Not the same as hardware implementation, but equavalent
            # [outc, bs] x [bs, inc] = [outc, inc]
            if(ctx.needs_input_grad[2]):
                if(feature_sampler is not None):
                    x = feature_sampler.reconstruct(x)
                grad_weight = grad_output.t().matmul(x)
                grad_weight = grad_weight.view(p, k, q, k).permute(
                    [0, 2, 1, 3])  # [p, q, k, k]

                # x   -> (V*_noisy -> I -> I_noisy)  -> yv
                # dys <- (I_noisy* <- I <- U^*_noisy) <- dy
                # u^T x dw = [p, q, k, k] or [p, q, bp_rank, k]
                if(rank_sampler is not None):
                    grad_sigma = rank_sampler.sample(
                        u, s, v, grad_weight, I_U, I_V)

            if(profiler is not None):
                profiler.update_backward(
                    x, weight, grad_output, ctx.needs_input_grad[0], ctx.needs_input_grad[2], feature_sampler, feedback_sampler, rank_sampler)
            return grad_input, None, grad_sigma, None
    return SparseBPLinearFunction.apply(x, u, s, v)


class SparseBPLinear(torch.nn.Module):
    def __init__(self,
                 in_channel,
                 out_channel,
                 miniblock=8,
                 bias=False,
                 bp_sparsity=0,
                 bp_rank=8,
                 device=torch.device("cuda")
                 ) -> None:
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.miniblock = miniblock
        self.grid_dim_x = int(np.ceil(self.in_channel / miniblock))
        self.grid_dim_y = int(np.ceil(self.out_channel / miniblock))
        self.in_channel_pad = self.grid_dim_x * miniblock
        self.out_channel_pad = self.grid_dim_y * miniblock

        self.bias = bias
        self.bp_sparsity = bp_sparsity
        self.bp_rank = bp_rank
        self.device = device
        self.build_parameters()
        self.bp_mask = None
        self.x_zero_pad = None

    def build_parameters(self):
        self.U = torch.nn.Parameter(torch.empty(self.grid_dim_y, self.grid_dim_x,
                                                self.miniblock, self.miniblock, device=self.device), requires_grad=False)
        self.S = torch.nn.Parameter(torch.empty(
            self.grid_dim_y, self.grid_dim_x, self.miniblock, device=self.device), requires_grad=True)
        self.V = torch.nn.Parameter(torch.empty(self.grid_dim_y, self.grid_dim_x,
                                                self.miniblock, self.miniblock, device=self.device), requires_grad=False)

    def set_bp_rank(self, bp_rank):
        assert 1 <= bp_rank <= self.miniblock
        self.bp_rank = bp_rank

    def reset_parameters(self):
        torch.nn.init.kaiming_normal_(self.U)
        u, s, v = self.U.data.svd()
        v = v.transpose(-1, -2).contiguous()
        self.U.data.copy_(u)
        self.S.data.copy_(s)
        self.V.data.copy_(v)

    def gen_deterministic_gradient_mask(self, bp_sparsity=None):
        if(bp_sparsity is None):
            bp_sparsity = self.bp_sparsity
        assert 0 <= bp_sparsity < 1
        # we prefer uniform column-wise sparsity in W, i.e., row-wise sparsity in W^T
        self.bp_mask = torch.ones(
            self.grid_dim_y, self.grid_dim_x, device=self.device, dtype=torch.bool)
        # pruned blocks has small total singular value
        s_sum = self.S.data.abs().sum(dim=-1)
        s_thres = torch.quantile(s_sum, q=bp_sparsity, dim=0)
        self.bp_mask.masked_fill_(s_sum < s_thres.unsqueeze(0), 0)
        return self.bp_mask

    def forward(self, x):
        if(self.in_channel_pad > self.in_channel):
            if(self.x_zero_pad is None or self.x_zero_pad.size(0) != x.size(0)):
                self.x_zero_pad = torch.zeros(x.size(
                    0), self.in_channel_pad - self.in_channel, device=x.device, dtype=x.dtype)
            x = torch.cat([x, self.x_zero_pad], dim=1)

        x = x.view(-1, self.grid_dim_x, self.miniblock)
        x = sparse_bp_linear(x, self.U, self.S, self.V,
                             bp_mask=self.bp_mask, bp_rank=self.bp_rank)
        if(self.out_channel < self.out_channel_pad):
            x = x[..., :self.out_channel]
        return x


class MZIBlockLinear(torch.nn.Module):
    '''
    description: SVD-based Linear layer. Blocking matrix multiplication. Support on-chip learning SZO-SCD [AAAI'21]. Support on-chip learning via subspace mapping and sparse BP [on-going]
    '''

    def __init__(self,
                 in_channel: int,
                 out_channel: int,
                 miniblock: int = 8,
                 bias: bool = False,
                 mode: str = "weight",
                 v_max: float = 4.36,  # 0-pi for clements, # 6.166 is v_2pi, 0-2pi for reck
                 v_pi: float = 4.36,
                 w_bit: int = 16,
                 in_bit: int = 16,
                 photodetect: bool = False,
                 device: Device = torch.device("cuda")
                 ) -> None:
        super(MZIBlockLinear, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.miniblock = miniblock
        self.grid_dim_x = int(np.ceil(self.in_channel / miniblock))
        self.grid_dim_y = int(np.ceil(self.out_channel / miniblock))
        self.in_channel_pad = self.grid_dim_x * miniblock
        self.out_channel_pad = self.grid_dim_y * miniblock
        self.mode = mode
        assert mode in {"weight", "usv", "phase", "voltage"}, logging.error(
            f"Mode not supported. Expected one from (weight, usv, phase, voltage) but got {mode}.")
        self.v_max = v_max
        self.v_pi = v_pi
        self.gamma = np.pi / self.v_pi**2
        self.w_bit = w_bit
        self.in_bit = in_bit
        self.photodetect = photodetect
        self.device = device

        # build parameters
        self.build_parameters(mode)

        # backpropagation sparsity in feedback matrix
        self.bp_feedback_sampler = FeedbackSampler(0, 0, alg="topk", mode="linear")
        # backpropagation sparsity in input features
        self.bp_input_sparsity = 0
        self.bp_input_sampler = LinearFeatureSampler(0, self.miniblock)
        # backpropagation sparsity in the Sigma rank
        self.bp_rank = miniblock
        self.bp_rank_sampler = SingularValueGradientSampler(
            self.bp_rank, alg="topk")

        # unitary parametrization tool
        self.decomposer = RealUnitaryDecomposerBatch(alg="clements")

        self.decomposer.v2m = vector_to_checkerboard
        self.decomposer.m2v = checkerboard_to_vector
        # quantization tool
        # self.input_quantizer = PACT_Act(self.in_bit, device=self.device)
        self.input_quantizer = input_quantize_fn(
            self.in_bit, device=self.device)
        # self.unitary_quantizer = UnitaryQuantizer(
        #     self.w_bit, alg="clements", device=self.device)
        # self.weight_quantizer = weight_quantize_fn(self.w_bit, alg="dorefa_sym")
        # self.diag_quantizer = DiagonalQuantizer(self.w_bit, device=self.device)
        self.voltage_quantizer = voltage_quantize_fn(
            self.w_bit, self.v_pi, self.v_max)
        # self.phase_quantizer = PhaseQuantizer(self.w_bit, device=self.device)
        self.phase_U_quantizer = PhaseQuantizer(self.w_bit, self.v_pi, self.v_max, gamma_noise_std=0,
                                                crosstalk_factor=0, crosstalk_filter_size=5, random_state=0, mode="rectangle", device=self.device)
        self.phase_V_quantizer = PhaseQuantizer(self.w_bit, self.v_pi, self.v_max, gamma_noise_std=0,
                                                crosstalk_factor=0, crosstalk_filter_size=5, random_state=0, mode="rectangle", device=self.device)
        self.phase_S_quantizer = PhaseQuantizer(self.w_bit, self.v_pi, self.v_max, gamma_noise_std=0,
                                                crosstalk_factor=0, crosstalk_filter_size=5, random_state=0, mode="diagonal", device=self.device)

        # default set to slow forward
        self.disable_fast_forward()
        # default set no phase noise
        self.set_phase_variation(0)
        # default set no gamma noise
        self.set_gamma_noise(0)
        # default set no crosstalk
        self.set_crosstalk_factor(0)
        # default enable noisy identity
        self.set_noisy_identity(True)
        # default disable mixed training
        self.disable_mixedtraining()
        # setup learning profiler
        self.profiler = LearningProfiler(False)
        # zero pad for input
        self.x_zero_pad = None

        if(bias):
            self.bias = Parameter(torch.Tensor(out_channel).to(self.device))
        else:
            self.register_parameter('bias', None)

        # self.reset_parameters()
    def build_parameters(self, mode: str = "weight") -> None:
        # weight mode
        weight = torch.Tensor(
            self.grid_dim_y, self.grid_dim_x, self.miniblock, self.miniblock).to(self.device).float()
        # usv mode
        U = torch.Tensor(self.grid_dim_y, self.grid_dim_x,
                         self.miniblock, self.miniblock).to(self.device).float()
        S = torch.Tensor(self.grid_dim_y, self.grid_dim_x,
                         self.miniblock).to(self.device).float()
        V = torch.Tensor(self.grid_dim_y, self.grid_dim_x,
                         self.miniblock, self.miniblock).to(self.device).float()
        # phase mode
        delta_list_U = torch.Tensor(
            self.grid_dim_y, self.grid_dim_x, self.miniblock).to(self.device).float()
        phase_U = torch.Tensor(self.grid_dim_y, self.grid_dim_x,
                               self.miniblock*(self.miniblock-1)//2).to(self.device).float()
        phase_S = torch.Tensor(
            self.grid_dim_y, self.grid_dim_x, self.miniblock).to(self.device).float()
        delta_list_V = torch.Tensor(
            self.grid_dim_y, self.grid_dim_x, self.miniblock).to(self.device).float()
        phase_V = torch.Tensor(self.grid_dim_y, self.grid_dim_x,
                               self.miniblock*(self.miniblock-1)//2).to(self.device).float()
        # voltage mode
        voltage_U = torch.Tensor(
            self.grid_dim_y, self.grid_dim_x, self.miniblock*(self.miniblock-1)//2).to(self.device).float()
        voltage_S = torch.Tensor(
            self.grid_dim_y, self.grid_dim_x, self.miniblock).to(self.device).float()
        voltage_V = torch.Tensor(
            self.grid_dim_y, self.grid_dim_x, self.miniblock*(self.miniblock-1)//2).to(self.device).float()
        # TIA gain
        S_scale = torch.Tensor(
            self.grid_dim_y, self.grid_dim_x, 1).to(self.device).float()
        phase_bias_U = torch.zeros_like(phase_U)
        phase_bias_V = torch.zeros_like(phase_V)

        # Identity, can be saved/loaded to/from checkpoint
        I_U = torch.diag_embed(torch.ones(self.grid_dim_y, self.grid_dim_x,
                                          self.miniblock, device=self.device))
        I_V = torch.diag_embed(torch.ones(self.grid_dim_y, self.grid_dim_x,
                                          self.miniblock, device=self.device))

        if(mode == 'weight'):
            self.weight = Parameter(weight)
        elif(mode == "usv"):
            self.U = Parameter(U)
            self.S = Parameter(S)
            self.V = Parameter(V)
        elif(mode == "phase"):
            self.phase_U = Parameter(phase_U)
            self.phase_S = Parameter(phase_S)
            self.phase_V = Parameter(phase_V)
            self.S_scale = Parameter(S_scale)
        elif(mode == "voltage"):
            self.voltage_U = Parameter(voltage_U)
            self.voltage_S = Parameter(voltage_S)
            self.voltage_V = Parameter(voltage_V)
            self.S_scale = Parameter(S_scale)
        else:
            raise NotImplementedError

        for p_name, p in {"weight": weight, "U": U, "S": S, "V": V, "phase_U": phase_U, "phase_S": phase_S, "phase_V": phase_V, "S_scale": S_scale, "voltage_U": voltage_U, "voltage_S": voltage_S, "voltage_V": voltage_V, "I_U": I_U, "I_V": I_V, "delta_list_U": delta_list_U, "delta_list_V": delta_list_V, "phase_bias_U": phase_bias_U, "phase_bias_V": phase_bias_V}.items():
            if(not hasattr(self, p_name)):
                self.register_buffer(p_name, p)

    def reset_parameters(self) -> None:
        if(self.mode == "weight"):
            init.kaiming_normal_(self.weight.data)
        elif(self.mode == "usv"):
            W = init.kaiming_normal_(torch.empty(self.grid_dim_y, self.grid_dim_x,
                                                 self.miniblock, self.miniblock, dtype=self.U.dtype, device=self.device))
            U, S, V = torch.svd(W, some=False)
            V = V.transpose(-2, -1)
            self.U.data.copy_(U)
            self.V.data.copy_(V)
            self.S.data.copy_(S)
        elif(self.mode == "phase"):
            W = init.kaiming_normal_(torch.empty(self.grid_dim_y, self.grid_dim_x,
                                                 self.miniblock, self.miniblock, dtype=self.U.dtype, device=self.device))
            U, S, V = torch.svd(W, some=False)
            V = V.transpose(-2, -1)
            delta_list, phi_mat = self.decomposer.decompose(U)
            self.delta_list_U.data.copy_(delta_list)
            self.phase_U.data.copy_(self.decomposer.m2v(phi_mat))
            delta_list, phi_mat = self.decomposer.decompose(V)
            self.delta_list_V.data.copy_(delta_list)
            self.phase_V.data.copy_(self.decomposer.m2v(phi_mat))
            self.S_scale.data.copy_(S.abs().max(dim=-1, keepdim=True)[0])
            self.phase_S.data.copy_(S.div(self.S_scale.data).acos())
        elif(self.mode == "voltage"):
            # This is not the real MZI implementation. The voltage is a conceptual voltage. Real MZI will be much more complicated, and at least 3 phase shifters need to be considered.
            W = init.kaiming_normal_(torch.empty(self.grid_dim_y, self.grid_dim_x,
                                                 self.miniblock, self.miniblock, dtype=self.U.dtype, device=self.device))
            U, S, V = torch.svd(W, some=False)
            V = V.transpose(-2, -1)
            delta_list, phi_mat = self.decomposer.decompose(U)
            self.delta_list_U.data.copy_(delta_list)
            self.voltage_U.data.copy_(phase_to_voltage(
                self.decomposer.m2v(phi_mat), self.gamma))
            delta_list, phi_mat = self.decomposer.decompose(V)
            self.delta_list_V = delta_list
            self.voltage_V.data.copy_(phase_to_voltage(
                self.decomposer.m2v(phi_mat), self.gamma))
            self.S_scale.data.copy_(S.abs().max(dim=-1, keepdim=True)[0])
            self.voltage_S.data.copy_(phase_to_voltage(
                S.div(self.S_scale.data).acos(), self.gamma))
        else:
            raise NotImplementedError

        if self.bias is not None:
            init.uniform_(self.bias, 0, 0)

    def build_weight_from_usv(self, U: Tensor, S: Tensor, V: Tensor) -> Tensor:
        # differentiable feature is gauranteed
        weight = U.matmul(S.unsqueeze(-1) * V)
        self.weight.data.copy_(weight)
        return weight

    def build_weight_from_phase(self, delta_list_U: Tensor, phase_U: Tensor, delta_list_V: Tensor, phase_V: Tensor, phase_S: Tensor, S_scale: Tensor, update_list: Dict = {"phase_U", "phase_S", "phase_V"}) -> Tensor:
        ### not differentiable
        # reconstruct is time-consuming, a fast method is to only reconstruct based on updated phases

        return self.build_weight_from_usv(*self.build_usv_from_phase(delta_list_U, phase_U, delta_list_V, phase_V, phase_S, S_scale, update_list=update_list))

    def build_weight_from_voltage(self, delta_list_U: Tensor, voltage_U: Tensor, delta_list_V: Tensor, voltage_V: Tensor, voltage_S: Tensor, gamma_U: Union[Tensor, float], gamma_V: Union[Tensor, float], gamma_S: Union[Tensor, float], S_scale: Tensor) -> None:
        # This is not the real MZI implementation. The voltage is a conceptual voltage. Real MZI will be much more complicated, and at least 3 phase shifters need to be considered.
        self.phase_U = voltage_to_phase(voltage_U, gamma_U)
        self.phase_V = voltage_to_phase(voltage_V, gamma_V)
        self.phase_S = voltage_to_phase(voltage_S, gamma_S)
        return self.build_weight_from_phase(delta_list_U, self.phase_U, delta_list_V, self.phase_V, self.phase_S, S_scale)

    def build_phase_from_usv(self, U: Tensor, S: Tensor, V: Tensor) -> Tuple[Tensor, ...]:
        delta_list, phi_mat = self.decomposer.decompose(U.data.clone())
        self.delta_list_U.data.copy_(delta_list)
        self.phase_U.data.copy_(self.decomposer.m2v(phi_mat))

        delta_list, phi_mat = self.decomposer.decompose(V.data.clone())
        self.delta_list_V.data.copy_(delta_list)
        self.phase_V.data.copy_(self.decomposer.m2v(phi_mat))

        self.S_scale.data.copy_(S.data.abs().max(dim=-1, keepdim=True)[0])
        self.phase_S.data.copy_(S.data.div(self.S_scale.data).acos())

        return self.delta_list_U, self.phase_U, self.delta_list_V, self.phase_V, self.phase_S, self.S_scale

    def build_usv_from_phase(self, delta_list_U: Tensor, phase_U: Tensor, delta_list_V: Tensor, phase_V: Tensor, phase_S: Tensor, S_scale: Tensor, update_list: Dict = {"phase_U", "phase_S", "phase_V"}) -> Tuple[Tensor, ...]:
        ### not differentiable
        # reconstruct is time-consuming, a fast method is to only reconstruct based on updated phases
        if("phase_U" in update_list):
            self.U.data.copy_(self.decomposer.reconstruct(
                delta_list_U, self.decomposer.v2m(phase_U)))
        if("phase_V" in update_list):
            self.V.data.copy_(self.decomposer.reconstruct(
                delta_list_V, self.decomposer.v2m(phase_V)))
        if("phase_S" in update_list):
            self.S.data.copy_(phase_S.data.cos().mul_(S_scale))
        return self.U, self.S, self.V

    def build_usv_from_weight(self, weight: Tensor) -> Tuple[Tensor, ...]:
        # differentiable feature is gauranteed
        U, S, V = weight.svd(some=False)
        V = V.transpose(-2, -1).contiguous()
        self.U.data.copy_(U)
        self.S.data.copy_(S)
        self.V.data.copy_(V)
        return U, S, V

    def build_phase_from_weight(self, weight: Tensor) -> Tuple[Tensor, ...]:
        return self.build_phase_from_usv(*self.build_usv_from_weight(weight))

    def build_voltage_from_phase(self, delta_list_U: Tensor, phase_U: Tensor, delta_list_V: Tensor, phase_V: Tensor, phase_S: Tensor, S_scale: Tensor) -> Tuple[Tensor, ...]:
        self.delta_list_U = delta_list_U
        self.delta_list_V = delta_list_V
        self.voltage_U.data.copy_(phase_to_voltage(phase_U, self.gamma))
        self.voltage_S.data.copy_(phase_to_voltage(phase_S, self.gamma))
        self.voltage_V.data.copy_(phase_to_voltage(phase_V, self.gamma))
        self.S_scale.data.copy_(S_scale)

        return self.delta_list_U, self.voltage_U, self.delta_list_V, self.voltage_V, self.voltage_S, self.S_scale

    def build_voltage_from_usv(self, U: Tensor, S: Tensor, V: Tensor) -> Tuple[Tensor, ...]:
        return self.build_voltage_from_phase(*self.build_phase_from_usv(U, S, V))

    def build_voltage_from_weight(self, weight: Tensor) -> Tuple[Tensor, ...]:
        return self.build_voltage_from_phase(*self.build_phase_from_usv(*self.build_usv_from_weight(weight)))

    def sync_parameters(self, src: str = "weight") -> None:
        '''
        description: synchronize all parameters from the source parameters
        '''
        if(src == "weight"):
            # self.build_voltage_from_weight(self.weight) # voltage not supported
            self.build_phase_from_weight(self.weight)
        elif(src == "usv"):
            self.build_phase_from_usv(self.U, self.S, self.V)
            self.build_weight_from_usv(self.U, self.S, self.V)
        elif(src == "phase"):
            if(self.w_bit < 16):
                phase_U = self.phase_U_quantizer(self.phase_U.data)
                phase_S = self.phase_S_quantizer(self.phase_S.data)
                phase_V = self.phase_V_quantizer(self.phase_V.data)
            else:
                phase_U = self.phase_U
                phase_S = self.phase_S
                phase_V = self.phase_V
            if(self.phase_noise_std > 1e-5):
                phase_U = phase_U + gen_gaussian_noise(
                    phase_U, 0, self.phase_noise_std, trunc_range=(-2*self.phase_noise_std, 2*self.phase_noise_std), )

            if(self.phase_bias_U is not None):
                phase_U = phase_U + self.phase_bias_U
            if(self.phase_bias_V is not None):
                phase_V = phase_V + self.phase_bias_V

            self.build_weight_from_phase(
                self.delta_list_U, phase_U, self.delta_list_V, phase_V, phase_S, self.S_scale)
            # self.build_voltage_from_phase(self.delta_list_U, self.phase_U, self.delta_list_V, self.phase_V, self.phase_S, self.S_scale)
        elif(src == "voltage"):
            self.build_weight_from_voltage(self.delta_list_U, self.voltage_U, self.delta_list_V,
                                           self.voltage_V, self.voltage_S, self.gamma, self.gamma, self.gamma, self.S_scale)
        else:
            raise NotImplementedError

    def build_weight(self, update_list: Dict = {"phase_U", "phase_S", 'phase_V'}) -> Tensor:
        if(self.mode == "weight"):  # does not support weight quantization
            return self.weight
        elif(self.mode == "usv"):  # does not support usv quantization
            return self.U, self.S, self.V
        elif(self.mode == "phase"):
            ### not differentiable
            # this mode is used in the initial mapping mode. ZCD on the phases of U and V*; Analytical projection on Sigma
            if(self.w_bit < 16):
                # phase_U = self.phase_quantizer(self.phase_U, self.mixedtraining_mask["phase_U"] if self.mixedtraining_mask is not None else None, mode="triangle")
                # phase_S = self.phase_quantizer(self.phase_S, self.mixedtraining_mask["phase_S"] if self.mixedtraining_mask is not None else None, mode="diagonal")
                # phase_V = self.phase_quantizer(self.phase_V, self.mixedtraining_mask["phase_V"] if self.mixedtraining_mask is not None else None, mode="triangle")
                phase_U = self.phase_U_quantizer(self.phase_U.data)
                phase_S = self.phase_S_quantizer(self.phase_S.data)
                phase_V = self.phase_V_quantizer(self.phase_V.data)
            else:
                phase_U = self.phase_U
                phase_S = self.phase_S
                phase_V = self.phase_V
            if(self.phase_noise_std > 1e-5):
                phase_U = phase_U + gen_gaussian_noise(
                    phase_U, 0, self.phase_noise_std, trunc_range=(-2*self.phase_noise_std, 2*self.phase_noise_std), )

            if(self.phase_bias_U is not None):
                phase_U = phase_U + self.phase_bias_U
            if(self.phase_bias_V is not None):
                phase_V = phase_V + self.phase_bias_V
            return self.build_usv_from_phase(self.delta_list_U, phase_U, self.delta_list_V, phase_V, phase_S, self.S_scale, update_list=update_list)
        elif(self.mode == "voltage"):
            ### not differentiable
            raise NotImplementedError
            if(self.gamma_noise_std > 1e-5):
                gamma_U = gen_gaussian_noise(
                    self.voltage_U, noise_mean=self.gamma, noise_std=self.gamma_noise_std, trunc_range=())
                gamma_S = gen_gaussian_noise(
                    self.voltage_S, noise_mean=self.gamma, noise_std=self.gamma_noise_std, trunc_range=())
                gamma_V = gen_gaussian_noise(
                    self.voltage_V, noise_mean=self.gamma, noise_std=self.gamma_noise_std, trunc_range=())
            else:
                gamma_U = gamma_S = gamma_V = self.gamma
            if(self.w_bit < 16):
                voltage_U = clip_to_valid_quantized_voltage(self.voltage_quantizer(
                    self.voltage_U), self.gamma, self.w_bit, self.v_max, wrap_around=True)
                voltage_S = clip_to_valid_quantized_voltage(self.voltage_quantizer(
                    self.voltage_S), self.gamma, self.w_bit, self.v_max, wrap_around=True)
                voltage_V = clip_to_valid_quantized_voltage(self.voltage_quantizer(
                    self.voltage_V), self.gamma, self.w_bit, self.v_max, wrap_around=True)
            else:
                voltage_U = self.voltage_U
                voltage_S = self.voltage_S
                voltage_V = self.voltage_V
            weight = self.build_weight_from_voltage(
                self.delta_list_U, voltage_U, self.delta_list_V, voltage_V, voltage_S, gamma_U, gamma_V, gamma_S)
        else:
            raise NotImplementedError
        return

    def enable_fast_forward(self) -> None:
        self.fast_forward_flag = True

    def disable_fast_forward(self) -> None:
        self.fast_forward_flag = False

    def set_phase_variation(self, noise_std: float, random_state: Optional[int] = None) -> None:
        self.phase_noise_std = noise_std
        # self.phase_quantizer.set_phase_noise_std(noise_std, random_state)
        # self.unitary_quantizer.set_phase_noise_std(noise_std, random_state)

    def set_crosstalk_factor(self, crosstalk_factor: float) -> None:
        self.crosstalk_factor = crosstalk_factor
        self.phase_U_quantizer.set_crosstalk_factor(crosstalk_factor)
        self.phase_S_quantizer.set_crosstalk_factor(crosstalk_factor)
        self.phase_V_quantizer.set_crosstalk_factor(crosstalk_factor)
        # self.diagonal_quantizer.set_crosstalk_factor(crosstalk_factor)

    def set_gamma_noise(self, noise_std: float = 0, random_state: Optional[int] = None) -> None:
        self.gamma_noise_std = noise_std
        if(random_state is not None):
            random_state_U, random_state_V, random_state_S = random_state, random_state+1, random_state+2
        else:
            random_state_U, random_state_V, random_state_S = random_state, random_state, random_state
        self.phase_U_quantizer.set_gamma_noise(
            noise_std, self.phase_U.size(), random_state_U)
        self.phase_V_quantizer.set_gamma_noise(
            noise_std, self.phase_V.size(), random_state_V)
        self.phase_S_quantizer.set_gamma_noise(
            noise_std, self.phase_S.size(), random_state_S)

    def extract_gamma_noise(self) -> Tuple[Tensor, ...]:
        return self.phase_U_quantizer.noisy_gamma.data, self.phase_S_quantizer.noisy_gamma.data, self.phase_V_quantizer.noisy_gamma.data

    def set_weight_bitwidth(self, w_bit: int) -> None:
        self.w_bit = w_bit
        self.phase_U_quantizer.set_bitwidth(w_bit)
        self.phase_S_quantizer.set_bitwidth(w_bit)
        self.phase_V_quantizer.set_bitwidth(w_bit)

    def load_parameters(self, param_dict: Dict) -> None:
        '''
        description: update parameters based on this parameter dictionary\\
        param param_dict {dict of dict} {layer_name: {param_name: param_tensor, ...}, ...}
        '''
        for name, param in param_dict.items():
            getattr(self, name).data.copy_(param)
        if(self.mode == "phase"):
            self.build_weight(update_list=param_dict)

    def gen_mixedtraining_mask(self, sparsity: float, prefer_small: bool = False, random_state: Optional[int] = None, enable: bool = True) -> Dict[str, Tensor]:
        '''
        description: generate sparsity masks for mixed training \\
        param sparsity {float scalar} fixed parameter ratio, valid range: (0,1]
        prefer_small {bool scalar} True if select phases from small unitary first
        return mask {dict} a dict with all masks for trainable parameters in the current mode. 1/True is for trainable, 0/False is fixed.
        '''
        if(self.mode == "weight"):
            out = {"weight": self.weight.data > percentile(
                self.weight.data, 100*sparsity)}
        elif(self.mode == "usv"):
            # S is forced with 0 sparsity
            out = {"U": self.U.data > percentile(self.U.data, sparsity*100), "S": torch.ones_like(
                self.S.data, dtype=torch.bool), "V": self.V.data > percentile(self.V.data, sparsity*100)}
        elif(self.mode == "phase"):
            # phase_S is forced with 0 sparsity
            # no theoretical guarantee of importance of phase. So random selection is used.
            # for effciency, select phases from small unitary first. Another reason is that larger MZI array is less robust.
            if(prefer_small == False or self.phase_U.size(0) == self.phase_V.size(0)):
                if(random_state is not None):
                    set_torch_deterministic(random_state)
                mask_U = torch.zeros_like(
                    self.phase_U.data).bernoulli_(p=1-sparsity)
                mask_S = torch.ones_like(self.S.data, dtype=torch.bool)
                if(random_state is not None):
                    set_torch_deterministic(random_state+1)
                mask_V = torch.zeros_like(
                    self.phase_V.data).bernoulli_(p=1-sparsity)
            elif(self.phase_U.size(0) < self.phase_V.size(0)):
                total_nonzero = int(
                    (1-sparsity) * (self.phase_U.numel() + self.phase_V.numel()))
                if(total_nonzero <= self.phase_U.numel()):
                    indices = torch.from_numpy(np.random.choice(self.phase_U.numel(), size=[
                                               total_nonzero], replace=False)).to(self.phase_U.device).long()
                    mask_U = torch.zeros_like(
                        self.phase_U.data, dtype=torch.bool)
                    mask_U.data[indices] = 1
                    mask_V = torch.zeros_like(self.phase_V, dtype=torch.bool)
                    mask_S = torch.ones_like(self.S.data, dtype=torch.bool)
                else:
                    indices = torch.from_numpy(np.random.choice(self.phase_V.numel(), size=[
                                               total_nonzero - self.phase_U.numel()], replace=False)).to(self.phase_V.device).long()
                    mask_V = torch.zeros_like(
                        self.phase_V.data, dtype=torch.bool)
                    mask_V.data[indices] = 1
                    mask_U = torch.ones_like(self.phase_U, dtype=torch.bool)
                    mask_S = torch.ones_like(self.S.data, dtype=torch.bool)
            else:
                total_nonzero = int(
                    (1-sparsity) * (self.phase_U.numel() + self.phase_V.numel()))
                if(total_nonzero <= self.phase_V.numel()):
                    indices = torch.from_numpy(np.random.choice(self.phase_V.numel(), size=[
                                               total_nonzero], replace=False)).to(self.phase_V.device).long()
                    mask_V = torch.zeros_like(
                        self.phase_V.data, dtype=torch.bool)
                    mask_V.data[indices] = 1
                    mask_U = torch.zeros_like(self.phase_U, dtype=torch.bool)
                    mask_S = torch.ones_like(self.S.data, dtype=torch.bool)
                else:
                    indices = torch.from_numpy(np.random.choice(self.phase_U.numel(), size=[
                                               total_nonzero - self.phase_V.numel()], replace=False)).to(self.phase_U.device).long()
                    mask_U = torch.zeros_like(
                        self.phase_U.data, dtype=torch.bool)
                    mask_U.data[indices] = 1
                    mask_V = torch.ones_like(self.phase_V, dtype=torch.bool)
                    mask_S = torch.ones_like(self.S.data, dtype=torch.bool)

            out = {"phase_U": mask_U, "phase_S": mask_S, "phase_V": mask_V}
        elif(self.mode == "voltage"):
            # voltage_S is forced with 0 sparsity
            # no theoretical gaurantee of importance of phase. Given phase=gamma*v**2, we assume larger voltage is more important
            mask_U = self.voltage_U > percentile(self.voltage_U, 100*sparsity)
            mask_S = torch.ones_like(self.S.data, dtype=torch.bool)
            mask_V = self.voltage_V > percentile(self.voltage_V, 100*sparsity)
            out = {"voltage_U": mask_U, "voltage_S": mask_S, "voltage_V": mask_V}
        else:
            raise NotImplementedError
        if(enable):
            self.enable_mixedtraining(out)
        return out

    def enable_mixedtraining(self, masks: Tensor) -> None:
        '''
        description: mixed training masks\\
        param masks {dict} {param_name: mask_tensor}
        return
        '''
        self.mixedtraining_mask = masks

    def disable_mixedtraining(self) -> None:
        self.mixedtraining_mask = None

    def switch_mode_to(self, mode: str) -> None:
        self.mode = mode

    def assign_random_phase_bias(self, random_state: int = 42) -> None:
        ### [-1, 1]
        if(random_state is not None):
            set_torch_deterministic(random_state)
        self.phase_bias_U = torch.rand_like(self.phase_U.data)*2-1
        if(random_state is not None):
            set_torch_deterministic(random_state+1)
        self.phase_bias_V = torch.rand_like(self.phase_V.data)*2-1

    def clear_phase_bias(self) -> None:
        self.phase_bias_U.fill_(0)
        self.phase_bias_V.fill_(0)

    def get_power(self, mixtraining_mask: Optional[Tensor] = None) -> float:
        masks = mixtraining_mask if mixtraining_mask is not None else (
            self.mixedtraining_mask if self.mixedtraining_mask is not None else None)
        if(masks is not None):
            power = (
                (self.phase_U.data * masks["phase_U"]) % (2 * np.pi)).sum()
            power += ((self.phase_S.data *
                       masks["phase_S"]) % (2 * np.pi)).sum()
            power += ((self.phase_V.data *
                       masks["phase_V"]) % (2 * np.pi)).sum()
        else:
            power = ((self.phase_U.data) % (2 * np.pi)).sum()
            power += ((self.phase_S.data) % (2 * np.pi)).sum()
            power += ((self.phase_V.data) % (2 * np.pi)).sum()
        return power.item()

    def gen_deterministic_gradient_mask(self, bp_feedback_sparsity: Optional[float] = None) -> torch.Tensor:
        """generate deterministic feedback mask using top k block Frobenius norm strategy.

        Args:
            bp_feedback_sparsity (float, optional): Pruning sparsity. Defaults to None.

        Returns:
            torch.Tensor: feedback matrix mask.
        """
        if(bp_feedback_sparsity is None):
            bp_feedback_sparsity = self.bp_feedback_sparsity
        assert 0 <= bp_feedback_sparsity < 1
        # we prefer uniform column-wise sparsity in W, i.e., row-wise sparsity in W^T
        self.bp_feedback_mask = torch.ones(
            self.grid_dim_y, self.grid_dim_x, device=self.device, dtype=torch.bool)
        # pruned blocks has small total singular value
        s_sum = self.S.data.abs().sum(dim=-1)
        s_thres = torch.quantile(s_sum, q=bp_feedback_sparsity, dim=0)
        self.bp_feedback_mask.masked_fill_(s_sum < s_thres.unsqueeze(0), 0)
        return self.bp_feedback_mask

    def gen_uniform_gradient_mask(self, bp_feedback_sparsity: Optional[float] = None) -> torch.Tensor:
        """generate uniformly sampled feedback matrix mask.

        Args:
            bp_feedback_sparsity (float, optional): Pruning sparsity. Defaults to None.

        Returns:
            torch.Tensor: feedback matrix mask.
        """
        if(bp_feedback_sparsity is None):
            bp_feedback_sparsity = self.bp_feedback_sparsity
        assert 0 <= bp_feedback_sparsity < 1
        # we prefer uniform column-wise sparsity in W, i.e., row-wise sparsity in W^T
        self.bp_feedback_mask = torch.ones(
            self.grid_dim_y, self.grid_dim_x, device=self.device, dtype=torch.bool).bernoulli_(1-bp_feedback_sparsity)
        return self.bp_feedback_mask

    def set_random_input_sparsity(self, bp_input_sparsity: int = 0) -> None:
        assert 0 <= bp_input_sparsity < 1
        self.bp_input_sparsity = bp_input_sparsity

    def gen_random_input_mask(self, x: Tensor, bp_input_sparsity: int = 0) -> None:
        self.bp_input_mask = gen_boolean_mask(
            (x.size(0), self.grid_dim_x), true_prob=1-bp_input_sparsity, device=x.device)
        return self.bp_input_mask

    def set_bp_rank_sampler(self, bp_rank: int, alg: str = "topk", sign: bool = False, random_state: Optional[int] = None) -> None:
        self.bp_rank = min(bp_rank, self.miniblock)
        self.bp_rank_sampler.set_rank(bp_rank, random_state)
        self.bp_rank_sampler.alg = alg
        self.bp_rank_sampler.sign = sign

    def set_bp_feedback_sampler(self, forward_sparsity: float, backward_sparsity: float, alg: str = "topk", normalize: str = "none", random_state: Optional[int] = None):
        self.bp_feedback_sampler.set_sparsity(
            forward_sparsity, backward_sparsity, random_state)
        self.bp_feedback_sampler.alg = alg
        self.bp_feedback_sampler.normalize = normalize

    def set_bp_input_sampler(self, sparsity: float, normalize: str = "none", random_state: Optional[int] = None):
        self.bp_input_sampler.set_sparsity(sparsity, random_state)
        self.bp_input_sampler.normalize = normalize

    def set_noisy_identity(self, flag: bool = True) -> None:
        self.noisy_identity = flag

    def forward(self, x: Tensor) -> Tensor:
        if(self.in_bit < 16):
            x = self.input_quantizer(x)

        if(self.in_channel_pad > self.in_channel):
            if(self.x_zero_pad is None or self.x_zero_pad.size(0) != x.size(0)):
                self.x_zero_pad = torch.zeros(x.size(
                    0), self.in_channel_pad - self.in_channel, device=x.device, dtype=x.dtype)
            x = torch.cat([x, self.x_zero_pad], dim=1)

        if(not self.fast_forward_flag or self.weight is None):
            weight = self.build_weight()  # [p, q, k, k] or u, s, v
        else:
            weight = self.weight

        # if(self.bp_input_sparsity > 1e-8):
        #     self.bp_input_mask = self.gen_random_input_mask(
        #         x, self.bp_input_sparsity)
        if(isinstance(weight, torch.Tensor)):
            weight = weight.permute([0, 2, 1, 3]).contiguous().view(
                self.out_channel_pad, self.in_channel_pad)
            out = F.linear(x, weight, bias=None)
        else:
            u, s, v = weight
            out = sparse_bp_linear(x,
                                   u,
                                   s,
                                   v,
                                   self.I_U if self.noisy_identity else None,
                                   self.I_V if self.noisy_identity else None,
                                   feedback_sampler=self.bp_feedback_sampler,
                                   feature_sampler=self.bp_input_sampler,
                                   rank_sampler=self.bp_rank_sampler,
                                   profiler=self.profiler)

        out = out[:, :self.out_channel]
        if(self.photodetect):
            out = out.square()

        if(self.bias is not None):
            out = out + self.bias.unsqueeze(0)

        return out
