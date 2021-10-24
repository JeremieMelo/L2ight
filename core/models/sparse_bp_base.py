'''
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2021-10-24 16:23:19
LastEditors: Jiaqi Gu (jqgu@utexas.edu)
LastEditTime: 2021-10-24 16:23:20
'''
from typing import Callable, Dict, List, Optional, Tuple, Union

import mlflow
import numpy as np
import torch
import torch.nn.functional as F
from pyutils import SphereDistribution
from pyutils.general import TimerCtx, logger
from pyutils.torch_train import set_torch_deterministic
from torch import Tensor, nn, set_deterministic
from torch.nn.modules.batchnorm import BatchNorm2d
from torch.types import Device, _size
from torchonn.op.mzi_op import (
    RealUnitaryDecomposerBatch,
    checkerboard_to_vector,
    project_matrix_to_unitary,
    vector_to_checkerboard,
)

from .layers.custom_conv2d import MZIBlockConv2d
from .layers.custom_linear import MZIBlockLinear
from .layers.utils import PhaseQuantizer

__all__ = ["SparseBP_Base"]


class SparseBP_Base(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def reset_parameters(self, random_state: int = None) -> None:
        for name, m in self.named_modules():
            if isinstance(m, (MZIBlockConv2d, MZIBlockLinear)):
                if random_state is not None:
                    # deterministic seed, but different for different layer, and controllable by random_state
                    set_torch_deterministic(random_state + sum(map(ord, name)))
                m.reset_parameters()
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def backup_phases(self) -> None:
        self.phase_backup = {}
        for layer_name, layer in self.fc_layers.items():
            self.phase_backup[layer_name] = {
                "weight": layer.weight.data.clone() if layer.weight is not None else None,
                "U": layer.U.data.clone() if layer.U is not None else None,
                "S": layer.S.data.clone() if layer.S is not None else None,
                "V": layer.V.data.clone() if layer.V is not None else None,
                "delta_list_U": layer.delta_list_U.data.clone() if layer.delta_list_U is not None else None,
                "phase_U": layer.phase_U.data.clone() if layer.phase_U is not None else None,
                "delta_list_V": layer.delta_list_V.data.clone() if layer.delta_list_V is not None else None,
                "phase_V": layer.phase_V.data.clone() if layer.phase_V is not None else None,
            }

    def restore_phases(self) -> None:
        for layer_name, layer in self.fc_layers.items():
            backup = self.phase_backup[layer_name]
            for param_name, param_src in backup.items():
                param_dst = getattr(layer, param_name)
                if param_src is not None and param_dst is not None:
                    param_dst.data.copy_(param_src.data)

    def set_gamma_noise(self, noise_std: float = 0.0, random_state: Optional[int] = None) -> None:
        self.gamma_noise_std = noise_std
        for layer in self.modules():
            if isinstance(layer, (MZIBlockConv2d, MZIBlockLinear)):
                layer.set_gamma_noise(noise_std, random_state=random_state)

    def set_crosstalk_factor(self, crosstalk_factor: float = 0.0) -> None:
        self.crosstalk_factor = crosstalk_factor
        for layer in self.modules():
            if isinstance(layer, (MZIBlockConv2d, MZIBlockLinear)):
                layer.set_crosstalk_factor(crosstalk_factor)

    def set_weight_bitwidth(self, w_bit: int) -> None:
        self.w_bit = w_bit
        for layer in self.modules():
            if isinstance(layer, (MZIBlockLinear, MZIBlockConv2d)):
                layer.set_weight_bitwidth(w_bit)

    def get_num_device(self) -> Dict[str, int]:
        total_mzi = 0
        for layer in self.modules():
            if isinstance(layer, (MZIBlockLinear, MZIBlockConv2d)):
                total_mzi += layer.in_channel_pad * layer.out_channel_pad

        return {"mzi": total_mzi}

    def unitary_projection(self) -> None:
        for layer in self.modules():
            if isinstance(layer, (MZIBlockLinear, MZIBlockConv2d)):
                if layer.U is not None:
                    layer.U.data.copy_(projection_matrix_to_unitary(layer.U.data))
                    layer.V.data.copy_(projection_matrix_to_unitary(layer.V.data))

    def get_unitary_loss(self) -> Tensor:
        loss_list = []
        self.unitary_loss_cache = {}
        for layer_name, layer in self.fc_layers.items():
            if layer.U is not None and layer.U.requires_grad == True:
                if layer_name not in self.unitary_loss_cache:
                    eye_U, eye_V = self.unitary_loss_cache[layer_name] = [
                        torch.eye(layer.U.size(0), device=layer.U.device),
                        torch.eye(layer.V.size(0), device=layer.V.device),
                    ]
                else:
                    eye_U, eye_V = self.unitary_loss_cache[layer_name]
                loss_list.extend(
                    [
                        F.mse_loss(torch.matmul(layer.U, layer.U.t()), eye_U),
                        F.mse_loss(torch.matmul(layer.V, layer.V.t()), eye_V),
                    ]
                )

        return sum(loss_list) / len(loss_list)

    def load_parameters(self, param_dict: Dict[str, Dict[str, Tensor]]) -> None:
        """
        description: update parameters based on this parameter dictionary\\
        param param_dict {dict of dict} {layer_name: {param_name: param_tensor, ...}, ...}
        """
        for layer_name, layer_param_dict in param_dict.items():
            self.layers[layer_name].load_parameters(layer_param_dict)

    def build_obj_fn(self, X: Tensor, y: Tensor, criterion: Callable) -> Callable:
        def obj_fn(X_cur=None, y_cur=None, param_dict=None):
            if param_dict is not None:
                self.load_parameters(param_dict)
            if X_cur is None or y_cur is None:
                data, target = X, y
            else:
                data, target = X_cur, y_cur
            pred = self.forward(data)
            return criterion(pred, target)

        return obj_fn

    def enable_fast_forward(self) -> None:
        for layer in self.modules():
            if isinstance(layer, (MZIBlockLinear, MZIBlockConv2d)):
                layer.enable_fast_forward()

    def disable_fast_forward(self) -> None:
        for layer in self.modules():
            if isinstance(layer, (MZIBlockLinear, MZIBlockConv2d)):
                layer.disable_fast_forward()

    def sync_parameters(self, src: str = "weight") -> None:
        for layer in self.modules():
            if isinstance(layer, (MZIBlockLinear, MZIBlockConv2d)):
                layer.sync_parameters(src=src)

    def gen_mixedtraining_mask(
        self, sparsity: float, prefer_small: bool = False, random_state: Optional[int] = None
    ) -> Dict[str, Dict[str, Tensor]]:
        return {
            layer_name: layer.gen_mixedtraining_mask(sparsity, prefer_small, random_state)
            for layer_name, layer in self.named_modules()
            if isinstance(layer, (MZIBlockLinear, MZIBlockConv2d))
        }

    def switch_mode_to(self, mode: str) -> None:
        for layer in self.modules():
            if isinstance(layer, (MZIBlockLinear, MZIBlockConv2d)):
                layer.switch_mode_to(mode)

    def assign_random_phase_bias(self, random_state: Optional[int] = 42) -> None:
        for layer in self.modules():
            if isinstance(layer, (MZIBlockLinear, MZIBlockConv2d)):
                layer.assign_random_phase_bias(random_state)

    def clear_phase_bias(self) -> None:
        for layer in self.modules():
            if isinstance(layer, (MZIBlockLinear, MZIBlockConv2d)):
                layer.clear_phase_bias()

    def set_noisy_identity(self, flag: bool = True) -> None:
        for layer in self.modules():
            if isinstance(layer, (MZIBlockLinear, MZIBlockConv2d)):
                layer.set_noisy_identity(flag)

    def get_power(self, mixedtraining_mask: Optional[Tensor] = None) -> float:
        power = sum(
            layer.get_power(mixedtraining_mask[layer_name]) for layer_name, layer in self.fc_layers.items()
        )
        return power

    def gen_deterministic_gradient_mask(self, bp_feedback_sparsity: Optional[float] = None) -> None:
        for layer in self.fc_layers.values():
            layer.gen_deterministic_gradient_mask(bp_feedback_sparsity=bp_feedback_sparsity)

    def gen_uniform_gradient_mask(self, bp_feedback_sparsity: Optional[float] = None) -> None:
        for layer in self.fc_layers.values():
            layer.gen_uniform_gradient_mask(bp_feedback_sparsity=bp_feedback_sparsity)

    def set_random_input_sparsity(self, bp_input_sparsity: Optional[float] = None) -> None:
        for layer in self.fc_layers.values():
            layer.set_random_input_sparsity(bp_input_sparsity)

    def set_bp_feedback_sampler(
        self,
        forward_sparsity: float,
        backward_sparsity: float,
        alg: str = "topk",
        normalize: bool = False,
        random_state: Optional[int] = None,
    ):
        for layer in self.modules():
            # if(isinstance(layer, (MZIBlockLinear, MZIBlockConv2d))):
            # Linear is not the bottleneck but critical to performance. Recommend not to sample Linear layer!
            if isinstance(layer, (MZIBlockConv2d,)):
                layer.set_bp_feedback_sampler(
                    forward_sparsity, backward_sparsity, alg, normalize, random_state
                )

    def set_bp_input_sampler(
        self,
        sparsity: float,
        spatial_sparsity: float,
        column_sparsity: float,
        normalize: bool = False,
        random_state: Optional[int] = None,
        sparsify_first_conv: bool = True,
    ):
        counter = 0
        for layer in self.modules():
            if isinstance(layer, MZIBlockLinear):
                layer.set_bp_input_sampler(sparsity, normalize, random_state)
            elif isinstance(layer, MZIBlockConv2d):
                if counter == 0:
                    # always donot apply spatial sampling to first conv.
                    # first conv is not memory bottleneck, not runtime bottleneck, but energy bottleneck
                    if sparsify_first_conv:
                        layer.set_bp_input_sampler(0, column_sparsity, normalize, random_state)
                    else:
                        layer.set_bp_input_sampler(0, 0, normalize, random_state)
                    counter += 1
                else:
                    layer.set_bp_input_sampler(spatial_sparsity, column_sparsity, normalize, random_state)

    def set_bp_rank_sampler(
        self, bp_rank: int, alg: str = "topk", sign: bool = False, random_state: Optional[int] = None
    ) -> None:
        for layer in self.modules():
            if isinstance(layer, (MZIBlockLinear, MZIBlockConv2d)):
                layer.set_bp_rank_sampler(bp_rank, alg, sign, random_state)

    def stack_all_params_dict(self) -> Tuple[Tensor, ...]:
        delta_list_U_dict, phase_U_dict, S_dict, delta_list_V_dict, phase_V_dict = {}, {}, {}, {}, {}
        phase_bias_U_dict, phase_bias_V_dict = {}, {}
        for mode in ["conv", "linear"]:
            delta_list_U, phase_U, S, delta_list_V, phase_V = [], [], [], [], []
            phase_bias_U, phase_bias_V = [], []
            for layer in self.modules():
                if isinstance(layer, MZIBlockConv2d if mode == "conv" else MZIBlockLinear):
                    delta_list_U.append(layer.delta_list_U.data.flatten(0, 1))  # [p,q,k]
                    # [p,q,k(k-1)/2]
                    phase_U.append(layer.phase_U.data.flatten(0, 1))
                    phase_bias_U.append(layer.phase_bias_U.data.flatten(0, 1))
                    S.append(layer.S.data.flatten(0, 1))  # [p,q,k]
                    delta_list_V.append(layer.delta_list_V.data.flatten(0, 1))  # [p,q,k]
                    # [p,q,k(k-1)/2]
                    phase_V.append(layer.phase_V.data.flatten(0, 1))
                    phase_bias_V.append(layer.phase_bias_V.data.flatten(0, 1))
            delta_list_U = torch.cat(delta_list_U, dim=0)
            phase_U = torch.cat(phase_U, dim=0)
            phase_bias_U = torch.cat(phase_bias_U, dim=0)
            S = torch.cat(S, dim=0)
            delta_list_V = torch.cat(delta_list_V, dim=0)
            phase_V = torch.cat(phase_V, dim=0)
            phase_bias_V = torch.cat(phase_bias_V, dim=0)
            (
                delta_list_U_dict[mode],
                phase_U_dict[mode],
                phase_bias_U_dict[mode],
                S_dict[mode],
                delta_list_V_dict[mode],
                phase_V_dict[mode],
                phase_bias_V_dict[mode],
            ) = (delta_list_U, phase_U, phase_bias_U, S, delta_list_V, phase_V, phase_bias_V)

        return (
            delta_list_U_dict,
            phase_U_dict,
            phase_bias_U_dict,
            S_dict,
            delta_list_V_dict,
            phase_V_dict,
            phase_bias_V_dict,
        )

    def stack_all_target_dict(self) -> torch.Tensor:
        weight_dict = {}
        weights = []
        for mode in ["conv", "linear"]:
            for layer in self.modules():
                if isinstance(layer, MZIBlockConv2d if mode == "conv" else MZIBlockLinear):
                    weights.append(layer.weight.data.flatten(0, 1))
            weights = torch.cat(weights, dim=0)
            weight_dict[mode] = weights
        return weight_dict

    def stack_all_identity_dict(self) -> Tuple[torch.Tensor, torch.Tensor]:
        I_U_dict, I_V_dict = {}, {}
        I_U = []
        I_V = []
        for mode in ["conv", "linear"]:
            for layer in self.modules():
                if isinstance(layer, MZIBlockConv2d if mode == "conv" else MZIBlockLinear):
                    I_U.append(layer.I_U.data.flatten(0, 1))
                    I_V.append(layer.I_V.data.flatten(0, 1))
            I_U = torch.cat(I_U, dim=0)
            I_V = torch.cat(I_V, dim=0)
            I_U_dict[mode], I_V_dict[mode] = I_U, I_V
        return I_U_dict, I_V_dict

    def stack_all_phase_quantizer_dict(self) -> Tuple[PhaseQuantizer, PhaseQuantizer, PhaseQuantizer]:
        phase_U_quantizer_dict = {}
        phase_S_quantizer_dict = {}
        phase_V_quantizer_dict = {}
        for mode in ["conv", "linear"]:
            phase_U_quantizer = PhaseQuantizer(
                self.w_bit,
                self.v_pi,
                self.v_max,
                self.gamma_noise_std,
                self.crosstalk_factor,
                5,
                mode="rectangle",
                device=self.device,
            )
            phase_S_quantizer = PhaseQuantizer(
                self.w_bit,
                self.v_pi,
                self.v_max,
                self.gamma_noise_std,
                self.crosstalk_factor,
                3,
                mode="diagonal",
                device=self.device,
            )
            phase_V_quantizer = PhaseQuantizer(
                self.w_bit,
                self.v_pi,
                self.v_max,
                self.gamma_noise_std,
                self.crosstalk_factor,
                5,
                mode="rectangle",
                device=self.device,
            )
            noisy_gamma_U, noisy_gamma_S, noisy_gamma_V = [], [], []
            for layer in self.modules():
                if isinstance(layer, MZIBlockConv2d if mode == "conv" else MZIBlockLinear):
                    noisy_gamma_U.append(layer.phase_U_quantizer.noisy_gamma.data.flatten(0, 1))
                    noisy_gamma_S.append(layer.phase_S_quantizer.noisy_gamma.data.flatten(0, 1))
                    noisy_gamma_V.append(layer.phase_V_quantizer.noisy_gamma.data.flatten(0, 1))

            noisy_gamma_U = torch.cat(noisy_gamma_U, dim=0)
            noisy_gamma_S = torch.cat(noisy_gamma_S, dim=0)
            noisy_gamma_V = torch.cat(noisy_gamma_V, dim=0)
            phase_U_quantizer.set_gamma_noise(self.gamma_noise_std, noisy_gamma_U.size())
            phase_S_quantizer.set_gamma_noise(self.gamma_noise_std, noisy_gamma_S.size())
            phase_V_quantizer.set_gamma_noise(self.gamma_noise_std, noisy_gamma_V.size())
            phase_U_quantizer.noisy_gamma.data.copy_(noisy_gamma_U)
            phase_S_quantizer.noisy_gamma.data.copy_(noisy_gamma_S)
            phase_V_quantizer.noisy_gamma.data.copy_(noisy_gamma_V)
            phase_U_quantizer_dict[mode] = phase_U_quantizer
            phase_S_quantizer_dict[mode] = phase_S_quantizer
            phase_V_quantizer_dict[mode] = phase_V_quantizer
        return phase_U_quantizer_dict, phase_S_quantizer_dict, phase_V_quantizer_dict

    def stack_all_params(self) -> Tuple[Tensor, ...]:
        delta_list_U, phase_U, S, delta_list_V, phase_V = [], [], [], [], []
        phase_bias_U, phase_bias_V = [], []
        for layer in self.modules():
            if isinstance(layer, (MZIBlockLinear, MZIBlockConv2d)):
                delta_list_U.append(layer.delta_list_U.data.flatten(0, 1))  # [p,q,k]
                # [p,q,k(k-1)/2]
                phase_U.append(layer.phase_U.data.flatten(0, 1))
                phase_bias_U.append(layer.phase_bias_U.data.flatten(0, 1))
                S.append(layer.S.data.flatten(0, 1))  # [p,q,k]
                delta_list_V.append(layer.delta_list_V.data.flatten(0, 1))  # [p,q,k]
                # [p,q,k(k-1)/2]
                phase_V.append(layer.phase_V.data.flatten(0, 1))
                phase_bias_V.append(layer.phase_bias_V.data.flatten(0, 1))
        delta_list_U = torch.cat(delta_list_U, dim=0)
        phase_U = torch.cat(phase_U, dim=0)
        phase_bias_U = torch.cat(phase_bias_U, dim=0)
        S = torch.cat(S, dim=0)
        delta_list_V = torch.cat(delta_list_V, dim=0)
        phase_V = torch.cat(phase_V, dim=0)
        phase_bias_V = torch.cat(phase_bias_V, dim=0)
        return delta_list_U, phase_U, phase_bias_U, S, delta_list_V, phase_V, phase_bias_V

    def stack_all_target(self) -> torch.Tensor:
        weights = []
        for layer in self.modules():
            if isinstance(layer, (MZIBlockLinear, MZIBlockConv2d)):
                weights.append(layer.weight.data.flatten(0, 1))
        weights = torch.cat(weights, dim=0)
        return weights

    def stack_all_identity(self) -> Tuple[torch.Tensor, torch.Tensor]:
        I_U = []
        I_V = []
        for layer in self.modules():
            if isinstance(layer, (MZIBlockLinear, MZIBlockConv2d)):
                I_U.append(layer.I_U.data.flatten(0, 1))
                I_V.append(layer.I_V.data.flatten(0, 1))
        I_U = torch.cat(I_U, dim=0)
        I_V = torch.cat(I_V, dim=0)
        return I_U, I_V

    def stack_all_phase_quantizer(self) -> Tuple[PhaseQuantizer, PhaseQuantizer, PhaseQuantizer]:
        phase_U_quantizer = PhaseQuantizer(
            self.w_bit,
            self.v_pi,
            self.v_max,
            self.gamma_noise_std,
            self.crosstalk_factor,
            5,
            mode="rectangle",
            device=self.device,
        )
        phase_S_quantizer = PhaseQuantizer(
            self.w_bit,
            self.v_pi,
            self.v_max,
            self.gamma_noise_std,
            self.crosstalk_factor,
            3,
            mode="diagonal",
            device=self.device,
        )
        phase_V_quantizer = PhaseQuantizer(
            self.w_bit,
            self.v_pi,
            self.v_max,
            self.gamma_noise_std,
            self.crosstalk_factor,
            5,
            mode="rectangle",
            device=self.device,
        )
        noisy_gamma_U, noisy_gamma_S, noisy_gamma_V = [], [], []
        for layer in self.modules():
            if isinstance(layer, (MZIBlockLinear, MZIBlockConv2d)):
                noisy_gamma_U.append(layer.phase_U_quantizer.noisy_gamma.data.flatten(0, 1))
                noisy_gamma_S.append(layer.phase_S_quantizer.noisy_gamma.data.flatten(0, 1))
                noisy_gamma_V.append(layer.phase_V_quantizer.noisy_gamma.data.flatten(0, 1))

        noisy_gamma_U = torch.cat(noisy_gamma_U, dim=0)
        noisy_gamma_S = torch.cat(noisy_gamma_S, dim=0)
        noisy_gamma_V = torch.cat(noisy_gamma_V, dim=0)
        phase_U_quantizer.set_gamma_noise(self.gamma_noise_std, noisy_gamma_U.size())
        phase_S_quantizer.set_gamma_noise(self.gamma_noise_std, noisy_gamma_S.size())
        phase_V_quantizer.set_gamma_noise(self.gamma_noise_std, noisy_gamma_V.size())
        phase_U_quantizer.noisy_gamma.data.copy_(noisy_gamma_U)
        phase_S_quantizer.noisy_gamma.data.copy_(noisy_gamma_S)
        phase_V_quantizer.noisy_gamma.data.copy_(noisy_gamma_V)
        return phase_U_quantizer, phase_S_quantizer, phase_V_quantizer

    def identity_calibration_zgd(
        self,
        lr: Optional[float] = 0.1,
        n_epochs: Optional[int] = 400,
        lr_gamma: Optional[float] = 0.99,
        lr_min: Optional[float] = 0.006,
        random_state: Optional[int] = None,
        adaptive: bool = False,
        best_record: bool = False,
        verbose=True,
    ) -> None:
        lr_min = (2 * np.pi) / (2 ** self.w_bit - 1)
        if verbose:
            logger.info(
                f"Start identity calibration...\n\tlr: {lr}\tn_epochs: {n_epochs}\tlr_gamma: {lr_gamma}\tlr_min: {lr_min}"
            )
        # assume all layers share the same tensor core size! we can collect all blocks and optimize in parallel
        delta_list_U, phase_U, phase_bias_U, S, delta_list_V, phase_V, phase_bias_V = self.stack_all_params()
        delta_list_U.fill_(1)
        delta_list_V.fill_(1)
        phase_U.fill_(0)
        phase_V.fill_(0)
        phase_U_quantizer, phase_S_quantizer, phase_V_quantizer = self.stack_all_phase_quantizer()

        S = torch.linspace(0.1, 1, S.size(-1), device=self.device).unsqueeze(0)  # [1, k]
        x = 1 / S  # Sigma^-1
        target = torch.eye(S.size(-1), device=self.device).unsqueeze(0)

        decomposer = RealUnitaryDecomposerBatch(alg="clements")
        v2m = vector_to_checkerboard
        n_step = phase_U.size(-1)

        def build_unitary(delta_list, phase, phase_bias, quantizer=None):
            if quantizer is not None:
                phase = quantizer(phase)
            return decomposer.reconstruct(delta_list, v2m(phase + phase_bias))

        U = build_unitary(delta_list_U, phase_U, phase_bias_U, phase_U_quantizer)
        V = build_unitary(delta_list_V, phase_V, phase_bias_V, phase_V_quantizer)

        def model(x, phase_U=None, phase_V=None, U=None, V=None):
            if phase_U is not None:
                Ur = build_unitary(delta_list_U, phase_U, phase_bias_U, phase_U_quantizer)
            else:
                Ur = U
            if phase_V is not None:
                Vr = build_unitary(delta_list_V, phase_V, phase_bias_V, phase_V_quantizer)
            else:
                Vr = V
            # [bs*p*q, k, k]
            out = Ur.matmul(S.unsqueeze(-1).mul(Vr.mul(x.unsqueeze(-2))))

            return out

        def obj_fn(y, target):
            ### y [bs*p*q, k, k]
            ### target [1, k, k]
            out = (y - target).square_().mean(dim=(-1, -2))  # [bs*p*q]
            return out

        def criterion(W):
            ### W [bs*p*q, k, k]
            return (W.abs() - target).square().mean(dim=(-1, -2))

        loss = obj_fn(model(x, U=U, V=V), target)
        init_lr = 1
        lr_gamma = 0.99
        lr = torch.zeros_like(loss).fill_(init_lr)
        epoch_list = torch.zeros_like(loss)
        init_lr_list = lr.clone()

        def lr_scheduler(epoch_list):
            return init_lr_list.mul(lr_gamma ** epoch_list).clamp_(min=lr_min)

        def restart(loss, epoch_list, phase_U, phase_V):
            mask = loss > 0.1
            n_restart = mask.float().sum().item()
            if n_restart > 0:
                logger.info(f"{n_restart} blocks restarted")
            epoch_list[mask] = 0
            phase_U[mask] += 0.01
            phase_V[mask] += 0.01

        # record the best identity
        best_loss = loss.clone()
        best_U = U.clone()
        best_V = V.clone()
        total_step = 0
        global_epoch = 0

        metric_U = criterion(best_U)
        metric_V = criterion(best_V)
        metric_U_avg, metric_U_std, metric_U_min, metric_U_max = (
            metric_U.mean().item(),
            metric_U.std().item(),
            metric_U.min().item(),
            metric_U.max().item(),
        )
        metric_V_avg, metric_V_std, metric_V_min, metric_V_max = (
            metric_V.mean().item(),
            metric_V.std().item(),
            metric_V.min().item(),
            metric_V.max().item(),
        )
        logger.info(
            f"Epoch: {global_epoch}\tlr: {lr.mean().item():.5f}\tloss: {loss.mean().item():.6f}\teye_loss_U: {criterion(U).mean().item():.5f}\teye_loss_V: {criterion(V).mean().item():.5f}, best_eye_loss_U: avg={metric_U_avg:.5f}, std={metric_U_std:.5f}, max={metric_U_max:.5f}, best_eye_loss_V: avg={metric_V_avg:.5f}, std={metric_V_std:.5f}, max={metric_V_max:.5f}"
        )
        logger.info(best_U[0].diag())
        mlflow.log_metrics(
            {
                "iloss_avg": loss.data.mean().item(),
                "iloss_std": loss.std().item(),
                "iloss_min": loss.min().item(),
                "iloss_max": loss.max().item(),
                "imetric_u_avg": metric_U_avg,
                "imetric_u_std": metric_U_std,
                "imetric_u_min": metric_U_min,
                "imetric_u_max": metric_U_max,
                "imetric_v_avg": metric_V_avg,
                "imetric_v_std": metric_V_std,
                "imetric_v_min": metric_V_min,
                "imetric_v_max": metric_V_max,
                "icore_call": total_step * best_U.size(0),
            },
            step=total_step,
        )

        sigma = 0.1

        def sample_perturbation(size):
            with torch.random.fork_rng():
                torch.random.manual_seed(np.random.randint(0, 1000))
                return torch.randn(size, device=self.device).mul_(sigma)

        c = 1
        m = torch.zeros([2] + list(phase_U.shape), device=self.device)
        n = 3
        with TimerCtx() as t:
            while (epoch_list < n_epochs).any().item():
                # Step 1: optimize U
                for _ in range(phase_U.size(-1)):
                    grad = 0
                    for _ in range(n):
                        delta_phase = sample_perturbation([2] + list(phase_U.shape))
                        phase_U_perturb = phase_U + delta_phase[0]
                        phase_V_perturb = phase_V + delta_phase[1]
                        cur_loss = obj_fn(model(x, phase_U=phase_U_perturb, phase_V=phase_V_perturb), target)
                        total_step += 1
                        loss_diff = c * (cur_loss - loss)

                        grad += 1 / n * delta_phase * loss_diff.unsqueeze(-1).unsqueeze(0)
                    m = 0.9 * m + lr.unsqueeze(-1).unsqueeze(0) * grad
                    phase_U -= m[0]
                    phase_V -= m[1]

                    U = build_unitary(delta_list_U, phase_U, phase_bias_U, phase_U_quantizer)
                    V = build_unitary(delta_list_V, phase_V, phase_bias_V, phase_V_quantizer)
                    loss = obj_fn(model(x, U=U, V=V), target)
                    total_step += 1
                    # record
                    if best_record:
                        mask = best_loss > loss
                        best_loss[mask] = loss[mask]
                        best_U[mask, ...] = U[mask, ...]
                        best_V[mask, ...] = V[mask, ...]
                    else:
                        best_loss = loss
                        best_U = U

                epoch_list += 1
                global_epoch += 1

                if adaptive:
                    restart(loss, epoch_list, phase_U, phase_V)
                    loss = obj_fn(model(x, phase_V=phase_V, phase_U=phase_U), target)

                lr = lr_scheduler(epoch_list)

                if global_epoch % 20 == 0 and verbose:
                    metric_U = criterion(best_U)
                    metric_V = criterion(best_V)
                    metric_U_avg, metric_U_std, metric_U_min, metric_U_max = (
                        metric_U.mean().item(),
                        metric_U.std().item(),
                        metric_U.min().item(),
                        metric_U.max().item(),
                    )
                    metric_V_avg, metric_V_std, metric_V_min, metric_V_max = (
                        metric_V.mean().item(),
                        metric_V.std().item(),
                        metric_V.min().item(),
                        metric_V.max().item(),
                    )
                    logger.info(
                        f"Epoch: {global_epoch}\tlr: {lr.mean().item():.5f}\tloss: {loss.mean().item():.6f}\teye_loss_U: {criterion(U).mean().item():.5f}\teye_loss_V: {criterion(V).mean().item():.5f}, best_eye_loss_U: avg={metric_U_avg:.5f}, std={metric_U_std:.5f}, max={metric_U_max:.5f}, best_eye_loss_V: avg={metric_V_avg:.5f}, std={metric_V_std:.5f}, max={metric_V_max:.5f}"
                    )
                    logger.info(best_U[0].diag())
                    mlflow.log_metrics(
                        {
                            "iloss_avg": loss.data.mean().item(),
                            "iloss_std": loss.std().item(),
                            "iloss_min": loss.min().item(),
                            "iloss_max": loss.max().item(),
                            "imetric_u_avg": metric_U_avg,
                            "imetric_u_std": metric_U_std,
                            "imetric_u_min": metric_U_min,
                            "imetric_u_max": metric_U_max,
                            "imetric_v_avg": metric_V_avg,
                            "imetric_v_std": metric_V_std,
                            "imetric_v_min": metric_V_min,
                            "imetric_v_max": metric_V_max,
                            "icore_call": total_step * best_U.size(0),
                        },
                        step=total_step,
                    )

        # record pseudo-identity
        i = 0
        for layer in self.modules():
            if isinstance(layer, (MZIBlockConv2d, MZIBlockLinear)):
                p, q, k = layer.grid_dim_y, layer.grid_dim_x, layer.miniblock
                layer.I_U = torch.nn.Parameter(
                    best_U[i : i + p * q].clone().view(p, q, k, k), requires_grad=False
                )  # can be saved in checkpoint
                layer.I_V = torch.nn.Parameter(
                    best_V[i : i + p * q].clone().view(p, q, k, k), requires_grad=False
                )
                i += p * q
        if verbose:
            logger.info(
                f"Identity calibration done in {t.interval} s (Total step: {total_step}, total core call: {total_step * best_U.size(0)}). Pseudo-I is recorded."
            )

    def identity_calibration_zcd(
        self,
        lr: Optional[float] = 0.1,
        n_epochs: Optional[int] = 400,
        lr_gamma: Optional[float] = 0.99,
        lr_min: Optional[float] = 0.006,
        random_state: Optional[int] = None,
        adaptive: bool = False,
        best_record: bool = False,
        verbose=True,
    ) -> None:
        lr_min = (2 * np.pi) / (2 ** self.w_bit - 1)
        if verbose:
            logger.info(
                f"Start identity calibration...\n\tlr: {lr}\tn_epochs: {n_epochs}\tlr_gamma: {lr_gamma}\tlr_min: {lr_min}"
            )
        # assume all layers share the same tensor core size! we can collect all blocks and optimize in parallel
        delta_list_U, phase_U, phase_bias_U, S, delta_list_V, phase_V, phase_bias_V = self.stack_all_params()
        delta_list_U.fill_(1)
        delta_list_V.fill_(1)
        phase_U.fill_(0)
        phase_V.fill_(0)
        phase_U_quantizer, phase_S_quantizer, phase_V_quantizer = self.stack_all_phase_quantizer()

        S = torch.linspace(0.1, 1, S.size(-1), device=self.device).unsqueeze(0)  # [1, k]
        x = 1 / S  # Sigma^-1
        target = torch.eye(S.size(-1), device=self.device).unsqueeze(0)

        decomposer = RealUnitaryDecomposerBatch(alg="clements")
        v2m = vector_to_checkerboard
        n_step = phase_U.size(-1)

        def build_unitary(delta_list, phase, phase_bias, quantizer=None):
            if quantizer is not None:
                phase = quantizer(phase)
            return decomposer.reconstruct(delta_list, v2m(phase + phase_bias))

        U = build_unitary(delta_list_U, phase_U, phase_bias_U, phase_U_quantizer)
        V = build_unitary(delta_list_V, phase_V, phase_bias_V, phase_V_quantizer)

        def model(x, phase_U=None, phase_V=None, U=None, V=None):
            if phase_U is not None:
                Ur = build_unitary(delta_list_U, phase_U, phase_bias_U, phase_U_quantizer)
            else:
                Ur = U
            if phase_V is not None:
                Vr = build_unitary(delta_list_V, phase_V, phase_bias_V, phase_V_quantizer)
            else:
                Vr = V
            # [bs*p*q, k, k]
            out = Ur.matmul(S.unsqueeze(-1).mul(Vr.mul(x.unsqueeze(-2))))

            return out

        def obj_fn(y, target):
            ### y [bs*p*q, k, k]
            ### target [1, k, k]
            out = (y - target).square_().mean(dim=(-1, -2))  # [bs*p*q]
            return out

        def criterion(W):
            ### W [bs*p*q, k, k]
            return (W.abs() - target).square().mean(dim=(-1, -2))

        loss = obj_fn(model(x, U=U, V=V), target)
        init_lr = lr
        lr = torch.zeros_like(loss).fill_(init_lr)
        epoch_list = torch.zeros_like(loss)
        init_lr_list = lr.clone()

        def lr_scheduler(epoch_list):
            return init_lr_list.mul(lr_gamma ** epoch_list).clamp_(min=lr_min)

        def restart(loss, epoch_list, phase_U, phase_V):
            mask = loss > 0.09
            n_restart = mask.float().sum().item()
            if n_restart > 0:
                logger.info(f"{n_restart} blocks restarted")
            epoch_list[mask] = 0
            phase_U[mask] += 0.01
            phase_V[mask] += 0.01

        # record the best identity
        best_loss = loss.clone()
        best_U = U.clone()
        best_V = V.clone()
        total_step = 0
        global_epoch = 0
        metric_U = criterion(best_U)
        metric_V = criterion(best_V)
        metric_U_avg, metric_U_std, metric_U_min, metric_U_max = (
            metric_U.mean().item(),
            metric_U.std().item(),
            metric_U.min().item(),
            metric_U.max().item(),
        )
        metric_V_avg, metric_V_std, metric_V_min, metric_V_max = (
            metric_V.mean().item(),
            metric_V.std().item(),
            metric_V.min().item(),
            metric_V.max().item(),
        )
        logger.info(
            f"Epoch: {global_epoch}\tlr: {lr.mean().item():.5f}\tloss: {loss.mean().item():.6f}\teye_loss_U: {criterion(U).mean().item():.5f}\teye_loss_V: {criterion(V).mean().item():.5f}, best_eye_loss_U: avg={metric_U_avg:.5f}, std={metric_U_std:.5f}, max={metric_U_max:.5f}, best_eye_loss_V: avg={metric_V_avg:.5f}, std={metric_V_std:.5f}, max={metric_V_max:.5f}"
        )
        logger.info(best_U[0].diag())
        mlflow.log_metrics(
            {
                "iloss_avg": loss.data.mean().item(),
                "iloss_std": loss.std().item(),
                "iloss_min": loss.min().item(),
                "iloss_max": loss.max().item(),
                "imetric_u_avg": metric_U_avg,
                "imetric_u_std": metric_U_std,
                "imetric_u_min": metric_U_min,
                "imetric_u_max": metric_U_max,
                "imetric_v_avg": metric_V_avg,
                "imetric_v_std": metric_V_std,
                "imetric_v_min": metric_V_min,
                "imetric_v_max": metric_V_max,
                "icore_call": total_step * best_U.size(0),
            },
            step=total_step,
        )

        with TimerCtx() as t:
            while (epoch_list < n_epochs).any().item():
                # Step 1: optimize U
                phase = phase_U
                indices = np.arange(n_step)
                np.random.shuffle(indices)
                for i in indices:
                    phase[..., i] += lr
                    loss_pos = obj_fn(model(x, phase_U=phase, V=V), target)
                    total_step += 1
                    ascent = loss_pos >= loss
                    phase[..., i] -= ascent.float().mul(2 * lr)
                    loss_neg = obj_fn(model(x, phase_U=phase, V=V), target)
                    total_step += 1
                    loss = torch.where(ascent, loss_neg, loss_pos)

                phase_U = phase
                U = build_unitary(delta_list_U, phase_U, phase_bias_U, phase_U_quantizer)
                # record
                if best_record:
                    mask = best_loss > loss
                    best_loss[mask] = loss[mask]
                    best_U[mask, ...] = U[mask, ...]
                else:
                    best_loss = loss
                    best_U = U

                # step 2: optimize V
                phase = phase_V
                for i in indices:
                    phase[..., i] += lr
                    loss_pos = obj_fn(model(x, phase_V=phase, U=U), target)
                    total_step += 1
                    ascent = loss_pos >= loss
                    phase[..., i] -= ascent.float().mul(2 * lr)
                    loss_neg = obj_fn(model(x, phase_V=phase, U=U), target)
                    total_step += 1
                    loss = torch.where(ascent, loss_neg, loss_pos)

                phase_V = phase
                V = build_unitary(delta_list_V, phase_V, phase_bias_V, phase_V_quantizer)
                # record
                if best_record:
                    mask = best_loss > loss
                    best_loss[mask] = loss[mask]
                    best_V[mask, ...] = V[mask, ...]
                else:
                    best_loss = loss
                    best_V = V

                epoch_list += 1
                global_epoch += 1

                if adaptive:
                    restart(loss, epoch_list, phase_U, phase_V)
                    loss = obj_fn(model(x, phase_V=phase_V, phase_U=phase_U), target)

                lr = lr_scheduler(epoch_list)

                if global_epoch % 20 == 0 and verbose:
                    metric_U = criterion(best_U)
                    metric_V = criterion(best_V)
                    metric_U_avg, metric_U_std, metric_U_min, metric_U_max = (
                        metric_U.mean().item(),
                        metric_U.std().item(),
                        metric_U.min().item(),
                        metric_U.max().item(),
                    )
                    metric_V_avg, metric_V_std, metric_V_min, metric_V_max = (
                        metric_V.mean().item(),
                        metric_V.std().item(),
                        metric_V.min().item(),
                        metric_V.max().item(),
                    )
                    logger.info(
                        f"Epoch: {global_epoch}\tlr: {lr.mean().item():.5f}\tloss: {loss.mean().item():.6f}\teye_loss_U: {criterion(U).mean().item():.5f}\teye_loss_V: {criterion(V).mean().item():.5f}, best_eye_loss_U: avg={metric_U_avg:.5f}, std={metric_U_std:.5f}, max={metric_U_max:.5f}, best_eye_loss_V: avg={metric_V_avg:.5f}, std={metric_V_std:.5f}, max={metric_V_max:.5f}"
                    )
                    logger.info(best_U[0].diag())
                    mlflow.log_metrics(
                        {
                            "iloss_avg": loss.data.mean().item(),
                            "iloss_std": loss.std().item(),
                            "iloss_min": loss.min().item(),
                            "iloss_max": loss.max().item(),
                            "imetric_u_avg": metric_U_avg,
                            "imetric_u_std": metric_U_std,
                            "imetric_u_min": metric_U_min,
                            "imetric_u_max": metric_U_max,
                            "imetric_v_avg": metric_V_avg,
                            "imetric_v_std": metric_V_std,
                            "imetric_v_min": metric_V_min,
                            "imetric_v_max": metric_V_max,
                            "icore_call": total_step * best_U.size(0),
                        },
                        step=total_step,
                    )

        # record pseudo-identity
        i = 0
        for layer in self.modules():
            if isinstance(layer, (MZIBlockConv2d, MZIBlockLinear)):
                p, q, k = layer.grid_dim_y, layer.grid_dim_x, layer.miniblock
                layer.I_U = torch.nn.Parameter(
                    best_U[i : i + p * q].clone().view(p, q, k, k), requires_grad=False
                )  # can be saved in checkpoint
                layer.I_V = torch.nn.Parameter(
                    best_V[i : i + p * q].clone().view(p, q, k, k), requires_grad=False
                )
                i += p * q
        if verbose:
            logger.info(
                f"Identity calibration done in {t.interval} s (Total step: {total_step}, total core call: {total_step * best_U.size(0)}). Pseudo-I is recorded."
            )

    def identity_calibration_ztp(
        self,
        lr: Optional[float] = 0.1,
        n_epochs: Optional[int] = 400,
        lr_gamma: Optional[float] = 0.99,
        lr_min: Optional[float] = 0.006,
        random_state: Optional[int] = None,
        adaptive: bool = False,
        best_record: bool = False,
        verbose=True,
    ) -> None:
        # NOTE: too greedy, bad solution quality
        lr_min = (2 * np.pi) / (2 ** self.w_bit - 1)
        if verbose:
            logger.info(
                f"Start identity calibration...\n\tlr: {lr}\tn_epochs: {n_epochs}\tlr_gamma: {lr_gamma}\tlr_min: {lr_min}"
            )
        # assume all layers share the same tensor core size! we can collect all blocks and optimize in parallel
        delta_list_U, phase_U, phase_bias_U, S, delta_list_V, phase_V, phase_bias_V = self.stack_all_params()
        delta_list_U.fill_(1)
        delta_list_V.fill_(1)
        phase_U.fill_(0)
        phase_V.fill_(0)
        phase_U_quantizer, phase_S_quantizer, phase_V_quantizer = self.stack_all_phase_quantizer()

        S = torch.linspace(0.1, 1, S.size(-1), device=self.device).unsqueeze(0)  # [1, k]
        x = 1 / S  # Sigma^-1
        target = torch.eye(S.size(-1), device=self.device).unsqueeze(0)

        decomposer = RealUnitaryDecomposerBatch(alg="clements")
        v2m = vector_to_checkerboard
        m2v = checkerboard_to_vector
        n_step = phase_U.size(-1)

        def build_unitary(delta_list, phase, phase_bias, quantizer=None):
            if quantizer is not None:
                phase = quantizer(phase)
            return decomposer.reconstruct(delta_list, v2m(phase + phase_bias))

        U = build_unitary(delta_list_U, phase_U, phase_bias_U, phase_U_quantizer)
        V = build_unitary(delta_list_V, phase_V, phase_bias_V, phase_V_quantizer)

        def model(x, phase_U=None, phase_V=None, U=None, V=None):
            if phase_U is not None:
                Ur = build_unitary(delta_list_U, phase_U, phase_bias_U, phase_U_quantizer)
            else:
                Ur = U
            if phase_V is not None:
                Vr = build_unitary(delta_list_V, phase_V, phase_bias_V, phase_V_quantizer)
            else:
                Vr = V
            # [bs*p*q, k, k]
            out = Ur.matmul(S.unsqueeze(-1).mul(Vr.mul(x.unsqueeze(-2))))

            return out

        def obj_fn(y, target):
            ### y [bs*p*q, k, k]
            ### target [1, k, k]
            out = (y - target).square_().mean(dim=(-1, -2))  # [bs*p*q]
            return out

        def criterion(W):
            ### W [bs*p*q, k, k]
            return (W.abs() - target).square().mean(dim=(-1, -2))

        init_lr = lr

        def lr_scheduler(lr, step):
            return max(init_lr * lr_gamma ** step, lr_min)

        loss = obj_fn(model(x, U=U, V=V), target)

        # record the best identity
        best_loss = loss.clone()
        best_U = U.clone()
        best_V = V.clone()
        total_step = 0

        metric_U = criterion(best_U)
        metric_V = criterion(best_V)
        metric_U_avg, metric_U_std, metric_U_min, metric_U_max = (
            metric_U.mean().item(),
            metric_U.std().item(),
            metric_U.min().item(),
            metric_U.max().item(),
        )
        metric_V_avg, metric_V_std, metric_V_min, metric_V_max = (
            metric_V.mean().item(),
            metric_V.std().item(),
            metric_V.min().item(),
            metric_V.max().item(),
        )
        logger.info(
            f"Epoch: {0}\tlr: {lr:.5f}\tloss: {loss.mean().item():.6f}\teye_loss_U: {criterion(U).mean().item():.5f}\teye_loss_V: {criterion(V).mean().item():.5f}, best_eye_loss_U: avg={metric_U_avg:.5f}, std={metric_U_std:.5f}, max={metric_U_max:.5f}, best_eye_loss_V: avg={metric_V_avg:.5f}, std={metric_V_std:.5f}, max={metric_V_max:.5f}"
        )
        logger.info(best_U[0].diag())
        mlflow.log_metrics(
            {
                "iloss_avg": loss.data.mean().item(),
                "iloss_std": loss.std().item(),
                "iloss_min": loss.min().item(),
                "iloss_max": loss.max().item(),
                "imetric_u_avg": metric_U_avg,
                "imetric_u_std": metric_U_std,
                "imetric_u_min": metric_U_min,
                "imetric_u_max": metric_U_max,
                "imetric_v_avg": metric_V_avg,
                "imetric_v_std": metric_V_std,
                "imetric_v_min": metric_V_min,
                "imetric_v_max": metric_V_max,
                "icore_call": total_step * best_U.size(0),
            },
            step=total_step,
        )

        with TimerCtx() as t:
            for epoch in range(n_epochs):
                # Step 1: optimize U
                phase = phase_U
                for i in range(n_step):
                    phase[..., i] += lr
                    loss_pos = obj_fn(model(x, phase_U=phase, V=V), target)
                    total_step += 1
                    phase[..., i] -= 2 * lr
                    loss_neg = obj_fn(model(x, phase_U=phase, V=V), target)
                    total_step += 1
                    loss, min_indices = torch.stack([loss_neg, loss, loss_pos], dim=-1).min(dim=-1)
                    # min_indices = {0, 1, 2} -> {+0, +lr, +2lr}

                    phase[..., i] += min_indices.float() * lr
                phase_U = phase
                U = build_unitary(delta_list_U, phase_U, phase_bias_U, phase_U_quantizer)

                # step 2: optimize V
                phase = phase_V
                for i in range(n_step):
                    phase[..., i] += lr
                    loss_pos = obj_fn(model(x, phase_V=phase, U=U), target)
                    total_step += 1
                    phase[..., i] -= 2 * lr
                    loss_neg = obj_fn(model(x, phase_V=phase, U=U), target)
                    total_step += 1
                    loss, min_indices = torch.stack([loss_neg, loss, loss_pos], dim=-1).min(dim=-1)
                    # min_indices = {0, 1, 2} -> {+0, +lr, +2lr}

                    phase[..., i] += min_indices.float() * lr
                phase_V = phase
                V = build_unitary(delta_list_V, phase_V, phase_bias_V, phase_V_quantizer)

                # lr = max(lr_min, lr * lr_gamma)
                lr = lr_scheduler(lr, epoch)

                mask = best_loss > loss
                best_loss[mask] = loss[mask]
                best_U[mask, ...] = U[mask, ...]
                best_V[mask, ...] = V[mask, ...]

                if verbose and (epoch % 20 == 0 or epoch == n_epochs - 1):
                    metric_U = criterion(best_U)
                    metric_V = criterion(best_V)
                    metric_U_avg, metric_U_std, metric_U_min, metric_U_max = (
                        metric_U.mean().item(),
                        metric_U.std().item(),
                        metric_U.min().item(),
                        metric_U.max().item(),
                    )
                    metric_V_avg, metric_V_std, metric_V_min, metric_V_max = (
                        metric_V.mean().item(),
                        metric_V.std().item(),
                        metric_V.min().item(),
                        metric_V.max().item(),
                    )
                    logger.info(
                        f"Epoch: {epoch}\tlr: {lr:.5f}\tloss: {loss.mean().item():.6f}\teye_loss_U: {criterion(U).mean().item():.5f}\teye_loss_V: {criterion(V).mean().item():.5f}, best_eye_loss_U: avg={metric_U_avg:.5f}, std={metric_U_std:.5f}, max={metric_U_max:.5f}, best_eye_loss_V: avg={metric_V_avg:.5f}, std={metric_V_std:.5f}, max={metric_V_max:.5f}"
                    )
                    logger.info(best_U[0].diag())
                    mlflow.log_metrics(
                        {
                            "iloss_avg": loss.data.mean().item(),
                            "iloss_std": loss.std().item(),
                            "iloss_min": loss.min().item(),
                            "iloss_max": loss.max().item(),
                            "imetric_u_avg": metric_U_avg,
                            "imetric_u_std": metric_U_std,
                            "imetric_u_min": metric_U_min,
                            "imetric_u_max": metric_U_max,
                            "imetric_v_avg": metric_V_avg,
                            "imetric_v_std": metric_V_std,
                            "imetric_v_min": metric_V_min,
                            "imetric_v_max": metric_V_max,
                            "icore_call": total_step * best_U.size(0),
                        },
                        step=total_step,
                    )

        # record pseudo-identity
        i = 0
        for layer in self.modules():
            if isinstance(layer, (MZIBlockConv2d, MZIBlockLinear)):
                p, q, k = layer.grid_dim_y, layer.grid_dim_x, layer.miniblock
                layer.I_U = torch.nn.Parameter(
                    best_U[i : i + p * q].clone().view(p, q, k, k), requires_grad=False
                )  # can be saved in checkpoint
                layer.I_V = torch.nn.Parameter(
                    best_V[i : i + p * q].clone().view(p, q, k, k), requires_grad=False
                )
                i += p * q
        if verbose:
            logger.info(
                f"Identity calibration done in {t.interval} s (Total step: {total_step}, total core call: {total_step * best_U.size(0)}). Pseudo-I is recorded."
            )

    def identity_calibration(
        self,
        alg: str = "zcd",
        lr: Optional[float] = 0.1,
        n_epochs: Optional[int] = 400,
        lr_gamma: Optional[float] = 0.99,
        lr_min: Optional[float] = 0.006,
        random_state: Optional[int] = None,
        adaptive: bool = False,
        best_record: bool = False,
        verbose=True,
    ) -> None:
        if alg == "zgd":
            self.identity_calibration_zgd(
                lr, n_epochs, lr_gamma, lr_min, random_state, adaptive, best_record, verbose
            )
        elif alg == "ztp":
            self.identity_calibration_ztp(
                lr, n_epochs, lr_gamma, lr_min, random_state, adaptive, best_record, verbose
            )
        elif alg == "zcd":
            self.identity_calibration_zcd(
                lr, n_epochs, lr_gamma, lr_min, random_state, adaptive, best_record, verbose
            )
        else:
            raise ValueError("Only support (zgd, ztp, zcd), but got alg = {alg}")

    def parallel_mapping_zgd(
        self,
        lr: Optional[float] = 0.1,
        n_epochs: Optional[int] = 400,
        lr_gamma: Optional[float] = 0.99,
        lr_min: Optional[float] = 0.006,
        best_record: bool = False,
        verbose=True,
        validate_fn: Callable = None,
        ideal_I: Optional[bool] = False,
    ) -> None:
        lr_min = (2 * np.pi) / (2 ** self.w_bit - 1)
        if verbose:
            logger.info(
                f"Start parallel mapping...\n\tlr: {lr}\tn_epochs: {n_epochs}\tlr_gamma: {lr_gamma}\tlr_min: {lr_min}"
            )

        # assume all layers share the same tensor core size! we can collect all blocks and optimize in parallel
        delta_list_U, phase_U, phase_bias_U, S, delta_list_V, phase_V, phase_bias_V = self.stack_all_params()
        target = self.stack_all_target()  # [bs, k, k]
        target_T = target.transpose(-1, -2)
        I_U, I_V = self.stack_all_identity()
        phase_U_quantizer, phase_S_quantizer, phase_V_quantizer = self.stack_all_phase_quantizer()

        decomposer = RealUnitaryDecomposerBatch(alg="clements")
        v2m = vector_to_checkerboard
        m2v = checkerboard_to_vector
        n_step = phase_U.size(-1)

        def build_unitary(delta_list, phase, phase_bias, quantizer=None):
            if quantizer is not None:
                phase = quantizer(phase)
            return decomposer.reconstruct(delta_list, v2m(phase + phase_bias))

        U = build_unitary(delta_list_U, phase_U, phase_bias_U, phase_U_quantizer)
        V = build_unitary(delta_list_V, phase_V, phase_bias_V, phase_V_quantizer)

        def model(phase_U=None, phase_V=None, U=None, S=None, V=None):
            if phase_U is not None:
                Ur = build_unitary(delta_list_U, phase_U, phase_bias_U, phase_U_quantizer)
            else:
                Ur = U
            if phase_V is not None:
                Vr = build_unitary(delta_list_V, phase_V, phase_bias_V, phase_V_quantizer)
            else:
                Vr = V
            out = Ur.matmul(S.unsqueeze(-1).mul(Vr))  # [bs*p*q, k, k]

            return out

        def obj_fn(y, target):
            ### y [bs, k, k]
            ### target [bs, k, k]
            out = (y - target).square_().mean(dim=(-1, -2))  # [bs]
            return out

        def criterion(W):
            ### W [bs*p*q, k, k]
            return (W - target).square().sum(dim=(-1, -2)) / W.flatten(1).norm(p=2, dim=-1).square()

        # commit mapped phase
        def commit(best_phase_U, best_phase_V, best_S):
            i = 0
            for layer in self.modules():
                if isinstance(layer, (MZIBlockConv2d, MZIBlockLinear)):
                    p, q = layer.grid_dim_y, layer.grid_dim_x
                    layer.phase_U.data.copy_(best_phase_U[i : i + p * q].view_as(layer.phase_U))
                    layer.phase_V.data.copy_(best_phase_V[i : i + p * q].view_as(layer.phase_V))
                    Sigma = best_S[i : i + p * q]
                    S_max = Sigma.abs().max(dim=-1, keepdim=True)[0]
                    layer.phase_S.data.copy_(Sigma.div(S_max).acos().view_as(layer.phase_S))
                    layer.S_scale.data.copy_(S_max.view_as(layer.S_scale))
                    i += p * q

        loss = obj_fn(model(U=U, S=S, V=V), target)

        # record the best phase
        best_loss = loss.clone()
        best_phase_U = phase_U.clone()
        best_phase_V = phase_V.clone()
        best_S = S.clone()

        total_step = 0
        sigma = 0.1

        def sample_perturbation(size):
            with torch.random.fork_rng():
                torch.random.manual_seed(np.random.randint(0, 1000))
                return torch.randn(size, device=self.device).mul_(sigma)

        c = 1  # / (sigma*np.sqrt(phase_U.numel()+phase_V.numel()))**2
        m = torch.zeros([2] + list(phase_U.shape), device=self.device)
        n = 3

        def validate_and_log():
            if validate_fn is not None:
                commit(best_phase_U, best_phase_V, best_S)
                accv = []
                validate_fn(accv)
                norm_err = criterion(model(phase_U=best_phase_U, S=best_S, phase_V=best_phase_V))
                mlflow.log_metrics(
                    {
                        "mloss_avg": best_loss.mean().item(),
                        "mloss_std": best_loss.std().item(),
                        "mloss_min": best_loss.min().item(),
                        "mloss_max": best_loss.max().item(),
                        "mmetric_avg": norm_err.mean().item(),
                        "mmetric_std": norm_err.std().item(),
                        "mmetric_min": norm_err.min().item(),
                        "mmetric_max": norm_err.max().item(),
                        "macc": accv[-1].item(),
                        "mcore_call": total_step * best_phase_U.size(0),
                    },
                    step=total_step,
                )

        validate_and_log()
        with TimerCtx() as t:
            for epoch in range(n_epochs):
                # Step 1: optimize U
                for _ in range(phase_U.size(-1)):
                    grad = 0
                    for _ in range(n):
                        delta_phase = sample_perturbation([2] + list(phase_U.shape))
                        phase_U_perturb = phase_U + delta_phase[0]
                        phase_V_perturb = phase_V + delta_phase[1]
                        cur_loss = obj_fn(
                            model(phase_U=phase_U_perturb, phase_V=phase_V_perturb, S=S), target
                        )
                        total_step += 1
                        loss_diff = c * (cur_loss - loss)

                        grad += 1 / n * delta_phase * loss_diff.unsqueeze(-1).unsqueeze(0)
                    m = 0.9 * m + lr * grad
                    phase_U -= m[0]
                    phase_V -= m[1]

                    U = build_unitary(delta_list_U, phase_U, phase_bias_U, phase_U_quantizer)
                    V = build_unitary(delta_list_V, phase_V, phase_bias_V, phase_V_quantizer)
                    loss = obj_fn(model(U=U, S=S, V=V), target)
                    total_step += 1

                    # record the best phase
                    if best_record:
                        mask = best_loss > loss
                        best_loss = torch.where(mask, loss, best_loss)
                        best_phase_U[mask, :] = phase_U[mask, :]
                        best_phase_V[mask, :] = phase_V[mask, :]
                    else:
                        best_loss = loss
                        best_phase_U, best_phase_V = phase_U, phase_V

                lr = max(lr_min, lr * lr_gamma)

                if verbose and (epoch % 20 == 0 or epoch == n_epochs - 1):
                    logger.info(
                        f"Epoch: {epoch}\tlr: {lr:.5f}\tloss: {loss.mean().item():.6f}\tbest_loss: {best_loss.mean().item():.6f}"
                    )
                    validate_and_log()

        # step 2: project S only at the last step and only when identity has good quality

        # diag(I_U* x V* x W* x U x I_V)
        U = build_unitary(delta_list_U, best_phase_U, phase_bias_U, phase_U_quantizer)
        V = build_unitary(delta_list_V, best_phase_V, phase_bias_V, phase_V_quantizer)
        validate_and_log()
        if ideal_I:
            S = U.mul(target.matmul(V.transpose(-1, -2))).sum(dim=-2)  # [bs, k]
        else:
            S = I_U.mul(V.matmul(target_T.matmul(U.matmul(I_V)))).sum(dim=-2)  # [bs, k]

        loss = obj_fn(model(U=U, S=S, V=V), target)
        total_step += 3
        if best_record:
            mask = best_loss > loss
            best_loss = torch.where(mask, loss, best_loss)
            best_S[mask, :] = S[mask, :]
        else:
            best_loss = loss
            best_S = S
        validate_and_log()

        commit(best_phase_U, best_phase_V, best_S)

        if verbose:
            logger.info(
                f"Parallel mapping is done in {t.interval} s (Total step: {total_step}, total core call: {total_step * best_phase_U.size(0)}). Phases are commited to the model parameters."
            )

    def parallel_mapping_zcd(
        self,
        lr: Optional[float] = 0.1,
        n_epochs: Optional[int] = 400,
        lr_gamma: Optional[float] = 0.99,
        lr_min: Optional[float] = 0.006,
        adaptive: Optional[bool] = False,
        best_record: bool = False,
        verbose=True,
        validate_fn: Callable = None,
        ideal_I: Optional[bool] = False,
    ) -> None:
        lr_min = (2 * np.pi) / (2 ** self.w_bit - 1)
        if verbose:
            logger.info(
                f"Start parallel mapping...\n\tlr: {lr}\tn_epochs: {n_epochs}\tlr_gamma: {lr_gamma}\tlr_min: {lr_min}"
            )

        # assume all layers share the same tensor core size! we can collect all blocks and optimize in parallel
        delta_list_U, phase_U, phase_bias_U, S, delta_list_V, phase_V, phase_bias_V = self.stack_all_params()
        target = self.stack_all_target()  # [bs, k, k]
        target_T = target.transpose(-1, -2)
        I_U, I_V = self.stack_all_identity()
        phase_U_quantizer, phase_S_quantizer, phase_V_quantizer = self.stack_all_phase_quantizer()

        decomposer = RealUnitaryDecomposerBatch(alg="clements")
        v2m = vector_to_checkerboard
        m2v = checkerboard_to_vector
        n_step = phase_U.size(-1)

        def build_unitary(delta_list, phase, phase_bias, quantizer=None):
            if quantizer is not None:
                phase = quantizer(phase)
            return decomposer.reconstruct(delta_list, v2m(phase + phase_bias))

        U = build_unitary(delta_list_U, phase_U, phase_bias_U, phase_U_quantizer)
        V = build_unitary(delta_list_V, phase_V, phase_bias_V, phase_V_quantizer)

        def model(phase_U=None, phase_V=None, U=None, S=None, V=None):
            if phase_U is not None:
                Ur = build_unitary(delta_list_U, phase_U, phase_bias_U, phase_U_quantizer)
            else:
                Ur = U
            if phase_V is not None:
                Vr = build_unitary(delta_list_V, phase_V, phase_bias_V, phase_V_quantizer)
            else:
                Vr = V
            out = Ur.matmul(S.unsqueeze(-1).mul(Vr))  # [bs*p*q, k, k]

            return out

        def obj_fn(y, target):
            ### y [bs, k, k]
            ### target [bs, k, k]
            out = (y - target).square_().mean(dim=(-1, -2))  # [bs]
            return out

        def criterion(W):
            ### W [bs*p*q, k, k]
            out = (W - target).square().sum(dim=(-1, -2)) / W.flatten(1).norm(p=2, dim=-1).square()
            out = torch.nan_to_num(out, nan=0, posinf=0)
            return out

        # commit mapped phase
        def commit(best_phase_U, best_phase_V, best_S):
            i = 0
            for layer in self.modules():
                if isinstance(layer, (MZIBlockConv2d, MZIBlockLinear)):
                    p, q = layer.grid_dim_y, layer.grid_dim_x
                    layer.phase_U.data.copy_(best_phase_U[i : i + p * q].view_as(layer.phase_U))
                    layer.phase_V.data.copy_(best_phase_V[i : i + p * q].view_as(layer.phase_V))
                    Sigma = best_S[i : i + p * q]
                    S_max = Sigma.abs().max(dim=-1, keepdim=True)[0]
                    layer.phase_S.data.copy_(Sigma.div(S_max).acos().view_as(layer.phase_S))
                    layer.S_scale.data.copy_(S_max.view_as(layer.S_scale))
                    i += p * q

        loss = obj_fn(model(U=U, S=S, V=V), target)

        init_lr = lr
        lr = torch.zeros_like(loss).fill_(init_lr)
        epoch_list = torch.zeros_like(loss)
        init_lr_list = lr.clone()

        def lr_scheduler(epoch_list):
            return init_lr_list.mul(lr_gamma ** epoch_list).clamp_(min=lr_min)

        restart_thres = 0.4

        def restart(criterion, epoch_list, phase_U, phase_V, restart_thres):
            mask = criterion > restart_thres
            n_restart = mask.float().sum().item()
            if n_restart > 0:
                logger.info(f"{n_restart} blocks restarted")
            epoch_list[mask] = 0
            phase_U[mask] += 0.01
            phase_V[mask] += 0.01

        # record the best phase
        best_loss = loss.clone()
        best_phase_U = phase_U.clone()
        best_phase_V = phase_V.clone()
        best_S = S.clone()

        total_step = 0
        global_epoch = 0

        def validate_and_log():
            if validate_fn is not None:
                commit(best_phase_U, best_phase_V, best_S)
                accv = []
                validate_fn(accv)
                norm_err = criterion(model(phase_U=best_phase_U, S=best_S, phase_V=best_phase_V))
                mlflow.log_metrics(
                    {
                        "mloss_avg": best_loss.mean().item(),
                        "mloss_std": best_loss.std().item(),
                        "mloss_min": best_loss.min().item(),
                        "mloss_max": best_loss.max().item(),
                        "mmetric_avg": norm_err.mean().item(),
                        "mmetric_std": norm_err.std().item(),
                        "mmetric_min": norm_err.min().item(),
                        "mmetric_max": norm_err.max().item(),
                        "macc": accv[-1].item(),
                        "mcore_call": total_step * best_phase_U.size(0),
                    },
                    step=total_step,
                )
                return norm_err
            return None

        validate_and_log()
        with TimerCtx() as t:
            while (epoch_list < n_epochs).any() == True:
                # Step 1: optimize U
                phase = phase_U
                for i in range(n_step):
                    phase[..., i] += lr
                    loss_pos = obj_fn(model(phase_U=phase, S=S, V=V), target)
                    total_step += 1
                    ascent = loss_pos >= loss
                    phase[..., i] -= ascent.float().mul(2 * lr)
                    loss_neg = obj_fn(model(phase_U=phase, S=S, V=V), target)
                    total_step += 1
                    loss = torch.where(ascent, loss_neg, loss_pos)
                phase_U = phase
                U = build_unitary(delta_list_U, phase_U, phase_bias_U, phase_U_quantizer)

                # step 2: optimize V
                phase = phase_V
                for i in range(n_step):
                    phase[..., i] += lr
                    loss_pos = obj_fn(model(phase_V=phase, U=U, S=S), target)
                    total_step += 1
                    ascent = loss_pos >= loss
                    phase[..., i] -= ascent.float().mul(2 * lr)
                    loss_neg = obj_fn(model(phase_V=phase, U=U, S=S), target)
                    total_step += 1
                    loss = torch.where(ascent, loss_neg, loss_pos)
                phase_V = phase
                V = build_unitary(delta_list_V, phase_V, phase_bias_V, phase_V_quantizer)

                # record the best phase
                if best_record:
                    mask = best_loss > loss
                    best_loss = torch.where(mask, loss, best_loss)
                    best_phase_U[mask, :] = phase_U[mask, :]
                    best_phase_V[mask, :] = phase_V[mask, :]
                else:
                    best_loss = loss
                    best_phase_U, best_phase_V = phase_U, phase_V

                global_epoch += 1
                epoch_list += 1

                lr = lr_scheduler(epoch_list)
                if adaptive:
                    restart(criterion(model(U=U, S=S, V=V)), epoch_list, phase_U, phase_V, restart_thres)
                    restart_thres *= 1.02

                if verbose and (global_epoch % 20 == 0):
                    norm_err = validate_and_log()
                    logger.info(
                        f"Epoch: {global_epoch}\tlr: {lr.mean().item():.5f}\tloss: {loss.mean().item():.6f}\tbest_loss: {best_loss.mean().item():.6f} norm_err: {norm_err.mean().item()} norm_err_max: {norm_err.max().item()}"
                    )

        # step 3: project S only at the last step and only when identity has good quality
        # diag(I_U* x V* x W* x U x I_V)
        U = build_unitary(delta_list_U, best_phase_U, phase_bias_U, phase_U_quantizer)
        V = build_unitary(delta_list_V, best_phase_V, phase_bias_V, phase_V_quantizer)
        validate_and_log()

        if ideal_I:
            S = U.mul(target.matmul(V.transpose(-1, -2))).sum(dim=-2)  # [bs, k]
        else:
            S = I_U.mul(V.matmul(target_T.matmul(U.matmul(I_V)))).sum(dim=-2)  # [bs, k]

        loss = obj_fn(model(U=U, S=S, V=V), target)
        total_step += 3
        if best_record:
            mask = best_loss > loss
            best_loss = torch.where(mask, loss, best_loss)
            best_S[mask, :] = S[mask, :]
        else:
            best_loss = loss
            best_S = S

        validate_and_log()

        commit(best_phase_U, best_phase_V, best_S)

        if verbose:
            logger.info(
                f"Parallel mapping is done in {t.interval} s (Total step: {total_step}, total core call: {total_step * best_phase_U.size(0)}). Phases are commited to the model parameters."
            )

    def parallel_mapping_ztp(
        self,
        lr: Optional[float] = 0.1,
        n_epochs: Optional[int] = 400,
        lr_gamma: Optional[float] = 0.99,
        lr_min: Optional[float] = 0.006,
        adaptive: bool = False,
        best_record: bool = False,
        verbose=True,
        validate_fn: Callable = None,
        ideal_I: Optional[bool] = False,
    ) -> None:
        lr_min = (2 * np.pi) / (2 ** self.w_bit - 1)
        if verbose:
            logger.info(
                f"Start parallel mapping...\n\tlr: {lr}\tn_epochs: {n_epochs}\tlr_gamma: {lr_gamma}\tlr_min: {lr_min}"
            )

        # assume all layers share the same tensor core size! we can collect all blocks and optimize in parallel
        delta_list_U, phase_U, phase_bias_U, S, delta_list_V, phase_V, phase_bias_V = self.stack_all_params()
        target = self.stack_all_target()  # [bs, k, k]
        target_T = target.transpose(-1, -2)
        I_U, I_V = self.stack_all_identity()
        phase_U_quantizer, phase_S_quantizer, phase_V_quantizer = self.stack_all_phase_quantizer()

        decomposer = RealUnitaryDecomposerBatch(alg="clements")
        v2m = vector_to_checkerboard
        m2v = checkerboard_to_vector
        n_step = phase_U.size(-1)

        def build_unitary(delta_list, phase, phase_bias, quantizer=None):
            if quantizer is not None:
                phase = quantizer(phase)
            return decomposer.reconstruct(delta_list, v2m(phase + phase_bias))

        U = build_unitary(delta_list_U, phase_U, phase_bias_U, phase_U_quantizer)
        V = build_unitary(delta_list_V, phase_V, phase_bias_V, phase_V_quantizer)

        def model(phase_U=None, phase_V=None, U=None, S=None, V=None):
            if phase_U is not None:
                Ur = build_unitary(delta_list_U, phase_U, phase_bias_U, phase_U_quantizer)
            else:
                Ur = U
            if phase_V is not None:
                Vr = build_unitary(delta_list_V, phase_V, phase_bias_V, phase_V_quantizer)
            else:
                Vr = V
            out = Ur.matmul(S.unsqueeze(-1).mul(Vr))  # [bs*p*q, k, k]

            return out

        def obj_fn(y, target):
            ### y [bs, k, k]
            ### target [bs, k, k]
            out = (y - target).square_().mean(dim=(-1, -2))  # [bs]
            return out

        def criterion(W):
            ### W [bs*p*q, k, k]
            return (W - target).square().sum(dim=(-1, -2)) / W.flatten(1).norm(p=2, dim=-1).square()

        # commit mapped phase
        def commit(best_phase_U, best_phase_V, best_S):
            i = 0
            for layer in self.modules():
                if isinstance(layer, (MZIBlockConv2d, MZIBlockLinear)):
                    p, q = layer.grid_dim_y, layer.grid_dim_x
                    layer.phase_U.data.copy_(best_phase_U[i : i + p * q].view_as(layer.phase_U))
                    layer.phase_V.data.copy_(best_phase_V[i : i + p * q].view_as(layer.phase_V))
                    Sigma = best_S[i : i + p * q]
                    S_max = Sigma.abs().max(dim=-1, keepdim=True)[0]
                    layer.phase_S.data.copy_(Sigma.div(S_max).acos().view_as(layer.phase_S))
                    layer.S_scale.data.copy_(S_max.view_as(layer.S_scale))
                    i += p * q

        loss = obj_fn(model(U=U, S=S, V=V), target)

        init_lr = lr
        lr = torch.zeros_like(loss).fill_(init_lr)
        epoch_list = torch.zeros_like(loss)
        init_lr_list = lr.clone()

        def lr_scheduler(epoch_list):
            return init_lr_list.mul(lr_gamma ** epoch_list).clamp_(min=lr_min)

        def restart(criterion, epoch_list, phase_U, phase_V):
            mask = criterion > 0.4
            n_restart = mask.float().sum().item()
            if n_restart > 0:
                logger.info(f"{n_restart} blocks restarted")
            epoch_list[mask] = 0
            phase_U[mask] += 0.01
            phase_V[mask] += 0.01

        # record the best phase
        best_loss = loss.clone()
        best_phase_U = phase_U.clone()
        best_phase_V = phase_V.clone()
        best_S = S.clone()

        total_step = 0
        global_epoch = 0

        def validate_and_log():
            if validate_fn is not None:
                commit(best_phase_U, best_phase_V, best_S)
                accv = []
                validate_fn(accv)
                norm_err = criterion(model(phase_U=best_phase_U, S=best_S, phase_V=best_phase_V))
                mlflow.log_metrics(
                    {
                        "mloss_avg": best_loss.mean().item(),
                        "mloss_std": best_loss.std().item(),
                        "mloss_min": best_loss.min().item(),
                        "mloss_max": best_loss.max().item(),
                        "mmetric_avg": norm_err.mean().item(),
                        "mmetric_std": norm_err.std().item(),
                        "mmetric_min": norm_err.min().item(),
                        "mmetric_max": norm_err.max().item(),
                        "macc": accv[-1].item(),
                        "mcore_call": total_step * best_phase_U.size(0),
                    },
                    step=total_step,
                )
                return norm_err
            return None

        validate_and_log()
        with TimerCtx() as t:
            while (epoch_list < n_epochs).any() == True:
                # Step 1: optimize U
                phase = phase_U
                for i in range(n_step):
                    phase[..., i] += lr
                    loss_pos = obj_fn(model(phase_U=phase, S=S, V=V), target)
                    total_step += 1
                    phase[..., i] -= 2 * lr
                    loss_neg = obj_fn(model(phase_U=phase, S=S, V=V), target)
                    total_step += 1
                    loss, min_indices = torch.stack([loss_neg, loss, loss_pos], dim=-1).min(dim=-1)
                    # min_indices = {0, 1, 2} -> {+0, +lr, +2lr}
                    phase[..., i] += min_indices.float() * lr

                phase_U = phase
                U = build_unitary(delta_list_U, phase_U, phase_bias_U, phase_U_quantizer)

                # step 2: optimize V
                phase = phase_V
                for i in range(n_step):
                    phase[..., i] += lr
                    loss_pos = obj_fn(model(phase_V=phase, S=S, U=U), target)
                    total_step += 1
                    phase[..., i] -= 2 * lr
                    loss_neg = obj_fn(model(phase_V=phase, S=S, U=U), target)
                    total_step += 1
                    loss, min_indices = torch.stack([loss_neg, loss, loss_pos], dim=-1).min(dim=-1)
                    # min_indices = {0, 1, 2} -> {+0, +lr, +2lr}
                    phase[..., i] += min_indices.float() * lr
                phase_V = phase
                V = build_unitary(delta_list_V, phase_V, phase_bias_V, phase_V_quantizer)

                # record the best phase
                if best_record:
                    mask = best_loss > loss
                    best_loss = torch.where(mask, loss, best_loss)
                    best_phase_U[mask, :] = phase_U[mask, :]
                    best_phase_V[mask, :] = phase_V[mask, :]
                else:
                    best_loss = loss
                    best_phase_U, best_phase_V = phase_U, phase_V

                global_epoch += 1
                epoch_list += 1
                lr = lr_scheduler(epoch_list)
                if adaptive:
                    restart(criterion(model(U=U, S=S, V=V)), epoch_list, phase_U, phase_V)
                if verbose and (global_epoch % 20 == 0):
                    norm_err = validate_and_log()
                    logger.info(
                        f"Epoch: {global_epoch}\tlr: {lr.mean().item():.5f}\tloss: {loss.mean().item():.6f}\tbest_loss: {best_loss.mean().item():.6f} norm_err: {norm_err.mean().item()} norm_err_max: {norm_err.max().item()}"
                    )

        # step 3: project S only at the last step and only when identity has good quality

        # diag(I_U* x V* x W* x U x I_V)
        U = build_unitary(delta_list_U, best_phase_U, phase_bias_U, phase_U_quantizer)
        V = build_unitary(delta_list_V, best_phase_V, phase_bias_V, phase_V_quantizer)
        validate_and_log()
        if ideal_I:
            S = U.mul(target.matmul(V.transpose(-1, -2))).sum(dim=-2)  # [bs, k]
        else:
            S = I_U.mul(V.matmul(target_T.matmul(U.matmul(I_V)))).sum(dim=-2)  # [bs, k]

        loss = obj_fn(model(U=U, S=S, V=V), target)
        total_step += 3
        if best_record:
            mask = best_loss > loss
            best_loss = torch.where(mask, loss, best_loss)
            best_S[mask, :] = S[mask, :]
        else:
            best_loss = loss
            best_S = S
        validate_and_log()
        commit(best_phase_U, best_phase_V, best_S)

        if verbose:
            logger.info(
                f"Parallel mapping is done in {t.interval} s (Total step: {total_step}, total core call: {total_step * best_phase_U.size(0)}). Phases are commited to the model parameters."
            )

    def parallel_mapping(
        self,
        alg: str = "zcd",
        lr: Optional[float] = 0.1,
        n_epochs: Optional[int] = 400,
        lr_gamma: Optional[float] = 0.99,
        lr_min: Optional[float] = 0.006,
        adaptive: Optional[bool] = False,
        best_record: bool = False,
        verbose=True,
        validate_fn: Callable = None,
        ideal_I: Optional[bool] = False,
    ) -> None:
        if alg == "zgd":
            self.parallel_mapping_zgd(
                lr, n_epochs, lr_gamma, lr_min, best_record, verbose, validate_fn, ideal_I
            )
        elif alg == "ztp":
            self.parallel_mapping_ztp(
                lr, n_epochs, lr_gamma, lr_min, adaptive, best_record, verbose, validate_fn, ideal_I
            )
        elif alg == "zcd":
            self.parallel_mapping_zcd(
                lr, n_epochs, lr_gamma, lr_min, adaptive, best_record, verbose, validate_fn, ideal_I
            )
        else:
            raise ValueError("Only support (zgd, ztp, zcd), but got alg = {alg}")

    def reset_learning_profiling(self):
        for m in self.modules():
            if isinstance(m, (MZIBlockConv2d, MZIBlockLinear)):
                m.profiler.enable()
                m.profiler.reset()

    def get_learning_profiling(
        self, flat: Optional[bool] = False, input_size: Tuple = (32, 3, 32, 32)
    ) -> Dict[str, Union[float, Dict]]:
        report = {
            "total": {"core_call": 0, "accum_step": 0},
            "breakdown": {
                "forward_core_call": 0,
                "forward_accum_step": 0,  # addition only
                "backward_weight_core_call": 0,
                "backward_weight_accum_step": 0,  # addition and multiplication, doubles the cost
                "backward_input_core_call": 0,
                "backward_input_accum_step": 0,  # addition only
            },
            "act_mem": {"total": 0, "total_sparse": 0, "sparsity": 0},
        }
        breakdown = report["breakdown"] = sum(
            m.profiler for m in self.modules() if isinstance(m, (MZIBlockConv2d, MZIBlockLinear))
        ).report
        total = report["total"]
        for k in total:
            total[k] = sum(breakdown[i] for i in breakdown if k in i)
        self.profiling_report = report

        # count activation memory of pooling and BN
        bs, inc, h, w = input_size
        mem = {}
        mem_sparse = {}
        hooks = []

        def hook_fn(m, i, o):
            if isinstance(m, MZIBlockConv2d):
                sparsity = 1 - m.bp_input_sampler.spatial_sparsity
                mem[m] = i[0].numel()
                mem_sparse[m] = i[0].numel() * sparsity
            elif isinstance(m, MZIBlockLinear):
                sparsity = 1 - m.bp_input_sampler.sparsity
                mem[m] = i[0].numel()
                mem_sparse[m] = i[0].numel() * sparsity
            elif isinstance(m, nn.MaxPool2d):
                mem[m] = o.numel() * np.log2(m.kernel_size * m.kernel_size)
                mem_sparse[m] = o.numel() * np.log2(m.kernel_size * m.kernel_size)

            else:
                mem[m] = i[0].numel()
                mem_sparse[m] = i[0].numel()

        for name, layer in self.named_modules():
            # If it is a sequential, don't register a hook on it
            # but recursively register hook on all it's module children
            if isinstance(layer, (MZIBlockConv2d, MZIBlockLinear, nn.MaxPool2d, BatchNorm2d)):
                hook = layer.register_forward_hook(hook_fn)
                hooks.append(hook)
        with torch.no_grad():
            self.forward(torch.randn(input_size, device=self.device))
        for hook in hooks:
            hook.remove()
        # Just to check whether we got all layers
        all_mem = sum(mem.values())
        all_mem_sparse = sum(mem_sparse.values())
        report["act_mem"]["total"] = all_mem
        report["act_mem"]["total_sparse"] = all_mem_sparse
        report["act_mem"]["sparsity"] = 1 - all_mem_sparse / all_mem

        if flat:
            report = {f"{i}_{j}": report[i][j] for i in report for j in report[i]}
        return report
