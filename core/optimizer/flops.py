"""
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2021-10-24 16:25:58
LastEditors: Jiaqi Gu (jqgu@utexas.edu)
LastEditTime: 2021-10-24 16:25:59
"""
from typing import Callable

import numpy as np
import torch
from core.models.layers.custom_conv2d import MZIBlockConv2d
from core.models.layers.custom_linear import MZIBlockLinear
from pyutils.general import logger
from pyutils.torch_train import get_learning_rate
from torch import nn
from torch.functional import Tensor
from torch.optim import Optimizer
from torchonn.op.mzi_op import RealUnitaryDecomposerBatch, checkerboard_to_vector, vector_to_checkerboard

__all__ = ["FLOPSOptimizer"]


class FLOPSOptimizer(Optimizer):
    def __init__(
        self,
        model: nn.Module,
        lr: float = 2,
        sigma: float = 0.1,
        n_sample: int = 20,
        criterion: Callable = None,
        random_state: int = None,
    ):
        defaults = dict(lr=lr)
        super().__init__(model.parameters(), defaults)
        self.sigma = sigma
        self.n_sample = n_sample
        self.forward_counter = 0
        self.global_step = 0
        self.model = model
        self.model.switch_mode_to("usv")
        self.random_state = random_state
        self.criterion = criterion
        self.init_state()

    def init_state(self):
        self.model.sync_parameters(src="usv")
        self.modules = self.extract_modules(self.model)
        self.trainable_params = self.extract_trainable_parameters(self.model)
        self.untrainable_params = self.extract_untrainable_parameters(self.model)
        self.quantizers = self.extract_quantizer(self.model)
        self.decomposer = RealUnitaryDecomposerBatch(alg="clements")
        self.m2v = checkerboard_to_vector
        self.v2m = vector_to_checkerboard

    def extract_modules(self, model):
        return {
            layer_name: layer
            for layer_name, layer in model.named_modules()
            if isinstance(layer, (MZIBlockLinear, MZIBlockConv2d))
        }

    def extract_trainable_parameters(self, model):
        # always flatten the parameters
        return {
            layer_name: {
                param_name: getattr(layer, param_name).view(-1)
                for param_name in ["phase_U", "phase_S", "phase_V"]
            }
            for layer_name, layer in model.named_modules()
            if isinstance(layer, (MZIBlockLinear, MZIBlockConv2d))
        }

    def extract_untrainable_parameters(self, model):
        return {
            layer_name: {
                param_name: getattr(layer, param_name)
                for param_name in ["phase_bias_U", "phase_bias_V", "delta_list_U", "delta_list_V"]
            }
            for layer_name, layer in model.named_modules()
            if isinstance(layer, (MZIBlockLinear, MZIBlockConv2d))
        }

    def extract_quantizer(self, model):
        return {
            layer_name: {
                param_name: getattr(layer, param_name)
                for param_name in ["phase_U_quantizer", "phase_V_quantizer"]
            }
            for layer_name, layer in model.named_modules()
            if isinstance(layer, (MZIBlockLinear, MZIBlockConv2d))
        }

    def _sample_perturbation(self, x, sigma):
        with torch.random.fork_rng():
            torch.random.manual_seed(np.random.randint(0, 1000))
            return torch.randn(x.size(), device=x.device).mul_(sigma)

    def perturb(self, params, sigma):
        perturbs = {
            layer_name: {p_name: self._sample_perturbation(p, sigma) for p_name, p in layer_params.items()}
            for layer_name, layer_params in params.items()
        }
        perturbed_params = {
            layer_name: {p_name: p + perturbs[layer_name][p_name] for p_name, p in layer_params.items()}
            for layer_name, layer_params in params.items()
        }
        return perturbs, perturbed_params

    def commit(self, params) -> None:
        # layer = self.modules[layer_name]
        for layer_name, layer in self.modules.items():
            for param_name, phases in params[layer_name].items():
                if param_name == "phase_U":
                    phase_bias = self.untrainable_params[layer_name]["phase_bias_U"]
                    delta_list = self.untrainable_params[layer_name]["delta_list_U"]
                    quantizer = self.quantizers[layer_name]["phase_U_quantizer"]
                    layer.U.data.copy_(
                        self.decomposer.reconstruct(
                            delta_list,
                            self.v2m(
                                quantizer(phases.view(phase_bias.size(0), phase_bias.size(1), -1))
                                + phase_bias
                            ),
                        )
                    )
                elif param_name == "phase_V":
                    phase_bias = self.untrainable_params[layer_name]["phase_bias_V"]
                    delta_list = self.untrainable_params[layer_name]["delta_list_V"]
                    quantizer = self.quantizers[layer_name]["phase_V_quantizer"]

                    layer.V.data.copy_(
                        self.decomposer.reconstruct(
                            delta_list,
                            self.v2m(
                                quantizer(phases.view(phase_bias.size(0), phase_bias.size(1), -1))
                                + phase_bias
                            ),
                        )
                    )

                elif param_name == "phase_S":
                    layer.S.data.copy_(phases.data.cos().view_as(layer.S).mul_(layer.S_scale))
                else:
                    raise ValueError(f"Wrong param_name {param_name}")

    def _compute_gradient(self, loss_diff, perturb, sigma):
        c = 1 / sigma ** 2
        loss_diff = c * loss_diff
        return {
            layer_name: {p_name: p.mul(loss_diff) for p_name, p in layer_params.items()}
            for layer_name, layer_params in perturb.items()
        }

    def _accumulate_gradient(self, buffer, grad, scale):
        return {
            layer_name: {
                p_name: p.add_(grad[layer_name][p_name] * scale) for p_name, p in layer_params.items()
            }
            for layer_name, layer_params in buffer.items()
        }

    def _apply_gradients(self, params, grad, lr):
        return {
            layer_name: {p_name: p.sub_(grad[layer_name][p_name] * lr) for p_name, p in layer_params.items()}
            for layer_name, layer_params in params.items()
        }

    def zo_gradient_descent(self, obj_fn, params):
        """
        description: stochastic zo gradient descent.
        """
        # evaluate objective on the current parameters
        with torch.no_grad():
            y, old_loss = obj_fn()
            self.forward_counter += 1
        lr = get_learning_rate(self)
        grads = None
        for _ in range(self.n_sample):
            perturb, perturbed_params = self.perturb(params, self.sigma)
            with torch.no_grad():  # training=True to enable profiling, but do not save graph
                self.commit(perturbed_params)
                _, new_loss = obj_fn()
                self.forward_counter += 1
            grad = self._compute_gradient(new_loss - old_loss, perturb, self.sigma)
            grads = grad if grads is None else self._accumulate_gradient(grads, grad, 1 / self.n_sample)
        self._apply_gradients(params, grads, lr)

        return y, old_loss

    def build_obj_fn(self, data, target, model, criterion):
        def _obj_fn():
            y = model(data)
            return y, criterion(y, target)

        return _obj_fn

    def step(self, data, target):
        self.obj_fn = self.build_obj_fn(data, target, self.model, self.criterion)
        y, loss = self.zo_gradient_descent(self.obj_fn, self.trainable_params)
        # update internal parameters
        self.global_step += 1
        return y, loss
