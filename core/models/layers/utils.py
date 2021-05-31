import os
import sys
from typing import Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import cm
from pyutils.compute import gen_boolean_mask
from pyutils.general import logger
from pyutils.mzi_op import (checkerboard_to_vector, upper_triangle_to_vector,
                            vector_to_checkerboard, vector_to_upper_triangle)
from pyutils.torch_train import set_torch_deterministic
from torch import Tensor
from torch.types import Device, _size

sys.path.insert(0, os.path.join(os.path.dirname(
    os.path.abspath(__file__)), '../../..'))


__all__ = ["PhaseQuantizer", "LinearFeatureSampler",
           "Conv2dFeatureSampler", "FeedbackSampler", "SingularValueGradientSampler", "LearningProfiler"]


class PhaseQuantizer(torch.nn.Module):
    __mode_list__ = {"rectangle", "triangle", "diagonal"}

    def __init__(self,
                 bit: int,
                 v_pi: float = 4.36,
                 v_max: float = 10.8,
                 gamma_noise_std: float = 0.0,
                 crosstalk_factor: float = 0.0,
                 crosstalk_filter_size: int = 5,
                 random_state: Optional[int] = None,
                 mode: str = "rectangle",
                 device: torch.device = torch.device("cuda")) -> None:
        """2021/04/01: Uniform phase-space quantization. Support gamma noise and thermal crosstalk simulation
        Args:
            bit (int): bitwidth
            phase_onise_std (float, optional): std dev of Gaussian phase noise. Defaults to 0.
            random_state (None or int, optional): random_state for noise injection. Defaults to None.
            device (torch.Device, optional): torch.Device. Defaults to torch.device("cuda").
        """
        super().__init__()
        self.bit = bit
        self.v_pi = v_pi
        self.v_max = v_max
        self.gamma = np.pi / v_pi**2
        self.gamma_noise_std = gamma_noise_std
        self.crosstalk_factor = crosstalk_factor
        self.crosstalk_filter_size = crosstalk_filter_size
        self.random_state = random_state
        self.mode = mode
        assert mode in self.__mode_list__, logger.error(
            f"Only support {self.__mode_list__}, but got {mode}.")
        self.device = device

        self.crosstal_simulator = ThermalCrosstalkSimulator(
            plotting=False, filter_size=crosstalk_filter_size, crosstalk_factor=crosstalk_factor, device=self.device)
        self.register_buffer("noisy_gamma", None)  # can be saved in checkpoint

    def set_gamma_noise(self, noise_std: float, size: _size, random_state: Optional[int] = None):
        self.gamma_noise_std = noise_std
        self.random_state = random_state
        if(random_state is not None):
            set_torch_deterministic(random_state)
        self.noisy_gamma = torch.nn.init.trunc_normal_(torch.zeros(
            size, device=self.device)).mul_(noise_std).add_(self.gamma)

    def set_crosstalk_factor(self, crosstalk_factor):
        self.crosstalk_factor = crosstalk_factor
        self.crosstal_simulator.set_crosstalk_factor(crosstalk_factor)

    def set_bitwidth(self, bit: int) -> None:
        self.bit = bit

    def forward(self, x):
        x = x % (2*np.pi)
        if(self.bit < 16):
            if(self.mode in {"rectangle", "triangle"}):  # [0, 2pi] quantize
                ratio = 2 * np.pi / (2**self.bit - 1)
                x.div_(ratio).round_().mul_(ratio)
            elif(self.mode in {"diagonal"}):  # [0, pi] quantize
                x = torch.where(x > np.pi, 2 * np.pi - x, x)
                ratio = np.pi / (2**self.bit - 1)
                x.div_(ratio).round_().mul_(ratio)
            else:
                raise NotImplementedError(self.mode)

        if(self.noisy_gamma is not None):
            x.mul_(self.noisy_gamma.div(self.gamma))

        if(self.crosstalk_factor > 1e-5):
            x = self.crosstal_simulator.simple_simulate(
                x, mixedtraining_mask=None, mode=self.mode)

        return x


class ThermalCrosstalkSimulator(object):
    __mode_list__ = {"rectangle", "triangle", "diagonal"}

    def __init__(self,
                 # interval bet/ heat source (um)
                 heat_source_interval: float = 8.0,
                 # SetPad=0,
                 grid_precision: float = 10.0,  # um
                 power_density_multipier: float = 1e-3,
                 # W/(um K) thermal conductivity
                 thermal_conductivity: float = 1.4e-6,
                 max_iter: int = 2000,      # max # of iterations
                 # material options
                 boundary_cond: bool = False,
                 # plotting options
                 plotting: bool = True,
                 display_iter: int = 10,
                 hold_time: float = 0.00001,
                 filter_size: int = 3,
                 crosstalk_factor: float = 0.01,
                 device: Device = torch.device("cuda:0")):
        super().__init__()

        self.heat_source_interval = heat_source_interval
        self.grid_precision = grid_precision
        self.power_density_multiplier = power_density_multipier
        self.thermal_conductivity = thermal_conductivity
        self.max_iter = max_iter
        self.boundary_cond = boundary_cond
        self.plotting = plotting
        self.display_iter = display_iter
        self.hold_time = hold_time
        self.filter_size = filter_size
        self.crosstalk_factor = crosstalk_factor
        self.device = device
        self.power_density = None

        # self.init_phase_distribution(self.phases)
        self.init_filter(filter_size, crosstalk_factor)
        self.mixedtraining_mask = None

    def init_filter(self, filter_size: int, crosstalk_factor: float) -> None:
        c = crosstalk_factor
        if(filter_size == 3):
            self.filter = torch.tensor(
                [[0, c, 0],
                 [c, 1, c],
                 [0, c, 0]], device=self.device)
        elif(filter_size == 5):
            self.filter = torch.tensor(
                [[0, c, 0],
                 [c, 0, c],
                 [0, 1, 0],
                 [c, 0, c],
                 [0, c, 0]], device=self.device)
        else:
            raise ValueError(
                f"Does not support filter sizes other than 3 or 5, but got {filter_size}")
        self.filter.unsqueeze_(0).unsqueeze_(0)

        self.filter_zero_center = self.filter.clone()
        self.filter_zero_center[0, 0,
                                self.filter.size(-2)//2, self.filter.size(-1)//2] = 0

    def init_phase_distribution(self, phases: Tensor, dim: int) -> None:
        self.power_density = np.zeros(
            [self.heat_source_interval * dim, self.heat_source_interval * dim])
        cnt = 0
        # for i in range(1, dim):
        #     for j in range(1, dim - i + 1):
        #         self.power_density[self.heat_source_interval*i, self.heat_source_interval*j] = phases[cnt]
        #         cnt = cnt + 1
        pointer = 0
        for i in range(1, dim):
            number_of_sources = dim - i
            interval = self.heat_source_interval
            self.power_density[interval * i, interval:number_of_sources *
                               interval+1:interval] = phases[pointer:pointer+number_of_sources]
            pointer += number_of_sources

    def simulate(self, phases: Tensor, dim: int) -> None:
        self.init_phase_distribution(phases, dim)
        # *SetSpace      # number of steps in x
        nx = self.power_density.shape[0]
        ny = self.power_density.shape[1]  # *SetSpace   # number of steps in y
        dx = self.grid_precision  # nx/(nx-1) # width of step
        dy = self.grid_precision  # ny/(ny-1) # width of step

        # Initial Conditions
        p = torch.zeros((1, 1, nx, ny)).float().to(self.device)
        power_density = (torch.from_numpy(self.power_density.copy()).unsqueeze(0).unsqueeze(
            0)*dx*dx*dy*dy*self.thermal_conductivity / (2*(dx*dx+dy*dy))).float().to(self.device)
        kernel = torch.from_numpy(np.array([[0,    dy*dy,     0],
                                            [dx*dx,  0, dx*dx],
                                            [0,    dy*dy,    0]], dtype=np.float32)) / (2*(dx*dx+dy*dy))
        kernel = kernel.unsqueeze(0).unsqueeze(0).to(self.device)
        mask = torch.zeros(nx, ny, dtype=torch.float32, device=self.device)
        for row in range(1, nx-2):
            mask[row, 1:ny-row-1] = 1

        conv_err = []
        if self.plotting is True:
            plt.ion()  # continuous SetPlotting
            fig = plt.figure()
            ax = fig.gca(projection='3d')
            x = np.linspace(dx/2, nx - dx/2, nx)
            y = np.linspace(dy/2, ny - dy/2, ny)  # orig no setspace
            X, Y = np.meshgrid(x, y)

        for it in range(self.max_iter+1):
            # print(f"[I] iteration: {it}")
            out = torch.nn.functional.conv2d(p, kernel, padding=(1, 1))
            out.add_(power_density).mul_(mask)

            conv_err.append((it, (out-p).abs().max().data.item()))
            p = out

            if self.plotting is True and it % (self.display_iter) == 0:
                surf = ax.plot_surface(X, Y, p.squeeze(0).squeeze(
                    0).numpy(), cmap=cm.rainbow, linewidth=0, antialiased=False)
                # ax.set_zlim(0,80)
                # ax.set_xlim(0,0.1)
                # ax.set_ylim(0,0.1)
                plt.title('it#%d' % it, y=1)
                ax.set_xlabel('Distance (x%d um)' % (self.grid_precision))
                ax.set_ylabel('Distance (x%d um)' % (self.grid_precision))
                ax.set_zlabel('Temperature (C)')
                # for tick in ax.xaxis.get_major_ticks():
                #     tick.label.set_fontsize(80)
                # for tick in ax.yaxis.get_major_ticks():
                #     tick.label.set_fontsize(80)

                plt.show()
                plt.pause(self.hold_time)

        return p.cpu().numpy().astype(np.float64)

    def set_crosstalk_factor(self, crosstalk_factor: float) -> None:
        self.crosstalk_factor = crosstalk_factor
        self.init_filter(self.filter_size, crosstalk_factor)

    def simple_simulate_triangle(self, phases: Tensor, mixedtraining_mask: Optional[Tensor]) -> Tensor:
        size = phases.size()
        phases = phases % (2 * np.pi)
        if(mixedtraining_mask is None):
            # batchify phases [bs, k(k-1)/2]
            phases = phases.view(-1, 1, phases.size(-1))  # [bs, 1, k(k-1)/2]
            phases = vector_to_checkerboard(phases)
            filter = self.filter
            padding1, padding2 = self.filter.size(
                -2) // 2, self.filter.size(-1) // 2
            phases = torch.nn.functional.conv2d(
                phases, filter, padding=(padding1, padding2))
            phases = checkerboard_to_vector(phases)
            phases = phases.view(size)
        else:
            # only active devices marked as 1/True in the mixed training mask will influcence others
            # passive devices will be influenced by active devices, but will not incluence others
            # batchify phases [bs, k(k-1)/2]
            phase_mat_active = vector_to_upper_triangle(phases.mul(
                mixedtraining_mask.float()).view(-1, 1, phases.size(-1)))
            phases = phases.view(-1, 1, phases.size(-1))  # [bs, 1, k(k-1)/2]
            filter = self.filter_zero_center
            padding1, padding2 = self.filter.size(
                -2) // 2, self.filter.size(-1) // 2
            # influence map
            phase_mat_active = torch.nn.functional.conv2d(
                phase_mat_active, filter, padding=(padding1, padding2))
            # add influence map and original phases together
            phases = upper_triangle_to_vector(phase_mat_active) + phases
            phases = phases.view(size)

        return phases

    def simple_simulate_diagonal(self, phases: Tensor, mixedtraining_mask: Optional[Tensor]) -> Tensor:
        return phases

    def simple_simulate_butterfly(self, phases: Tensor, mixedtraining_mask: Optional[Tensor]) -> Tensor:
        phases = phases % (2 * np.pi)
        ## [n_level, k/2, 2]
        size = phases.size()

        if(mixedtraining_mask is None):
            # [1, 1, n_level, k]
            phases = phases.view([1, 1] + list(size)
                                 [:-2] + [phases.size(-1)*phases.size(-2)])
            filter = self.filter
            padding = self.filter_size // 2
            phases = torch.nn.functional.conv2d(
                phases, filter, padding=(padding, padding))
            phases = phases.view(size)

        else:
            # only active devices marked as 1/True in the mixed training mask will influcence others
            # poassive devices will be influenced by active devices, but will not incluence others

            phases_active = phases * mixedtraining_mask.float()
            filter = self.filter_zero_center
            padding = self.filter_size // 2
            # influence map
            phases_active = torch.nn.functional.conv2d(phases_active.view(
                [1, 1] + list(size)[:-2] + [phases.size(-1)*phases.size(-2)]), filter, padding=(padding, padding))
            # add influence map and original phases together
            phases = phases_active.view_as(phases) + phases

        return phases

    def simple_simulate_rectangle(self, phases: Tensor, mixedtraining_mask: Optional[Tensor]) -> Tensor:
        size = phases.size()
        phases = phases % (2 * np.pi)
        if(mixedtraining_mask is None):
            # batchify phases [bs, k(k-1)/2]
            phases = phases.view(-1, 1, phases.size(-1))  # [bs, 1, k(k-1)/2]
            phases = vector_to_checkerboard(phases)
            filter = self.filter
            padding1, padding2 = self.filter.size(
                -2) // 2, self.filter.size(-1) // 2
            phases = torch.nn.functional.conv2d(
                phases, filter, padding=(padding1, padding2))
            phases = checkerboard_to_vector(phases)
            phases = phases.view(size)
        else:
            # only active devices marked as 1/True in the mixed training mask will influcence others
            # passive devices will be influenced by active devices, but will not incluence others
            # batchify phases [bs, k(k-1)/2]
            phase_mat_active = vector_to_upper_triangle(phases.mul(
                mixedtraining_mask.float()).view(-1, 1, phases.size(-1)))
            phases = phases.view(-1, 1, phases.size(-1))  # [bs, 1, k(k-1)/2]
            filter = self.filter_zero_center
            padding1, padding2 = self.filter.size(
                -2) // 2, self.filter.size(-1) // 2
            # influence map
            phase_mat_active = torch.nn.functional.conv2d(
                phase_mat_active, filter, padding=(padding1, padding2))
            # add influence map and original phases together
            phases = upper_triangle_to_vector(phase_mat_active) + phases
            phases = phases.view(size)

        return phases

    def simple_simulate(self, phases: Tensor, mixedtraining_mask: Optional[Tensor] = None, mode: str = "rectangle") -> Tensor:
        assert mode in self.__mode_list__, logger.error(
            f"Only support {self.__mode_list__}. But got {mode}")
        if(mode == "triangle"):
            return self.simple_simulate_triangle(phases, mixedtraining_mask)
        elif(mode == "rectangle"):
            return self.simple_simulate_rectangle(phases, mixedtraining_mask)
        elif(mode == "diagonal"):
            return self.simple_simulate_diagonal(phases, mixedtraining_mask)
        elif(mode == "butterfly"):
            return self.simple_simulate_butterfly(phases, mixedtraining_mask)
        else:
            return phases


class LinearFeatureSampler(torch.nn.Module):
    def __init__(self, sparsity: float = 0, miniblock: int = 8, normalize: str = "none", random_state: Optional[int] = None) -> None:
        super().__init__()
        self.sparsity = sparsity
        self.miniblock = miniblock
        self.normalize = normalize
        self.random_state = random_state
        self.mask = None

    def set_sparsity(self, sparsity: float, random_state: Optional[int] = None) -> None:
        assert 0 <= sparsity <= 1, logger.error(
            f"Illegal sparsity, must within [0,1] but got {sparsity}.")
        self.sparsity = sparsity
        self.random_state = random_state

    def sample(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Block-level structured sampling of input tensor. return sampled blocks and a boolean mask
        Args:
            x (Tensor): 2D padded hidden features

        Raises:
            NotImplementedError: Not supported tensor shape

        Returns:
            Tuple[Tensor, Tensor]: sampled blocks and boolean mask
        """

        # padded 2D input for Linear layer [bs, inc] = [bs, p*k]
        # samples must be different for different examples
        if(not self.training):  # DO NOT sampling during inference
            return x, None
        self.input_size = x.size()
        batch_size = x.size(0)
        self.n_block = x.size(-1)//self.miniblock
        self.mask = gen_boolean_mask(
            (batch_size, self.n_block), true_prob=1-self.sparsity, device=x.device)  # [bs, p]
        x = x.view(batch_size, self.n_block, -
                   1)[self.mask, :]  # [n_samples, k]
        if(self.normalize == "exp"):  # expectation maintained (unbiased)
            x = x.mul(1 / (self.mask.float().sum()/self.mask.numel() + 1e-12))
        elif(self.normalize == "var"):  # variance maintained
            x = x.mul(1 / (self.mask.float().sum() /
                           self.mask.numel() + 1e-12).sqrt())
        return x, self.mask

    def reconstruct(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        # x [n_samples, k]
        # mask [bs, n_block]
        if(mask is None):
            mask = self.mask
        if(mask is None):
            return x
        out = torch.zeros(
            self.input_size, device=x.device).view(-1, self.n_block, self.miniblock)
        out[mask, :] = x
        out = out.flatten(1)
        return out


class Conv2dFeatureSampler(torch.nn.Module):
    def __init__(self,
                 spatial_sparsity: float = 0,
                 column_sparsity: float = 0,
                 kernel_size: int = 3,
                 stride: int = 1,
                 padding: int = 1,
                 normalize: str = "none",
                 random_state: Optional[int] = None
                 ) -> None:
        super().__init__()
        self.spatial_sparsity = spatial_sparsity
        self.column_sparsity = column_sparsity
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.normalize = normalize
        self.random_state = random_state
        self.spatial_mask = None
        self.column_mask = None
        self.set_sparsity(spatial_sparsity, column_sparsity, random_state)

    def set_sparsity(self, spatial_sparsity: float, column_sparsity: float, random_state: Optional[int] = None) -> None:
        assert 0 <= spatial_sparsity <= 1, logger.error(
            f"Illegal spatial_sparsity, must within [0,1] but got {spatial_sparsity}.")
        assert 0 <= column_sparsity <= 1, logger.error(
            f"Illegal column_sparsity, must within [0,1] but got {column_sparsity}.")
        self.spatial_sparsity = spatial_sparsity
        self.column_sparsity = column_sparsity
        self.random_state = random_state
        if(self.kernel_size == 1):  # merge column sampling to spatial sampling to save memory
            self.spatial_sparsity = 1 - \
                (1-self.spatial_sparsity)*(1-self.column_sparsity)
            self.column_sparsity = 0

    def sample(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Block-level structured sampling of input tensor. return sampled blocks and a boolean mask
        Args:
            x (Tensor): 4D feature maps

        Raises:
            NotImplementedError: Not supported tensor shape

        Returns:
            Tuple[Tensor, Tensor, Tensor]: sampled blocks, boolean spatial mask and boolean column mask
        """

        # 4D input for Conv2d layer [bs, inc, h, w]
        # share samples between different channels and examples are OK
        # unstructured spatial sampling to save memory and strcutured column sampling to save backward runtime
        if(not self.training):  # DO NOT sampling during inference
            return x, None, None, None
        self.input_size = x.size()
        h_x, w_x = x.size(2), x.size(3)
        h_out = int((h_x - self.kernel_size + 2 *
                     self.padding) / self.stride + 1)
        w_out = int((w_x - self.kernel_size + 2 *
                     self.padding) / self.stride + 1)
        self.h_out, self.w_out = h_out, w_out
        if(self.spatial_sparsity > 0):
            self.spatial_mask = gen_boolean_mask(
                (h_x, w_x), true_prob=1-self.spatial_sparsity, random_state=self.random_state, device=x.device)  # [h, w]
        else:
            self.spatial_mask = None
        if(self.column_sparsity > 0):
            self.column_mask = gen_boolean_mask(
                [h_out*w_out], true_prob=1-self.column_sparsity, random_state=self.random_state, device=x.device)  # [h_out*w_out]
        else:
            self.column_mask = None
        if(self.spatial_mask is not None):
            x = x[:, :, self.spatial_mask]
        if(self.normalize in {"exp", "var"}):
            # Do not scale feature maps. Too costly. We also need to scale based on column_scaling_factor. Equivalently, we can scale grad_weight instead.
            real_spatial_sparsity = (1-self.spatial_mask.float().sum() /
                                     self.spatial_mask.numel()) if self.spatial_mask is not None else 0
            real_column_sparsity = (1-self.column_mask.float().sum() /
                                    self.column_mask.numel()) if self.column_mask is not None else 0
            self.scaling_factor = 1 / \
                ((1 - real_spatial_sparsity) * (1 - real_column_sparsity) + 1e-12)
            if(self.normalize == "var"):
                self.scaling_factor = self.scaling_factor**0.5
        else:
            self.scaling_factor = None

        return x, self.spatial_mask, self.column_mask, self.scaling_factor

    def reconstruct(self, x: Tensor, spatial_mask: Optional[Tensor] = None) -> Tensor:
        # reconstruct using the spatial mask
        if(spatial_mask is None):
            spatial_mask = self.spatial_mask
        if(spatial_mask is not None):
            out = torch.zeros(self.input_size, device=x.device)
            out[:, :, spatial_mask] = x
        else:
            out = x
        return out


class FeedbackSampler(torch.nn.Module):
    __alg_list__ = {"topk", "uniform", "gtopk"}
    __mode_list__ = {"linear", "conv"}

    def __init__(self, forward_sparsity: float, backward_sparsity: float, alg: str = "topk", normalize: str = "none", mode: str = "linear", random_state: Optional[int] = None):
        super().__init__()
        self.forward_sparsity = forward_sparsity
        self.backward_sparsity = backward_sparsity
        self.alg = alg
        assert alg in self.__alg_list__, logger.error(
            f"Only support {self.__alg_list__}, but got {alg}.")
        self.normalize = normalize
        self.mode = mode
        assert mode in self.__mode_list__, logger.error(
            f"Only support {self.__mode_list__}, but got {mode}.")

        self.random_state = random_state
        self.mask = None

    def set_sparsity(self, forward_sparsity: float, backward_sparsity: float, random_state: Optional[int] = None) -> None:
        assert 0 <= forward_sparsity <= 1, logger.error(
            f"Illegal forward_sparsity, must within [0,1] but got {forward_sparsity}.")
        self.forward_sparsity = forward_sparsity
        assert 0 <= backward_sparsity <= 1, logger.error(
            f"Illegal backward_sparsity, must within [0,1] but got {backward_sparsity}.")
        self.backward_sparsity = backward_sparsity
        self.random_state = random_state

    def sample_topk(self, x: Tensor, mask: Tensor, forward: bool = False) -> Tensor:
        # we prefer uniform column-wise sparsity in W, i.e., row-wise sparsity in W^T
        ## x: [p, q, k, k]
        # forward: topk along q dimension, backward: topk along p dimension
        if(mask is None):
            mask = self.mask = torch.ones(x.size(0), x.size(
                1), device=x.device, dtype=torch.bool)
        sparsity = self.forward_sparsity if forward else self.backward_sparsity
        if(sparsity < 1e-9):
            return x.clone()
        # pruned blocks has small total singular value
        norm = x.flatten(2).norm(p=2, dim=-1)  # [p, q]
        if(forward):
            thres = torch.quantile(
                norm, q=sparsity, dim=1, keepdim=True)  # forward: [p, 1]
        else:
            thres = torch.quantile(
                norm, q=sparsity, dim=0, keepdim=True)  # backward: [1, q]
        mask.masked_fill_(norm < thres, 0)
        x = x.clone()
        x[~mask, :, :] = 0
        if(self.normalize == "exp"):  # expectation maintained (unbiased)
            x = x.mul_(1 / (mask.float().sum()/mask.numel() + 1e-12))
        elif(self.normalize == "var"):  # variance maintained
            x = x.mul_(1 / (mask.float().sum()/mask.numel() + 1e-12).sqrt())

        return x

    def sample_topk_(self, x: Tensor, mask: Tensor, forward: bool = False) -> Tensor:
        # we prefer uniform column-wise sparsity in W, i.e., row-wise sparsity in W^T
        ## x: [p, q, k, k]
        if(mask is None):
            mask = self.mask = torch.ones(x.size(0), x.size(
                1), device=x.device, dtype=torch.bool)
        sparsity = self.forward_sparsity if forward else self.backward_sparsity
        if(sparsity < 1e-9):
            return x
        # pruned blocks has small total singular value
        norm = x.flatten(2).norm(p=2, dim=-1)
        if(forward):
            thres = torch.quantile(
                norm, q=sparsity, dim=1, keepdim=True)  # forward: [p, 1]
        else:
            thres = torch.quantile(
                norm, q=sparsity, dim=0, keepdim=True)  # backward: [1, q]
        mask.masked_fill_(norm < thres, 0)
        x[~mask, :, :] = 0
        if(self.normalize == "exp"):
            x = x.mul_(1 / (mask.float().sum()/mask.numel() + 1e-12))
        elif(self.normalize == "var"):
            x = x.mul_(1 / (mask.float().sum()/mask.numel() + 1e-12).sqrt())
        return x

    def sample_gtopk(self, x: Tensor, mask: Tensor, forward: bool = False) -> Tensor:
        # global top k without load balancing
        ## x: [p, q, k, k]
        if(mask is None):
            mask = self.mask = torch.ones(x.size(0), x.size(
                1), device=x.device, dtype=torch.bool)
        sparsity = self.forward_sparsity if forward else self.backward_sparsity
        if(sparsity < 1e-9):
            return x.clone()
        # pruned blocks has small total singular value
        norm = x.flatten(2).norm(p=2, dim=-1)  # [p,q]
        thres = torch.quantile(norm, q=sparsity)  # [1]
        mask.masked_fill_(norm < thres, 0)
        x = x.clone()
        x[~mask, :, :] = 0
        if(self.normalize == "exp"):
            x = x.mul_(1 / (mask.float().sum()/mask.numel() + 1e-12))
        elif(self.normalize == "var"):
            x = x.mul_(1 / (mask.float().sum()/mask.numel() + 1e-12).sqrt())
        return x

    def sample_gtopk_(self, x: Tensor, mask: Tensor, forward: bool = False) -> Tensor:
        # global top k without load balancing
        ## x: [p, q, k, k]
        if(mask is None):
            mask = self.mask = torch.ones(x.size(0), x.size(
                1), device=x.device, dtype=torch.bool)
        sparsity = self.forward_sparsity if forward else self.backward_sparsity
        if(sparsity < 1e-9):
            return x
        # pruned blocks has small total singular value
        norm = x.flatten(2).norm(p=2, dim=-1)
        thres = torch.quantile(norm, q=sparsity)
        mask.masked_fill_(norm < thres, 0)
        x[~mask, :, :] = 0
        if(self.normalize == "exp"):
            x = x.mul_(1 / (mask.float().sum()/mask.numel() + 1e-12))
        elif(self.normalize == "var"):
            x = x.mul_(1 / (mask.float().sum()/mask.numel() + 1e-12).sqrt())
        return x

    def sample_uniform(self, x: Tensor, mask: Tensor, forward: bool = False) -> Tensor:
        sparsity = self.forward_sparsity if forward else self.backward_sparsity
        if(mask is None):
            mask = self.mask = gen_boolean_mask(size=(x.size(0), x.size(
                1)), true_prob=1-sparsity, random_state=self.random_state, device=x.device)
        if(sparsity < 1e-9):
            return x.clone()
        x = x.clone()
        x[~mask, :, :] = 0
        if(self.normalize == "exp"):
            x = x.mul_(1 / (mask.float().sum()/mask.numel() + 1e-12))
        elif(self.normalize == "var"):
            x = x.mul_(1 / (mask.float().sum()/mask.numel() + 1e-12).sqrt())
        return x

    def sample_uniform_(self, x: Tensor, mask: Tensor, forward: bool = False) -> Tensor:
        sparsity = self.forward_sparsity if forward else self.backward_sparsity
        if(mask is None):
            mask = self.mask = gen_boolean_mask(size=(x.size(0), x.size(
                1)), true_prob=1-sparsity, random_state=self.random_state, device=x.device)

        if(sparsity < 1e-9):
            return x
        x[~mask, :, :] = 0
        if(self.normalize == "exp"):
            x = x.mul_(1 / (mask.float().sum()/mask.numel() + 1e-12))
        elif(self.normalize == "var"):
            x = x.mul_(1 / (mask.float().sum()/mask.numel() + 1e-12).sqrt())
        return x

    def sample(self, x: Tensor, mask: Tensor = None, forward: bool = False) -> Tensor:
        """ sample the weight matrix.

        Args:
            x (Tensor): weight matrix W [p, q, k, k]
            forward (bool): whether use forward sparsity or feedback sparsity in topk algorithm.

        Raises:
            NotImplementedError

        Returns:
            Tensor: sparse weight matrix
        """
        ## x [p, q, k, k]
        if(not self.training):  # DO NOT sampling during inference
            return x
        if(self.alg == "uniform"):
            return self.sample_uniform(x, mask, forward)
        elif(self.alg == "topk"):
            return self.sample_topk(x, mask, forward)
        elif(self.alg == "gtopk"):
            return self.sample_gtopk(x, mask, forward)
        else:
            raise NotImplementedError(
                f"Only support {self.__alg_list__}, but got {self.alg}.")

    def sample_(self, x: Tensor, mask: Tensor = None, forward: bool = False) -> Tensor:
        ## x [p, q, k, k]
        if(not self.training):  # DO NOT sampling during inference
            return x
        if(self.alg == "uniform"):
            return self.sample_uniform_(x, mask, forward)
        elif(self.alg == "topk"):
            return self.sample_topk_(x, mask, forward)
        elif(self.alg == "gtopk"):
            return self.sample_gtopk_(x, mask, forward)
        else:
            raise NotImplementedError(
                f"Only support {self.__alg_list__}, but got {self.alg}.")


class SingularValueGradientSampler(torch.nn.Module):
    __alg_list__ = {"topk", "uniform"}

    def __init__(self, rank: int, alg: str = "topk", sign: bool = False, random_state: Optional[int] = None):
        super().__init__()
        self.rank = rank
        self.alg = alg
        assert alg in self.__alg_list__, logger.error(
            f"Only support {self.__alg_list__}, but got {alg}.")
        self.sign = sign
        self.random_state = random_state
        self.mask = None

    def set_rank(self, rank: int, random_state: Optional[int] = None) -> None:
        self.rank = rank
        self.random_state = random_state

    def uniform_mask(self, x: Tensor) -> Tensor:
        ## x [p, q, k]
        if(self.rank < x.size(-1)):
            indices = torch.ones(x.size(0)*x.size(1), x.size(2), device=x.device).multinomial(
                num_samples=self.rank).view(x.size(0), x.size(1), -1)
            self.mask = torch.zeros_like(
                x, dtype=torch.bool).scatter_(-1, indices, 1)
        else:
            self.mask = None
        return self.mask

    def topk_mask(self, x: Tensor) -> Tensor:
        if(self.rank < x.size(-1)):
            indices = torch.topk(x.abs(), k=self.rank, dim=-1,
                                 largest=True, sorted=False)[1]
            self.mask = torch.zeros_like(
                x, dtype=torch.bool).scatter_(-1, indices, 1)
        else:
            self.mask = None
        return self.mask

    def sample(self, u: Tensor, s: Tensor, v: Tensor, grad_weight: Tensor, I_U: Optional[Tensor] = None, I_V: Optional[Tensor] = None, grad_scaling_factor: float = None):
        if(self.alg == "uniform"):
            mask = self.uniform_mask(s)
        elif(self.alg == "topk"):
            mask = self.topk_mask(s)
        else:
            raise NotImplementedError(
                f"Only support {self.__alg_list__}, but got {self.alg}.")
        p, q, k = s.size()
        if(I_V is not None):
            # u [p,q,k,k] x [p,q,k,k/rank] => [p,q,k,k/rank]
            if(mask is not None):
                I_V = I_V.masked_select(
                    mask.unsqueeze(-2)).view(p, q, k, self.rank)
            u = u.matmul(I_V)

        grad_sigma_by_v = u.permute([0, 1, 3, 2]).matmul(grad_weight)

        del grad_weight
        if(I_U is not None):
            if(mask is not None):
                I_U = I_U[mask].view(p, q, self.rank, k)
            v = I_U.matmul(v)

        grad_sigma = grad_sigma_by_v.mul_(v).sum(
            dim=-1)  # [p, q, k] or [p, q, bp_rank]
        if(mask is not None):
            grad_sigma = torch.zeros(
                p, q, k, device=grad_sigma.device, dtype=grad_sigma.dtype).masked_scatter_(mask, grad_sigma)
        if(grad_scaling_factor is not None):
            grad_sigma.mul_(grad_scaling_factor)
        if(self.sign):
            grad_sigma = grad_sigma.sign()
        return grad_sigma


class LearningProfiler(torch.nn.Module):
    def __init__(self, _enable: bool = True) -> None:
        super().__init__()
        self.report = None
        self._enable = _enable
        self.reset()

    def reset(self):
        self.report = {"forward_core_call": 0,
                       "forward_accum_step": 0,  # addition only
                       "backward_weight_core_call": 0,
                       "backward_weight_accum_step": 0,  # addition and multiplication, doubles the cost
                       "backward_input_core_call": 0,
                       "backward_input_accum_step": 0  # addition only
                       }

    def enable(self):
        self._enable = True

    def disable(self):
        self._enable = False

    def update_forward(self, x: Tensor, weight: Tensor, output: Tensor, feedback_sampler: FeedbackSampler) -> None:
        # important assumption:
        # p x k adders, k adders for each block row
        # k wavelength for parallel PTC forward and backward
        # x [bs, inc] or [bs, inc, h, w]
        # weight: [p, q, k, k]
        # output [bs, outc] or [bs, outc, h', w']
        p, q, k = weight.size(0), weight.size(1), weight.size(2)
        if(not self._enable or not self.training):
            return
        # forward
        if feedback_sampler.forward_sparsity > 1e-9 and feedback_sampler.mask is not None:
            active_weight_block = feedback_sampler.mask.float().sum().item()
            max_accum_step = max(
                0, feedback_sampler.mask.sum(1).max().item() - 1)
        else:
            active_weight_block = weight.size(0)*weight.size(1)
            max_accum_step = weight.size(1) - 1
        batch = x.size(0) if x.dim() == 2 else int(
            output.numel()/output.size(1))
        self.report["forward_core_call"] += active_weight_block * \
            batch  # p*q*sparsity*batch
        # do not forget we do im2col sequentially
        self.report["forward_accum_step"] += max_accum_step * batch + \
            np.ceil(batch / k)  # PTC forward also counted as  1 step/cycle

    def update_backward(self, x: Tensor, weight: Tensor, grad_output: Tensor, x_requires_grad: bool, weight_requires_grad: bool, feature_sampler: Union[LinearFeatureSampler, Conv2dFeatureSampler], feedback_sampler: FeedbackSampler, rank_sampler: SingularValueGradientSampler) -> None:
        p, q, k = weight.size(0), weight.size(1), weight.size(2)
        if(not self._enable or not self.training):
            return
        # weight backward
        if(weight_requires_grad):
            if(isinstance(feature_sampler, Conv2dFeatureSampler)):
                if(feature_sampler.kernel_size == 1):  # conv1x1: spatial sampling=column sampling
                    if(feature_sampler.spatial_sparsity > 1e-9):
                        # conv1x1 can have stride>1, padding is always 0
                        # in real X_col, the real sparsity can be obtained from the stride spatial mask
                        stride = feature_sampler.stride
                        active_column = feature_sampler.spatial_mask[::stride, ::stride].float(
                        ).sum().item() * x.size(0)
                    else:
                        active_column = int(
                            grad_output.numel() / grad_output.size(1))
                else:  # conv3x3
                    if feature_sampler.column_sparsity > 1e-9:
                        active_column = feature_sampler.column_mask.float().sum(
                        ).item() * x.size(0)  # column mask is shared between batch
                    else:
                        active_column = int(
                            grad_output.numel() / grad_output.size(1))
                self.report["backward_weight_core_call"] += weight.size(
                    0) * weight.size(1) * 2 * active_column
                # MAC doubles the cost. lowrank does not impact this.
                # self.report["backward_weight_accum_step"] += max(0,active_column - 1) * 2 * rank_sampler.rank / weight.size(-1)
                self.report["backward_weight_accum_step"] += active_column * 4
            elif(isinstance(feature_sampler, LinearFeatureSampler)):
                mask = feature_sampler.mask.float()  # [bs, q]
                # nonzero * p
                self.report["backward_weight_core_call"] += mask.sum().item() * \
                    weight.size(0)
                self.report["backward_weight_accum_step"] += mask.sum(
                    0).max().item() * x.size(0)
        # input backward
        if(x_requires_grad):
            # share sparsity with feedback or no forward weight sampling, has backward feedback sampling
            if(feedback_sampler.forward_sparsity > 1e-9 or feedback_sampler.backward_sparsity > 1e-9):
                active_weight_block = feedback_sampler.mask.float().sum().item()
                if(x.dim() == 2):  # fully-connected
                    max_accum_step = max(
                        0, feedback_sampler.mask.sum(0).max().item() - 1)  # p
                elif(x.dim() == 4 and feature_sampler.kernel_size != 1):  # CONV3x3
                    # max_accum_step = max(0, feedback_sampler.mask.sum(1).max().item() - 1)  # q ## this is wrong!!
                    max_accum_step = np.ceil(x.size(
                        1)/p)*np.ceil(np.log2(2*k))*np.ceil(feedback_sampler.mask.sum(0).max().item()/2)
                elif(x.dim() == 4 and feature_sampler.kernel_size == 1):  # conv1x1
                    max_accum_step = max(
                        0, feedback_sampler.mask.sum(0).max().item() - 1)  # p
            else:
                active_weight_block = weight.size(0) * weight.size(1)
                if(x.dim() == 2):  # fully-connected
                    max_accum_step = max(0, weight.size(0) - 1)  # p-1
                elif(x.dim() == 4 and feature_sampler.kernel_size != 1):  # CONV3x3
                    # max_accum_step = weight.size(1) - 1  # q-1 # wrong
                    max_accum_step = np.ceil(
                        x.size(1)/p)*np.ceil(np.log2(2*k))*np.ceil(p/2)
                elif(x.dim() == 4 and feature_sampler.kernel_size == 1):  # conv1x1
                    max_accum_step = max(0, weight.size(0) - 1)  # p-1
            if(x.dim() == 2): # linear
                batch = x.size(0)
            elif(x.dim() == 4 and feature_sampler.kernel_size == 1):  #CONV1x1, maybe have stride
                batch = int(grad_output.numel() / grad_output.size(1))
            elif(x.dim() == 4 and feature_sampler.kernel_size > 1):# CONV3x3 or 5X5
                batch = int(x.numel() / x.size(1))
            else:
                raise NotImplementedError
            self.report["backward_input_core_call"] += active_weight_block * batch #int(grad_output.numel()/grad_output.size(1))
            # do not forget we do matrix-matrix mul via sequential mat-vec mul
            self.report["backward_input_accum_step"] += max_accum_step * batch #int(grad_output.numel() / grad_output.size(1))

    def update(self, x: Tensor, weight: Tensor, grad_output: Tensor, x_requires_grad: bool, weight_requires_grad: bool, feature_sampler: Union[LinearFeatureSampler, Conv2dFeatureSampler], feedback_sampler: FeedbackSampler, rank_sampler: SingularValueGradientSampler) -> None:
        # important assumption:
        # p x k adders, k adders for each block row
        # k wavelength for parallel PTC forward and backward
        # x [bs, inc] or [bs, inc, h, w]
        # weight: [p, q, k, k]
        # grad_output [bs, outc] or [bs, outc, h', w']
        p, q, k = weight.size(0), weight.size(1), weight.size(2)
        if(not self._enable or not self.training):
            return
        # forward
        if feedback_sampler.forward_sparsity > 1e-9:
            active_weight_block = feedback_sampler.mask.float().sum().item()
            max_accum_step = max(
                0, feedback_sampler.mask.sum(1).max().item() - 1)
        else:
            active_weight_block = weight.size(0)*weight.size(1)
            max_accum_step = weight.size(1) - 1
        batch = x.size(0) if x.dim() == 2 else int(
            grad_output.numel()/grad_output.size(1))
        self.report["forward_core_call"] += active_weight_block * \
            batch  # p*q*sparsity*batch
        # do not forget we do im2col sequentially
        self.report["forward_accum_step"] += max_accum_step * batch + \
            np.ceil(batch / k)  # PTC forward also counted as  1 step/cycle

        # weight backward
        if(weight_requires_grad):
            if(isinstance(feature_sampler, Conv2dFeatureSampler)):
                if(feature_sampler.kernel_size == 1):  # conv1x1: spatial sampling=column sampling
                    if(feature_sampler.spatial_sparsity > 1e-9):
                        # conv1x1 can have stride>1, padding is always 0
                        # in real X_col, the real sparsity can be obtained from the stride spatial mask
                        stride = feature_sampler.stride
                        active_column = feature_sampler.spatial_mask[::stride, ::stride].float(
                        ).sum().item() * x.size(0)
                    else:
                        active_column = int(
                            grad_output.numel() / grad_output.size(1))
                else:  # conv3x3
                    if feature_sampler.column_sparsity > 1e-9:
                        active_column = feature_sampler.column_mask.float().sum(
                        ).item() * x.size(0)  # column mask is shared between batch
                    else:
                        active_column = int(
                            grad_output.numel() / grad_output.size(1))
                self.report["backward_weight_core_call"] += weight.size(
                    0) * weight.size(1) * 2 * active_column
                # MAC doubles the cost. lowrank does not impact this.
                # self.report["backward_weight_accum_step"] += max(0,active_column - 1) * 2 * rank_sampler.rank / weight.size(-1)
                self.report["backward_weight_accum_step"] += active_column * 4
            elif(isinstance(feature_sampler, LinearFeatureSampler)):
                mask = feature_sampler.mask.float()  # [bs, q]
                # nonzero * p
                self.report["backward_weight_core_call"] += mask.sum().item() * \
                    weight.size(0)
                self.report["backward_weight_accum_step"] += mask.sum(
                    0).max().item() * x.size(0)
        # input backward
        if(x_requires_grad):
            # share sparsity with feedback or no forward weight sampling, has backward feedback sampling
            if(feedback_sampler.forward_sparsity > 1e-9 or feedback_sampler.backward_sparsity > 1e-9):
                active_weight_block = feedback_sampler.mask.float().sum().item()
                if(x.dim() == 2):  # fully-connected
                    max_accum_step = max(
                        0, feedback_sampler.mask.sum(0).max().item() - 1)  # p
                elif(x.dim() == 4 and feature_sampler.kernel_size != 1):  # CONV3x3
                    # max_accum_step = max(0, feedback_sampler.mask.sum(1).max().item() - 1)  # q ## this is wrong!!
                    max_accum_step = np.ceil(x.size(
                        1)/p)*np.ceil(np.log2(2*k))*np.ceil(feedback_sampler.mask.sum(0).max().item()/2)
                elif(x.dim() == 4 and feature_sampler.kernel_size == 1):  # conv1x1
                    max_accum_step = max(
                        0, feedback_sampler.mask.sum(0).max().item() - 1)  # p
            else:
                active_weight_block = weight.size(0) * weight.size(1)
                if(x.dim() == 2):  # fully-connected
                    max_accum_step = weight.size(0) - 1  # p-1
                elif(x.dim() == 4 and feature_sampler.kernel_size != 1):  # CONV3x3
                    # max_accum_step = weight.size(1) - 1  # q-1 # wrong
                    max_accum_step = np.ceil(
                        x.size(1)/p)*np.ceil(np.log2(2*k))*np.ceil(p/2)
                elif(x.dim() == 4 and feature_sampler.kernel_size == 1):  # conv1x1
                    max_accum_step = weight.size(0) - 1  # p-1
            self.report["backward_input_core_call"] += active_weight_block * \
                int(grad_output.numel()/grad_output.size(1))
            # do not forget we do matrix-matrix mul via sequential mat-vec mul
            self.report["backward_input_accum_step"] += max_accum_step * \
                int(grad_output.numel() / grad_output.size(1))

    def __add__(self, other):
        out = LearningProfiler()
        for k in out.report:
            out.report[k] = self.report[k] + other.report[k]
        return out

    def __radd__(self, other):
        if other == 0:
            return self
        else:
            return self.__add__(other)

    def __sub__(self, other):
        out = LearningProfiler()
        for k in out.report:
            out.report[k] = self.report[k] - other.report[k]
        return out

    def __rsub__(self, other):
        return other.__sub__(self)

    def __truediv__(self, other):
        out = LearningProfiler()
        for k in out.report:
            out.report[k] = self.report[k] / other.report[k]
        return out

    def __rtruediv__(self, other):
        return other.__truediv__(self)


def test_crosstalk_simulator():
    device = "cuda"
    simu = ThermalCrosstalkSimulator(filter_size=5, device=device)
    phase = torch.randn((4*3)//2, device=device) % (2*np.pi)
    phase_c = simu.simple_simulate(phase, None, mode="rectangle")
    print(vector_to_checkerboard(phase))
    print(vector_to_checkerboard(phase_c))


def test_linear_feature_sampler():
    sampler = LinearFeatureSampler(0.5, 4, normalize="none", random_state=42)
    x = torch.randn(4, 4*4)
    xs, _ = sampler.sample(x)
    xr = sampler.reconstruct(xs)
    print(x, '\n', xr)


def test_conv2d_feature_sampler():
    sampler = Conv2dFeatureSampler(0.5, 3, normalize="none", random_state=42)
    x = torch.randn(2, 2, 4, 4)
    xs, _, _ = sampler.sample(x)
    xr = sampler.reconstruct(xs)
    print(x, '\n', xr)


def test_feedback_sampler():
    sampler = FeedbackSampler(0.5, "topk", random_state=43)
    w = torch.randn(2, 2, 2, 2)
    ws = sampler.sample(w)
    print(w)
    print(ws)


def test_singularvalue_sampler():
    p, q, k, r = 1, 1, 4, 4
    sampler = SingularValueGradientSampler(r, alg="uniform")

    s = torch.randn(p, q, k)
    u = torch.nn.init.orthogonal_(torch.randn(p, q, k, k))
    v = torch.nn.init.orthogonal_(torch.randn(p, q, k, k))
    I_U = torch.diag_embed(torch.ones(p, q, k))
    I_V = torch.diag_embed(torch.ones(p, q, k))
    grad_weight = torch.randn(p, q, k, k)
    grad_s = sampler.sample(u, s, v, grad_weight, I_U, I_V)
    print(grad_s)
    print(
        u.transpose(-1, -2).matmul(grad_weight.matmul(v.transpose(-1, -2)))[..., ::k+1])


if __name__ == "__main__":
    # test_crosstalk_simulator()
    # test_linear_feature_sampler()
    # test_conv2d_feature_sampler()
    # test_feedback_sampler()
    test_singularvalue_sampler()
