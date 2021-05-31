#####################
#       mzi_op      #
#####################
import logging
from functools import lru_cache
from multiprocessing.dummy import Pool

import matplotlib.pyplot as plt
import numpy as np
from numpy.core.numeric import cross
import torch
from matplotlib import cm

from .compute import add_gaussian_noise_cpu, clip_by_std, gaussian, gen_gaussian_noise, gen_gaussian_filter2d
from .general import TimerCtx
from .matrix_parametrization import (RealUnitaryDecomposer,
                                     RealUnitaryDecomposerBatch)
from .quantize import uniform_quantize, uniform_quantize_cpu
from .torch_train import apply_weight_decay, set_torch_deterministic

__all__ = ["phase_quantize_fn_cpu", "phase_quantize_fn", "voltage_quantize_fn_cpu", "voltage_quantize_fn", "clip_to_valid_quantized_voltage_cpu", "clip_to_valid_quantized_voltage", "clip_to_valid_quantized_voltage_", "wrap_to_valid_phase", "voltage_to_phase_cpu", "voltage_to_phase", "phase_to_voltage_cpu", "phase_to_voltage", "quantize_phase_of_matrix_cpu", "upper_triangle_to_vector_cpu", "vector_to_upper_triangle_cpu", "upper_triangle_to_vector", "vector_to_upper_triangle", "checkerboard_to_vector", "vector_to_checkerboard", "complex_to_real_projection", "projection_matrix_to_unitary", "real_matrix_parametrization_cpu",
           "real_matrix_reconstruction_cpu", "quantize_voltage_of_matrix_cpu", "maintain_quantized_voltage_cpu", "quantize_voltage_of_unitary_cpu", "maintain_quantized_voltage_of_unitary_cpu", "clip_voltage_of_unitary_cpu", "conditional_update_voltage_of_unitary_cpu", "add_gamma_noise_to_unitary_cpu", "add_phase_noise_to_unitary_cpu", "voltage_quantize_prune_with_gamma_noise_of_unitary_fn", "usv", "voltage_quantize_prune_with_gamma_noise_of_matrix_fn", "voltage_quantize_prune_with_gamma_noise_of_diag_fn",
           "DiagonalQuantizer", "UnitaryQuantizer", "PhaseQuantizer"]


class phase_quantize_fn_cpu(object):
    def __init__(self, p_bit):
        super(phase_quantize_fn_cpu, self).__init__()
        assert p_bit <= 8 or p_bit == 32
        self.p_bit = p_bit
        self.uniform_q = uniform_quantize_cpu(bits=p_bit)
        self.pi = np.pi

    def __call__(self, x):
        if self.p_bit == 32:
            phase_q = x
        elif self.p_bit == 1:
            E = np.mean(np.abs(x))
            phase_q = self.uniform_q(x / E) * E
        else:
            # phase = torch.tanh(x)
            # phase = phase / 2 / torch.max(torch.abs(phase)) + 0.5
            phase = x / 2 / self.pi + 0.5
            # phase_q = 2 * self.uniform_q(phase) - 1
            phase_q = self.uniform_q(phase) * 2 * self.pi - self.pi
        return phase_q


class voltage_quantize_fn_cpu(object):
    def __init__(self, v_bit, v_pi, v_max):
        super(voltage_quantize_fn_cpu, self).__init__()
        assert 0 < v_bit <= 32
        self.v_bit = v_bit
        self.v_pi = v_pi
        self.v_max = v_max
        self.gamma = np.pi / (self.v_pi**2)
        self.uniform_q = uniform_quantize_cpu(bits=v_bit)
        self.pi = np.pi

    def __call__(self, x, voltage_mask_old=None, voltage_mask_new=None, voltage_backup=None, strict_mask=True):
        if self.v_bit == 32:
            voltage_q = x
        elif self.v_bit == 1:
            E = np.mean(np.abs(x))
            voltage_q = self.uniform_q(x / E) * E
        else:
            # min_V = 0
            ### max voltage is determined by the voltage supply, not the phase shifter's characteristics!!! ###
            # max_V = np.sqrt(2*self.pi/self.gamma)
            max_V = self.v_max
            # voltage = (x - min_V) / (max_V - min_V)
            voltage = x / max_V
            # phase_q = 2 * self.uniform_q(phase) - 1
            voltage_q = self.uniform_q(voltage) * max_V

            if(voltage_mask_old is not None and voltage_mask_new is not None and voltage_backup is not None):
                if(strict_mask == True):
                    # strict mask will always fix masked voltages, even though they are not covered in the new mask
                    # "1" in mask indicates to apply quantization
                    voltage_mask_newly_marked = voltage_mask_new ^ voltage_mask_old
                    voltage_q_tmp = x.copy()
                    # maintain voltages that have already been masked
                    voltage_q_tmp[voltage_mask_old] = voltage_backup[voltage_mask_old]
                    # quantize new voltages those are marked in the new mask
                    # print("any newly marked voltages:", voltage_mask_newly_marked.any())
                    voltage_q_tmp[voltage_mask_newly_marked] = voltage_q[voltage_mask_newly_marked]
                    # only update newly quantized voltages, previously quantized voltages are maintained
                    # if (voltage_backup[voltage_mask_newly_marked].sum() > 1e-4):
                    #     print(voltage_backup[voltage_mask_newly_marked])

                    voltage_backup[voltage_mask_newly_marked] = voltage_q[voltage_mask_newly_marked]

                    voltage_q = voltage_q_tmp
                else:
                    # non-strict mask will make unmasked voltages trainable again
                    voltage_q_tmp = x.copy()
                    voltage_mask_old = voltage_mask_old & voltage_mask_new
                    voltage_mask_newly_marked = (
                        ~voltage_mask_old) & voltage_mask_new
                    # maintain voltages that have already been masked and being masked in the new mask
                    voltage_q_tmp[voltage_mask_old] = voltage_backup[voltage_mask_old]
                    # quantize new voltages those are marked in the new mask
                    voltage_q_tmp[voltage_mask_newly_marked] = voltage_q[voltage_mask_newly_marked]

                    voltage_backup[voltage_mask_newly_marked] = voltage_q[voltage_mask_newly_marked]
                    voltage_q = voltage_q_tmp

        return voltage_q


class voltage_quantize_fn(torch.nn.Module):
    def __init__(self, v_bit, v_pi, v_max):
        super(voltage_quantize_fn, self).__init__()
        assert 0 < v_bit <= 32
        self.v_bit = v_bit
        self.v_pi = v_pi
        self.v_max = v_max
        self.gamma = np.pi / (self.v_pi**2)
        self.uniform_q = uniform_quantize(k=v_bit)
        self.pi = np.pi

    def forward(self, x):
        if self.v_bit == 32:
            voltage_q = x
        elif self.v_bit == 1:
            E = x.data.abs().mean()
            voltage_q = self.uniform_q(x / E) * E
        else:
            max_V = self.v_max
            voltage = x / max_V
            voltage_q = self.uniform_q(voltage) * max_V
        return voltage_q


class phase_quantize_fn(torch.nn.Module):
    '''
    description: phase shifter voltage control quantization with gamma noise injection and thermal crosstalk
    '''
    def __init__(self, v_bit, v_pi, v_max, gamma_noise_std=0, crosstalk_factor=0, random_state=None, device=torch.device("cuda")):
        super(phase_quantize_fn, self).__init__()
        assert 0 < v_bit <= 32
        self.v_bit = v_bit
        self.v_pi = v_pi
        self.v_max = v_max
        self.gamma = np.pi / (self.v_pi**2)
        self.gamma_noise_std = gamma_noise_std
        self.crosstalk_factor = crosstalk_factor
        self.voltage_quantizer = voltage_quantize_fn(v_bit, v_pi, v_max)
        self.pi = np.pi
        self.random_state = random_state
        self.device = device

        self.crosstal_simulator = ThermalCrosstalkSimulator(plotting=False, gaussian_filter_size=3,gaussian_filter_std=crosstalk_factor,device=self.device)

    def set_gamma_noise(self, noise_std, random_state=None):
        self.gamma_noise_std = noise_std
        self.random_state = random_state

    def set_crosstalk_factor(self, crosstalk_factor):
        self.crosstalk_factor = crosstalk_factor
        self.crosstal_simulator.set_crosstalk_factor(crosstalk_factor)

    def forward(self, x, mixedtraining_mask=None, mode="triangle"):
        if(self.gamma_noise_std > 1e-5):
            gamma = gen_gaussian_noise(x, noise_mean=self.gamma, noise_std=self.gamma_noise_std, trunc_range=(self.gamma-3*self.gamma_noise_std, self.gamma+3*self.gamma_noise_std), random_state=self.random_state)
        else:
            gamma = self.gamma
        if(self.v_bit >= 16):
            ## no quantization
            ## add gamma noise with quick approach
            phase = (gamma / self.gamma * (x % (2 * np.pi))) % (2 * np.pi)
        else:
            ## quantization
            ## add gamma noise in transform
            phase = voltage_to_phase(
                    clip_to_valid_quantized_voltage(
                        self.voltage_quantizer(
                            phase_to_voltage(x, self.gamma)
                            ),
                        self.gamma, self.v_bit, self.v_max, wrap_around=True
                        ),
                    gamma
                    )
        ## add crosstalk with mixed training mask
        if(self.crosstalk_factor > 1e-5):
            phase = self.crosstal_simulator.simple_simulate(phase, mixedtraining_mask, mode)
        return phase


def clip_to_valid_quantized_voltage_cpu(voltages, gamma, v_bit, v_max, wrap_around=False):
    v_2pi = np.sqrt(2 * np.pi / gamma)
    v_interval = v_max / (2**v_bit-1)
    if(wrap_around):
        mask = voltages >= v_2pi
        voltages[mask] = 0
        # invalid_voltages = voltages[mask]
        # invalid_phases = gamma * invalid_voltages * invalid_voltages
        # invalid_phases -= 2 * np.pi
        # valid_voltages = np.sqrt(invalid_phases / gamma)
        # valid_voltages_q = np.round(valid_voltages / v_interval) * v_interval
        # voltages[mask] = valid_voltages_q
    else:
        voltages[voltages > v_2pi] -= v_interval
    return voltages


def clip_to_valid_quantized_voltage(voltages, gamma, v_bit, v_max, wrap_around=False):
    v_2pi = np.sqrt(2 * np.pi / gamma)
    v_interval = v_max / (2**v_bit-1)
    if((isinstance(voltages, np.ndarray))):
        if(wrap_around):
            mask = voltages >= v_2pi
            voltages = voltages.copy()
            voltages[mask] = 0
        else:
            voltages = voltages.copy()
            voltages[voltages > v_2pi] -= v_interval
    elif((isinstance(voltages, torch.Tensor))):
        if(wrap_around):
            mask = voltages.data < v_2pi
            voltages = voltages.mul(mask.float())
            # voltages = voltages.data.clone()
            # voltages[mask] = 0
        else:
            mask = voltages > v_2pi
            voltages = voltages.masked_scatter(mask, voltages[mask] - v_interval)
            # voltages = voltages.data.clone()
            # voltages[voltages > v_2pi] -= v_interval
    else:
        assert 0, logging.error(f"Array type not supported, must be numpy.ndarray or torch.Tensor, but got {type(voltages)}")

    return voltages


def clip_to_valid_quantized_voltage_(voltages, gamma, v_bit, v_max, wrap_around=False):
    v_2pi = np.sqrt(2 * np.pi / gamma)
    v_interval = v_max / (2**v_bit-1)
    if((isinstance(voltages, np.ndarray))):
        if(wrap_around):
            mask = voltages >= v_2pi
            voltages[mask] = 0
        else:
            voltages[voltages > v_2pi] -= v_interval
    elif((isinstance(voltages, torch.Tensor))):
        if(wrap_around):
            mask = voltages >= v_2pi
            voltages.data[mask] = 0
        else:
            voltages.data[voltages > v_2pi] -= v_interval
    else:
        assert 0, logging.error(f"Array type not supported, must be numpy.ndarray or torch.Tensor, but got {type(voltages)}")

    return voltages


def wrap_to_valid_phase(phases, mode="positive"):
    assert mode in {"symmetric", "positive"}
    if(mode == "positive"):
        phases = phases % (2 * np.pi)
        return phases
    elif(mode == "symmetric"):
        phases = phases % (2 * np.pi)
        phases.data[phases > np.pi] -= 2 * np.pi
        return phases


def voltage_to_phase_cpu(voltages, gamma):
    # phases = -np.clip(gamma * voltages * voltages, a_min=0, a_max=2 * np.pi)
    # change phase range from [0, 2*pi] to [-pi,pi]
    pi_2 = 2*np.pi
    phases = (gamma * voltages * voltages) % pi_2
    phases[phases > np.pi] -= pi_2
    return phases

voltage_to_phase = voltage_to_phase_cpu

def phase_to_voltage_cpu(phases, gamma):
    pi = np.pi
    if(isinstance(phases, np.ndarray)):
        phases_tmp = phases.copy()
        phases_tmp[phases_tmp > 0] -= 2 * pi  # change phase lead to phase lag
        voltage_max = np.sqrt((2 * pi) / gamma)
        voltages = np.clip(np.sqrt(np.abs(phases_tmp / gamma)),
                        a_min=0, a_max=voltage_max)
    else:
        voltages = (phases % (2*np.pi)).div(gamma).sqrt()
        # phases_tmp = phases.data.clone()
        # phases_tmp[phases_tmp > 0] -= 2 * pi
        # voltage_max = np.sqrt((2*pi) / gamma)
        # voltages = phases_tmp.mul_(1/gamma).abs_().sqrt_().clamp_(0, voltage_max)
    return voltages

phase_to_voltage = phase_to_voltage_cpu

def quantize_phase_of_matrix_cpu(W, p_bit, output_device=torch.device("cuda")):
    assert isinstance(
        p_bit, int) and p_bit >= 1, "[E] quantization bit must be integer larger than 1"

    decomposer = RealUnitaryDecomposer()
    phase_quantize_fn = phase_quantize_fn_cpu(p_bit=p_bit)

    if(isinstance(W, np.ndarray)):
        W_numpy = W.astype(np.float64)
    elif(isinstance(W, torch.Tensor)):
        W_numpy = W.detach().cpu().numpy().astype(np.float64)
    else:
        assert 0, "[E] Array type not supported, must be numpy.ndarray or torch.Tensor"

    M, N = W_numpy.shape[0], W_numpy.shape[1]
    U, Sigma, V = np.linalg.svd(W_numpy, full_matrices=True)

    Sigma = np.diag(Sigma)
    if(M > N):
        Sigma = np.concatenate([Sigma, np.zeros([M - N, N])], axis=0)
    elif(M < N):
        Sigma = np.concatenate([Sigma, np.zeros([M, N - M])], axis=1)

    delta_list_U, phi_mat_U = decomposer.decompose(U)
    delta_list_V, phi_mat_V = decomposer.decompose(V)

    phi_list_U = np.zeros([M*(M-1)//2], dtype=np.float64)
    phi_list_V = np.zeros([N*(N-1)//2], dtype=np.float64)
    count = 0
    for i in range(M):
        for j in range(M - i - 1):
            phi_list_U[count] = phi_mat_U[i, j]
            count += 1
    count = 0
    for i in range(N):
        for j in range(N - i - 1):
            phi_list_V[count] = phi_mat_V[i, j]
            count += 1

    phi_list_U_q = phase_quantize_fn(phi_list_U)
    phi_list_V_q = phase_quantize_fn(phi_list_V)
    # phi_list_U_q = phi_list_U
    # phi_list_V_q = phi_list_V

    count = 0
    for i in range(M):
        for j in range(M - i - 1):
            phi_mat_U[i, j] = phi_list_U_q[count]
            count += 1
    count = 0
    for i in range(N):
        for j in range(N - i - 1):
            phi_mat_V[i, j] = phi_list_V_q[count]
            count += 1

    U_recon = decomposer.reconstruct_2(delta_list_U, phi_mat_U)
    V_recon = decomposer.reconstruct_2(delta_list_V, phi_mat_V)

    U_recon = torch.from_numpy(U_recon).to(output_device)
    V_recon = torch.from_numpy(V_recon).to(output_device)
    Sigma = torch.from_numpy(Sigma).to(output_device)

    W_recon = torch.mm(U_recon, torch.mm(Sigma, V_recon)).to(torch.float32)
    # plt.figure()
    # plt.subplot(2,1,1)
    # plt.hist(W_numpy.reshape([-1]), bins=256, range=[-1, 1])
    # plt.subplot(2,1,2)
    # plt.hist(W_recon.detach().cpu().numpy().reshape([-1]), bins=256, range=[-1, 1])
    # plt.show()
    # plt.figure()
    # plt.subplot(2,1,1)
    # plt.hist(phi_list_U.reshape([-1]), bins=256, range=[-np.pi, np.pi])
    # plt.subplot(2,1,2)
    # plt.hist(phi_list_U_q.reshape([-1]), bins=256, range=[-np.pi, np.pi])
    # plt.show()
    return W_recon


@lru_cache(maxsize=32)
def upper_triangle_masks_cpu(N):
    rows, cols = np.triu_indices(N, 1)
    masks = (rows, cols - rows - 1)
    return masks


@lru_cache(maxsize=32)
def upper_triangle_masks(N, device=torch.device("cuda")):
    masks = torch.triu_indices(N, N, 1, device=device)
    masks[1,:] -= masks[0, :] + 1
    return masks


def upper_triangle_to_vector(mat, complex=False):
    if(isinstance(mat, np.ndarray)):
        N = mat.shape[-2] if complex else mat.shape[-1]
        masks = upper_triangle_masks_cpu(N)
        if(complex):
            vector = mat[..., masks[0], masks[1], :]
        else:
            vector = mat[..., masks[0], masks[1]]
    elif(isinstance(mat, torch.Tensor)):
        N = mat.shape[-2] if complex else mat.shape[-1]
        masks = upper_triangle_masks(N, device=mat.device)
        if(complex):
            vector = mat[..., masks[0], masks[1], :]
        else:
            vector = mat[..., masks[0], masks[1]]
    else:
        raise NotImplementedError

    return vector

upper_triangle_to_vector_cpu = upper_triangle_to_vector


def vector_to_upper_triangle(vec, complex=False):
    ### Support numpy ndarray and torch Tensor. Batched operation is supported
    if(isinstance(vec, np.ndarray)):
        M = vec.shape[-2] if complex else vec.shape[-1]
        N = (1 + int(np.sqrt(1 + 8 * M))) // 2
        masks = upper_triangle_masks_cpu(N)
        if(complex):
            mat = np.zeros(shape=list(vec.shape[:-2])+[N, N, vec.shape[-1]], dtype=vec.dtype)
            mat[..., masks[0], masks[1], :] = vec
        else:
            mat = np.zeros(shape=list(vec.shape[:-1])+[N, N], dtype=vec.dtype)
            mat[..., masks[0], masks[1]] = vec
    elif(isinstance(vec, torch.Tensor)):
        M = vec.shape[-2] if complex else vec.shape[-1]
        N = (1 + int(np.sqrt(1 + 8 * M))) // 2
        masks = upper_triangle_masks(N, device=vec.device)
        if(complex):
            mat = torch.zeros(size=list(vec.size())[:-2]+[N, N, vec.size(-1)], dtype=vec.dtype, device=vec.device)
            mat[..., masks[0], masks[1], :] = vec
        else:
            mat = torch.zeros(size=list(vec.size())[:-1]+[N, N], dtype=vec.dtype, device=vec.device)
            mat[..., masks[0], masks[1]] = vec
    else:
        raise NotImplementedError

    return mat

vector_to_upper_triangle_cpu = vector_to_upper_triangle


def checkerboard_to_vector(mat, complex=False):
    ### Support numpy ndarray and torch Tensor. Batched operation is supported
    ### even column phases + odd colum phases, compact layoutapplication

    if(isinstance(mat, np.ndarray)):
        if(complex):
            mat = np.transpose(mat, axes=np.roll(np.arange(mat.ndim), 1))
        N = mat.shape[-1]
        upper_oddN = N - (N % 2 == 1)
        upper_evenN = N - (N % 2 == 0)
        vector_even_col = np.swapaxes(mat[..., :upper_oddN:2, ::2], -1, -2).reshape([*mat.shape[:-2], -1])
        vector_odd_col = np.swapaxes(mat[..., 1:upper_evenN:2, 1::2], -1, -2).reshape([*mat.shape[:-2], -1])
        vector = np.concatenate([vector_even_col, vector_odd_col], -1)
        if(complex):
            vector = np.transpose(vector, axes=np.roll(np.arange(vector.ndim), -1))
    elif(isinstance(mat, torch.Tensor)):
        if(complex):
            mat = torch.permute(mat, list(np.roll(np.arange(mat.ndim), 1)))
        N = mat.shape[-1]
        upper_oddN = N - (N % 2 == 1)
        upper_evenN = N - (N % 2 == 0)
        vector_even_col = torch.transpose(mat[..., :upper_oddN:2, ::2], -1, -2).reshape(list(mat.size())[:-2] + [-1])
        vector_odd_col = torch.transpose(mat[..., 1:upper_evenN:2, 1::2], -1, -2).reshape(list(mat.size())[:-2] + [-1])
        vector = torch.cat([vector_even_col, vector_odd_col], -1)
        if(complex):
            vector = torch.permute(vector, list(np.roll(np.arange(vector.ndim), -1)))
    else:
        raise NotImplementedError
    return vector


def vector_to_checkerboard(vec, complex=False):
    ### Support numpy ndarray and torch Tensor. Batched operation is supported
    ### from compact phase vector (even col + odd col) to clements checkerboard
    if(isinstance(vec, np.ndarray)):
        if(complex):
            vec = np.transpose(vec, axes=np.roll(np.arange(vec.ndim), 1))
        M = vec.shape[-1]
        N = (1 + int(np.sqrt(1 + 8 * M))) // 2
        upper_oddN = N - (N % 2 == 1)
        upper_evenN = N - (N % 2 == 0)
        vector_even_col = vec[..., :(N//2)*((N+1)//2)]
        vector_odd_col = vec[..., (N//2)*((N+1)//2):]
        mat = np.zeros([*vec.shape[:-1], N, N], dtype=vec.dtype)
        mat[..., ::2, :upper_oddN:2] = vector_even_col.reshape([*vec.shape[:-1], (N+1)//2, -1])
        mat[..., 1::2, 1:upper_evenN:2] = vector_odd_col.reshape([*vec.shape[:-1], N//2, -1])
        mat = np.swapaxes(mat, -1, -2)
        if(complex):
            mat = np.transpose(mat, axes=np.roll(np.arange(mat.ndim), -1))
    elif(isinstance(vec, torch.Tensor)):
        if(complex):
            vec = torch.permute(vec, list(np.roll(np.arange(vec.ndim), 1)))
        M = vec.size(-1)
        N = (1 + int(np.sqrt(1 + 8 * M))) // 2
        upper_oddN = N - (N % 2 == 1)
        upper_evenN = N - (N % 2 == 0)
        vector_even_col = vec[..., :(N//2)*((N+1)//2)]
        vector_odd_col = vec[..., (N//2)*((N+1)//2):]
        mat = torch.zeros([*vec.shape[:-1], N, N], device=vec.device, dtype=vec.dtype)
        mat[..., ::2, :upper_oddN:2] = vector_even_col.reshape([*vec.shape[:-1], (N+1)//2, -1])
        mat[..., 1::2, 1:upper_evenN:2] = vector_odd_col.reshape([*vec.shape[:-1], N//2, -1])

        mat = torch.transpose(mat, -1, -2)
        if(complex):
            mat = torch.permute(mat, list(np.roll(np.arange(mat.ndim), -1)))
    else:
        raise NotImplementedError
    return mat

class ComplexToRealProjectionFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        mag = torch.abs(x)
        angle = torch.angle(x)
        pos_mask = (angle <= np.pi/2) & (angle >= -np.pi/2)
        del angle
        neg_mask = ~pos_mask
        ctx.save_for_backward(x, neg_mask)

        x = torch.empty_like(x.real)
        x[pos_mask] = mag[pos_mask]
        x[neg_mask] = -mag[neg_mask]
        return x
    @staticmethod
    def backward(ctx, grad_output):
        ### the gradient flow through angle is ignored
        x, neg_mask = ctx.saved_tensors()
        grad_mag = grad_output.clone()
        grad_mag[neg_mask] *= -1

        mag = torch.abs(x)
        grad_real = grad_mag * x.real / mag
        grad_imag = grad_mag * x.imag / mag
        return torch.complex(grad_real, grad_imag)


def complex_to_real_projection(x):
    if(isinstance(x, np.ndarray)):
        mag = np.abs(x)
        mask = x.real < 0
        mag[mask] *= -1
        x = mag
    elif(isinstance(x, torch.Tensor)):
        mag = x.real.square().add(x.imag.square()).add(1e-12).sqrt()
        mask = x.real < 0
        x = mag.masked_scatter(mask, -mag[mask])
    else:
        raise NotImplementedError
    return x

def projection_matrix_to_unitary(W):
    if(isinstance(W, np.ndarray)):
        U, _, V = np.linalg.svd(W, full_matrices=True)
        U_refine = np.matmul(U, V)
    elif(isinstance(W, torch.Tensor)):
        U, _, V = torch.svd(W, some=False)
        U_refine = torch.matmul(U, V.t())
    else:
        raise NotImplementedError
    return U_refine


def real_matrix_parametrization_cpu(W):
    decomposer = RealUnitaryDecomposer()
    M, N = W.shape[0], W.shape[1]
    U, Sigma, V = np.linalg.svd(W, full_matrices=True)

    Sigma = np.diag(Sigma)
    if(M > N):
        Sigma = np.concatenate([Sigma, np.zeros([M - N, N])], axis=0)
    elif(M < N):
        Sigma = np.concatenate([Sigma, np.zeros([M, N - M])], axis=1)

    delta_list_U, phi_mat_U = decomposer.decompose(U)
    delta_list_V, phi_mat_V = decomposer.decompose(V)

    phi_list_U = upper_triangle_to_vector_cpu(phi_mat_U)
    phi_list_V = upper_triangle_to_vector_cpu(phi_mat_V)

    return Sigma, delta_list_U, phi_list_U, delta_list_V, phi_list_V


def real_matrix_parametrization_cpu_ref(W):
    M, N = W.shape[0], W.shape[1]
    U, Sigma, V = np.linalg.svd(W, full_matrices=True)

    Sigma = np.diag(Sigma)
    if(M > N):
        Sigma = np.concatenate([Sigma, np.zeros([M - N, N])], axis=0)
    elif(M < N):
        Sigma = np.concatenate([Sigma, np.zeros([M, N - M])], axis=1)

    delta_list_U, phi_mat_U = decompose_ref(U)
    delta_list_V, phi_mat_V = decompose_ref(V)

    phi_list_U = upper_triangle_to_vector_cpu(phi_mat_U)
    phi_list_V = upper_triangle_to_vector_cpu(phi_mat_V)

    return Sigma, delta_list_U, phi_list_U, delta_list_V, phi_list_V


def real_matrix_reconstruction_cpu(Sigma, delta_list_U, phi_list_U, delta_list_V, phi_list_V):
    decomposer = RealUnitaryDecomposer()
    phi_mat_U = vector_to_upper_triangle_cpu(phi_list_U)
    phi_mat_V = vector_to_upper_triangle_cpu(phi_list_V)

    U_recon = decomposer.reconstruct_2(delta_list_U, phi_mat_U)
    V_recon = decomposer.reconstruct_2(delta_list_V, phi_mat_V)
    # print("checkU:",decomposer.checkUnitary(U_recon), decomposer.checkUnitary(V_recon))

    W_recon = np.dot(U_recon, np.dot(Sigma, V_recon))

    return W_recon


def quantize_voltage_of_matrix_cpu(W, v_bit, v_pi=4.36, v_max=10.8, voltage_mask_U=None, voltage_backup_U=None, voltage_mask_V=None, voltage_backup_V=None, quantize_voltage_percentile=0, clamp_small_phase_lead_percentile=1, output_device=torch.device("cuda")):
    assert isinstance(
        v_bit, int) and v_bit >= 1, "[E] quantization bit must be integer larger than 1"
    assert 0 < clamp_small_phase_lead_percentile <= 1, "[E] Clamp phase lead percentile must be within (0, 1]"
    assert 0 <= quantize_voltage_percentile <= 1, "[E] Quantize voltage percentile must be within [0, 1]"

    gamma = np.pi / (v_pi**2)
    voltage_quantize_fn = voltage_quantize_fn_cpu(
        v_bit=v_bit, v_pi=v_pi, v_max=v_max)

    if(isinstance(W, np.ndarray)):
        W_numpy = W.copy().astype(np.float64)
    elif(isinstance(W, torch.Tensor)):
        W_numpy = W.detach().cpu().numpy().copy().astype(np.float64)
    else:
        assert 0, "[E] Array type not supported, must be numpy.ndarray or torch.Tensor"

    Sigma, delta_list_U, phi_list_U, delta_list_V, phi_list_V = real_matrix_parametrization_cpu(
        W_numpy)

    v_list_U = phase_to_voltage_cpu(phi_list_U, gamma)
    v_list_V = phase_to_voltage_cpu(phi_list_V, gamma)

    # if(clamp_small_phase_lead_percentile < 1):
    #     thres = np.percentile(v_list_U, clamp_small_phase_lead_percentile*100)
    #     v_list_U[v_list_U > thres] = 0
    #     thres = np.percentile(v_list_V, clamp_small_phase_lead_percentile*100)
    #     v_list_V[v_list_V > thres] = 0
    if(voltage_mask_U is not None and voltage_backup_U is not None and voltage_mask_V is not None and voltage_backup_V is not None):
        v_thres_U = np.percentile(
            v_list_U, (1-quantize_voltage_percentile)*100)
        voltage_mask_U_old = voltage_mask_U.copy()
        voltage_mask_U_new = voltage_mask_U_old | (v_list_U >= v_thres_U)
        v_thres_V = np.percentile(
            v_list_V, (1-quantize_voltage_percentile)*100)
        voltage_mask_V_old = voltage_mask_V.copy()
        voltage_mask_V_new = voltage_mask_V_old | (v_list_V >= v_thres_V)
        print(f"[I] Voltage_threshold: U={v_thres_U}, V={v_thres_V}")

        v_list_U_q = voltage_quantize_fn(
            v_list_U, voltage_mask_U_old, voltage_mask_U_new, voltage_backup_U)
        v_list_V_q = voltage_quantize_fn(
            v_list_V, voltage_mask_V_old, voltage_mask_V_new, voltage_backup_V)
        voltage_mask_U[:] = voltage_mask_U_new[:]
        voltage_mask_V[:] = voltage_mask_V_new[:]
        print(f"{voltage_backup_U.max(), voltage_backup_U.min(),voltage_backup_V.max(), voltage_backup_V.min()}")
        print(voltage_mask_U.all(), voltage_mask_V.all())
        if(voltage_mask_U.all() and voltage_mask_V.all()):
            print("mask is all True")
            print("sigma", Sigma)
            print("delta:", delta_list_U)
            print("v_list_U_q:", v_list_U_q)
            print("v_backup_U", voltage_backup_U)
    else:
        v_list_U_q = voltage_quantize_fn(v_list_U, None, None, None)
        v_list_V_q = voltage_quantize_fn(v_list_V, None, None, None)
        # print("before Q:", v_list_U)
        # print("after Q:", v_list_U_q)
        # print("backup:", voltage_backup_U)
        # print("Mask Is All True")
        # print("sigma", Sigma)
        # print("delta:", delta_list_U)
        # print("v_list_U_q:", v_list_U_q)
        # print("v_backup_U", voltage_backup_U)

    # print("v_list_U_q:", v_list_U_q)
    # print("v_backup_U:", voltage_backup_U)
    # print("v_list_V_q:", v_list_V_q)
    # print("v_backup_V:", voltage_backup_V)

    phi_list_U_q = voltage_to_phase_cpu(v_list_U_q, gamma)
    phi_list_V_q = voltage_to_phase_cpu(v_list_V_q, gamma)
    # phi_list_U_q = phi_list_U
    # phi_list_V_q = phi_list_V

    W_recon = real_matrix_reconstruction_cpu(
        Sigma, delta_list_U, phi_list_U_q, delta_list_V, phi_list_V_q)

    W_recon = torch.from_numpy(W_recon).to(torch.float32).to(output_device)

    # res = check_equal_tensor(W, W_recon)
    # print("[I] checkEqual: ", res)
    # # print("S:", Sigma)
    # # print("delta_U:", delta_list_U)
    # # print("delta_V:", delta_list_V)
    # print("W:", W)
    # print("W_rec:", W_recon)

    # bin=2**v_bit
    # plt.figure()
    # plt.subplot(2,1,1)
    # plt.hist(W_numpy.reshape([-1]), bins=bin, range=[-1, 1])
    # plt.subplot(2,1,2)
    # plt.hist(W_recon.detach().cpu().numpy().reshape([-1]), bins=bin, range=[-1, 1])

    # plt.figure()
    # plt.subplot(2,1,1)
    # plt.hist(phi_list_U.reshape([-1]), bins=bin, range=[-np.pi, np.pi])
    # plt.subplot(2,1,2)
    # plt.hist(phi_list_U_q.reshape([-1]), bins=bin, range=[-np.pi, np.pi])

    # v_max = np.sqrt(2*np.pi/gamma)
    # plt.figure()
    # plt.subplot(2,1,1)
    # plt.hist(v_list_U.reshape([-1]), bins=bin, range=[0, v_max])
    # plt.subplot(2,1,2)
    # plt.hist(v_list_U_q.reshape([-1]), bins=bin, range=[0, v_max])
    # plt.show()

    return W_recon


def maintain_quantized_voltage_cpu(W, v_pi, voltage_mask_U, voltage_backup_U, voltage_mask_V, voltage_backup_V, output_device=torch.device("cuda")):
    if(isinstance(W, np.ndarray)):
        W_numpy = W.copy().astype(np.float64)
    elif(isinstance(W, torch.Tensor)):
        W_numpy = W.detach().cpu().numpy().copy().astype(np.float64)
    else:
        assert 0, "[E] Array type not supported, must be numpy.ndarray or torch.Tensor"

    Sigma, delta_list_U, phi_list_U, delta_list_V, phi_list_V = real_matrix_parametrization_cpu(
        W_numpy)

    gamma = np.pi / (v_pi**2)
    v_list_U = phase_to_voltage_cpu(phi_list_U, gamma)
    v_list_V = phase_to_voltage_cpu(phi_list_V, gamma)
    # print("maintain:",voltage_mask_U.all(), voltage_mask_V.all())
    v_list_U[voltage_mask_U] = voltage_backup_U[voltage_mask_U]
    v_list_V[voltage_mask_V] = voltage_backup_V[voltage_mask_V]

    phi_list_U = voltage_to_phase_cpu(v_list_U, gamma)
    phi_list_V = voltage_to_phase_cpu(v_list_V, gamma)

    W_recon = real_matrix_reconstruction_cpu(
        Sigma, delta_list_U, phi_list_U, delta_list_V, phi_list_V)

    W_recon = torch.from_numpy(W_recon).to(torch.float32).to(output_device)
    W.data.copy_(W_recon)


def quantize_voltage_of_unitary_cpu(W, v_bit, v_pi=4.36, v_max=10.8, voltage_mask=None, voltage_backup=None, quantize_voltage_percentile=0, strict_mask=True, clamp_small_phase_lead_percentile=1, output_device=torch.device("cuda")):
    assert isinstance(
        v_bit, int) and v_bit >= 1, "[E] quantization bit must be integer larger than 1"
    assert 0 < clamp_small_phase_lead_percentile <= 1, "[E] Clamp phase lead percentile must be within (0, 1]"
    assert 0 <= quantize_voltage_percentile <= 1, "[E] Quantize voltage percentile must be within [0, 1]"

    gamma = np.pi / (v_pi**2)
    voltage_quantize_fn = voltage_quantize_fn_cpu(
        v_bit=v_bit, v_pi=v_pi, v_max=v_max)

    if(isinstance(W, np.ndarray)):
        W_numpy = W.copy().astype(np.float64)
    elif(isinstance(W, torch.Tensor)):
        W_numpy = W.detach().cpu().numpy().copy().astype(np.float64)
    else:
        assert 0, "[E] Array type not supported, must be numpy.ndarray or torch.Tensor"
    batch_mode = len(W_numpy.shape) > 2

    if(batch_mode):
        decomposer = RealUnitaryDecomposerBatch()
        delta_list, phi_mat = decomposer.decompose_batch(W_numpy)
    else:
        decomposer = RealUnitaryDecomposer()
        delta_list, phi_mat = decomposer.decompose(W_numpy)
    phi_list = upper_triangle_to_vector_cpu(phi_mat)

    v_list = phase_to_voltage_cpu(phi_list, gamma)

    if(voltage_mask is not None and voltage_backup is not None):
        v_thres = np.percentile(
            v_list, (1-quantize_voltage_percentile)*100)
        voltage_mask_old = voltage_mask.copy()
        if(strict_mask == True):
            voltage_mask_new = voltage_mask_old | (v_list >= v_thres)
        else:
            voltage_mask_new = v_list >= v_thres
        # print(f"[I] Voltage_threshold: U={v_thres}")

        v_list_q = voltage_quantize_fn(
            v_list, voltage_mask_old, voltage_mask_new, voltage_backup, strict_mask)

        voltage_mask[:] = voltage_mask_new[:]

        # print(f"{voltage_backup.max(), voltage_backup.min()}")
        # if(voltage_mask.all()):
        #     print("mask is all True")
        #     print("delta:",delta_list)
        #     print("v_list_q:",v_list_q)
        #     print("v_backup",voltage_backup)
    else:
        v_list_q = voltage_quantize_fn(v_list, None, None, None, True)
        v_list_q = clip_to_valid_quantized_voltage_cpu(
            v_list_q, gamma, v_bit, v_max, wrap_around=True)
        # print("before Q:", v_list_U)
        # print("after Q:", v_list_U_q)
        # print("backup:", voltage_backup_U)
        # print("Mask Is All True")
        # print("delta:",delta_list)
        # print("v_list_q:",v_list_q)
        # print("v_backup",voltage_backup)

    # print("v_list_U_q:", v_list_U_q)
    # print("v_backup_U:", voltage_backup_U)
    # print("v_list_V_q:", v_list_V_q)
    # print("v_backup_V:", voltage_backup_V)

    phi_list_q = voltage_to_phase_cpu(v_list_q, gamma)
    # phi_list_U_q = phi_list_U
    # phi_list_V_q = phi_list_V
    phi_mat_q = vector_to_upper_triangle_cpu(phi_list_q)

    if(batch_mode):
        W_recon = decomposer.reconstruct_2_batch(delta_list, phi_mat_q)
    else:
        W_recon = decomposer.reconstruct_2(delta_list, phi_mat_q)

    W_recon = torch.from_numpy(W_recon).to(torch.float32).to(output_device)

    # res = check_equal_tensor(W, W_recon)
    # print("[I] checkEqual: ", res)
    # print("S:", Sigma)
    # print("delta_U:", delta_list_U)
    # print("delta_V:", delta_list_V)
    # print("W:", W)
    # print("W_rec:", W_recon)

    # bin=2**v_bit
    # plt.figure()
    # plt.subplot(2,1,1)
    # plt.hist(W_numpy.reshape([-1]), bins=bin, range=[-1, 1])
    # plt.subplot(2,1,2)
    # plt.hist(W_recon.detach().cpu().numpy().reshape([-1]), bins=bin, range=[-1, 1])

    # plt.figure()
    # plt.subplot(2,1,1)
    # plt.hist(phi_list_U.reshape([-1]), bins=bin, range=[-np.pi, np.pi])
    # plt.subplot(2,1,2)
    # plt.hist(phi_list_U_q.reshape([-1]), bins=bin, range=[-np.pi, np.pi])

    # v_max = np.sqrt(2*np.pi/gamma)
    # plt.figure()
    # plt.subplot(2,1,1)
    # plt.hist(v_list_U.reshape([-1]), bins=bin, range=[0, v_max])
    # plt.subplot(2,1,2)
    # plt.hist(v_list_U_q.reshape([-1]), bins=bin, range=[0, v_max])
    # plt.show()

    return W_recon


def maintain_quantized_voltage_of_unitary_cpu(W, v_pi, voltage_mask, voltage_backup, gamma_noise_std=0, weight_decay_rate=0, learning_rate=0, clip_voltage=False, lower_thres=float("-inf"), upper_thres=float("inf"), output_device=torch.device("cuda")):
    assert gamma_noise_std >= 0, "[E] Gamma noise standard deviation must be non-negative"
    if(isinstance(W, np.ndarray)):
        W_numpy = W.copy().astype(np.float64)
    elif(isinstance(W, torch.Tensor)):
        W_numpy = W.detach().cpu().numpy().copy().astype(np.float64)
    else:
        assert 0, "[E] Array type not supported, must be numpy.ndarray or torch.Tensor"

    gamma = np.pi / (v_pi**2)

    batch_mode = len(W_numpy.shape) > 2
    if(batch_mode):
        decomposer = RealUnitaryDecomposerBatch()
        delta_list, phi_mat = decomposer.decompose_batch(W_numpy)
    else:
        decomposer = RealUnitaryDecomposer()
        delta_list, phi_mat = decomposer.decompose(W_numpy)
    phi_list = upper_triangle_to_vector_cpu(phi_mat)

    v_list = phase_to_voltage_cpu(phi_list, gamma)

    v_list[voltage_mask] = voltage_backup[voltage_mask]

    if(weight_decay_rate > 1e-6 and learning_rate > 1e-6):
        apply_weight_decay(v_list, decay_rate=weight_decay_rate,
                           learning_rate=learning_rate, mask=voltage_mask)

    if(clip_voltage == True):
        v_max = np.sqrt(2*np.pi/gamma)
        lower_mask_1 = (v_list > (lower_thres + 0)/2) & (v_list < lower_thres)
        lower_mask_2 = v_list <= (lower_thres + 0)/2
        upper_mask_1 = (v_list > upper_thres) & (
            v_list < (upper_thres + v_max)/2)
        upper_mask_2 = (v_list >= (upper_thres + v_max)/2)
        v_list[lower_mask_1] = lower_thres
        v_list[lower_mask_2] = 0
        v_list[upper_mask_1] = upper_thres
        v_list[upper_mask_2] = 0

    if(gamma_noise_std > 1e-4):
        pool = Pool(2)
        # reconstruct unitary matrix without noise
        phi_list = voltage_to_phase_cpu(v_list, gamma)
        phi_mat = vector_to_upper_triangle_cpu(phi_list)
        # W_recon = decomposer.reconstruct_2(delta_list, phi_mat)
        # W_recon = torch.from_numpy(W_recon).to(torch.float32).to(output_device)

        # reconstruct unitary matrix with gamma noise
        N = W_numpy.shape[0]
        gamma_with_noise = add_gaussian_noise_cpu(np.zeros_like(
            v_list, dtype=np.float64), noise_mean=gamma, noise_std=gamma_noise_std, trunc_range=())
        # Must add noise to all voltages to model the correct noise distribution
        # gamma_with_noise[~voltage_mask] = gamma ### only add noise to masked voltages, avoid aggressive noise error?
        phi_list_n = voltage_to_phase_cpu(v_list, gamma_with_noise)
        phi_mat_n = vector_to_upper_triangle_cpu(phi_list_n)
        # W_recon_n = decomposer.reconstruct_2(delta_list, phi_mat_n)
        # W_recon_n = torch.from_numpy(W_recon_n).to(torch.float32).to(output_device)
        if(batch_mode):
            W_recon, W_recon_n = pool.map(lambda x: torch.from_numpy(decomposer.reconstruct_2_batch(
                delta_list=delta_list, phi_mat=x)).to(torch.float32).to(output_device), [phi_mat, phi_mat_n])
        else:
            W_recon, W_recon_n = pool.map(lambda x: torch.from_numpy(decomposer.reconstruct_2(
                delta_list=delta_list, phi_mat=x)).to(torch.float32).to(output_device), [phi_mat, phi_mat_n])
        pool.close()

        return W_recon, W_recon_n

    else:
        phi_list = voltage_to_phase_cpu(v_list, gamma)
        phi_mat = vector_to_upper_triangle_cpu(phi_list)
        if(batch_mode):
            W_recon = decomposer.reconstruct_2_batch(delta_list, phi_mat)
        else:
            W_recon = decomposer.reconstruct_2(delta_list, phi_mat)
        W_recon = torch.from_numpy(W_recon).to(torch.float32).to(output_device)

        return W_recon, None
    # W.data.copy_(W_recon)


def clip_voltage_of_unitary_cpu(W, v_pi, lower_thres=0, upper_thres=float('inf'), voltage_mask=None, voltage_backup=None, output_device=torch.device("cuda")):
    if(isinstance(W, np.ndarray)):
        W_numpy = W.copy().astype(np.float64)
    elif(isinstance(W, torch.Tensor)):
        W_numpy = W.detach().cpu().numpy().copy().astype(np.float64)
    else:
        assert 0, "[E] Array type not supported, must be numpy.ndarray or torch.Tensor"

    gamma = np.pi / (v_pi**2)

    N = W_numpy.shape[0]

    decomposer = RealUnitaryDecomposer()
    delta_list, phi_mat = decomposer.decompose(W_numpy)
    phi_list = upper_triangle_to_vector_cpu(phi_mat)

    v_list = phase_to_voltage_cpu(phi_list, gamma)

    if(voltage_mask is None and voltage_backup is None):
        v_max = np.sqrt(2*np.pi/gamma)
        # lower_thres = np.percentile(v_list, lower_perc * 100)
        # upper_thres = np.percentile(v_list, upper_perc * 100)
        lower_mask = v_list < lower_thres
        upper_mask_1 = (v_list > upper_thres) & (
            v_list < (upper_thres + v_max)/2)
        upper_mask_2 = (v_list >= (upper_thres + v_max)/2)
        v_list[lower_mask] = lower_thres
        v_list[upper_mask_1] = upper_thres
        v_list[upper_mask_2] = 0
    else:
        raise NotImplementedError

    phi_list = voltage_to_phase_cpu(v_list, gamma)

    phi_mat = vector_to_upper_triangle_cpu(phi_list)

    W_recon = decomposer.reconstruct_2(delta_list, phi_mat)

    W_recon = torch.from_numpy(W_recon).to(torch.float32).to(output_device)

    return W_recon


def conditional_update_voltage_of_unitary_cpu(W, v_bit, v_pi, v_max, lambda3, voltage_mask, voltage_backup, gamma_noise_std=0, weight_decay_rate=0, learning_rate=0, clip_voltage=False, lower_thres=float("-inf"), upper_thres=float("inf"), return_ori=True, crosstalk_factor=0, mixedtraining_mask=None, random_state=None, output_device=torch.device("cuda")):
    ## TODO crosstalk and mixed training
    if(crosstalk_factor > 0):
        logging.warn("Crosstalk is not supported. Crosstalk is ignored")
    if(mixedtraining_mask is not None):
        logging.warn("Mixed training is not Supported. Crosstalk and mixed training are ignored")
    assert gamma_noise_std >= 0, "[E] Gamma noise standard deviation must be non-negative"

    gamma = np.pi / (v_pi**2)
    voltage_quantizer = voltage_quantize_fn(v_bit=v_bit, v_pi=v_pi, v_max=v_max)
    decomposer = RealUnitaryDecomposerBatch()
    delta_list, phi_mat = decomposer.decompose(W)

    phi_list = upper_triangle_to_vector(phi_mat)
    v_list = phase_to_voltage(phi_list, gamma)

    if(weight_decay_rate > 1e-6 and learning_rate > 1e-6):
        apply_weight_decay(v_list, decay_rate=weight_decay_rate, learning_rate=learning_rate, mask=voltage_mask)

    v_list = voltage_quantizer(v_list)
    v_list = clip_to_valid_quantized_voltage(v_list, gamma, v_bit, v_max, wrap_around=True)

    if(clip_voltage == True):
        v_max = np.sqrt(2 * np.pi / gamma)
        upper_mask_1 = (v_list > upper_thres) & (v_list < (upper_thres + v_max)/2)
        upper_mask_2 = v_list >= (upper_thres + v_max)/2
        v_list[upper_mask_1] = upper_thres
        v_list[upper_mask_2] = 0

    if(gamma_noise_std > 1e-5):
        pool = Pool(2)
        # reconstruct unitary matrix with gamma noise
        gamma_with_noise = gen_gaussian_noise(v_list, noise_mean=gamma, noise_std=gamma_noise_std, trunc_range=(), random_state=random_state)
        # Must add noise to all voltages to model the correct noise distribution
        # gamma_with_noise[~voltage_mask] = gamma ### only add noise to masked voltages, avoid aggressive noise error?
        phi_list_n = voltage_to_phase(v_list, gamma_with_noise)
        phi_mat_n = vector_to_upper_triangle(phi_list_n)

        if(return_ori):
            phi_list = voltage_to_phase(v_list, gamma)
            phi_mat = vector_to_upper_triangle(phi_list)
            W_recon, W_recon_n = pool.map(lambda x: decomposer.reconstruct(delta_list=delta_list, phi_mat=x), [phi_mat, phi_mat_n])
        else:
            W_recon = None
            W_recon_n = decomposer.reconstruct(delta_list=delta_list, phi_mat=phi_mat_n)

        pool.close()
        return W_recon, W_recon_n
    else:
        # reconstruct unitary matrix without noise
        phi_list = voltage_to_phase(v_list, gamma)
        phi_mat = vector_to_upper_triangle(phi_list)

        W_recon = decomposer.reconstruct(delta_list=delta_list, phi_mat=phi_mat)
        W_recon_n = W_recon.clone()

        return W_recon, W_recon_n


def add_gamma_noise_to_unitary_cpu(W, v_pi=4.36, gamma_noise_std=0.002, output_device=torch.device("cuda")):
    assert 0 <= gamma_noise_std <= 1, "[E] Gamma noise standard diviation must be within [0, 1]"

    if(isinstance(W, np.ndarray)):
        W_numpy = W.copy().astype(np.float64)
    elif(isinstance(W, torch.Tensor)):
        W_numpy = W.detach().cpu().numpy().copy().astype(np.float64)
    else:
        assert 0, "[E] Array type not supported, must be numpy.ndarray or torch.Tensor"

    gamma = np.pi / (v_pi**2)

    batch_mode = len(W_numpy.shape) > 2
    if(batch_mode):
        decomposer = RealUnitaryDecomposerBatch()
        delta_list, phi_mat = decomposer.decompose_batch(W_numpy)
    else:
        decomposer = RealUnitaryDecomposer()
        delta_list, phi_mat = decomposer.decompose(W_numpy)
    phi_list = upper_triangle_to_vector_cpu(phi_mat)

    v_list = phase_to_voltage_cpu(phi_list, gamma)

    gamma_with_noise = add_gaussian_noise_cpu(np.zeros_like(
        v_list, dtype=np.float64), noise_mean=gamma, noise_std=gamma_noise_std, trunc_range=())

    # thres = np.percentile(v_list, 10)
    # thres2 = np.percentile(v_list, 0)
    # mask = (v_list > thres) ^ (v_list > thres2)
    # gamma_with_noise[~mask] = gamma

    phi_list_n = voltage_to_phase_cpu(v_list, gamma_with_noise)
    # phi_list_U_q = phi_list_U
    # phi_list_V_q = phi_list_V
    phi_mat_n = vector_to_upper_triangle_cpu(phi_list_n)
    if(batch_mode):
        W_recon = decomposer.reconstruct_2_batch(delta_list, phi_mat_n)
    else:
        W_recon = decomposer.reconstruct_2(delta_list, phi_mat_n)

    W_recon = torch.from_numpy(W_recon).to(torch.float32).to(output_device)

    # res = check_equal_tensor(W, W_recon)
    # print("[I] checkEqual: ", res)
    # print("S:", Sigma)
    # print("delta_U:", delta_list_U)
    # print("delta_V:", delta_list_V)
    # print("W:", W)
    # print("W_rec:", W_recon)

    # bin=2**v_bit
    # plt.figure()
    # plt.subplot(2,1,1)
    # plt.hist(W_numpy.reshape([-1]), bins=bin, range=[-1, 1])
    # plt.subplot(2,1,2)
    # plt.hist(W_recon.detach().cpu().numpy().reshape([-1]), bins=bin, range=[-1, 1])

    # plt.figure()
    # plt.subplot(2,1,1)
    # plt.hist(phi_list_U.reshape([-1]), bins=bin, range=[-np.pi, np.pi])
    # plt.subplot(2,1,2)
    # plt.hist(phi_list_U_q.reshape([-1]), bins=bin, range=[-np.pi, np.pi])

    # v_max = np.sqrt(2*np.pi/gamma)
    # plt.figure()
    # plt.subplot(2,1,1)
    # plt.hist(v_list_U.reshape([-1]), bins=bin, range=[0, v_max])
    # plt.subplot(2,1,2)
    # plt.hist(v_list_U_q.reshape([-1]), bins=bin, range=[0, v_max])
    # plt.show()

    return W_recon


def add_phase_noise_to_unitary_cpu(W, phase_noise_std=0.002, protect_mask=None, output_device=torch.device("cuda")):
    assert 0 <= phase_noise_std <= 1, "[E] Phase noise standard diviation must be within [0, 1]"

    if(isinstance(W, np.ndarray)):
        W_numpy = W.copy().astype(np.float64)
    elif(isinstance(W, torch.Tensor)):
        W_numpy = W.detach().cpu().numpy().copy().astype(np.float64)
    else:
        assert 0, "[E] Array type not supported, must be numpy.ndarray or torch.Tensor"

    batch_mode = len(W_numpy.shape) > 2
    if(batch_mode):
        decomposer = RealUnitaryDecomposerBatch()
        delta_list, phi_mat = decomposer.decompose_batch(W_numpy)
    else:
        decomposer = RealUnitaryDecomposer()
        delta_list, phi_mat = decomposer.decompose(W_numpy)
    phi_list = upper_triangle_to_vector_cpu(phi_mat)

    phi_list_n = add_gaussian_noise_cpu(
        phi_list, noise_mean=0, noise_std=phase_noise_std, trunc_range=())

    if(protect_mask is not None):
        phi_list_n[protect_mask] = phi_list[protect_mask]

    phi_mat_n = vector_to_upper_triangle_cpu(phi_list_n)
    if(batch_mode):
        W_recon = decomposer.reconstruct_2_batch(delta_list, phi_mat_n)
    else:
        W_recon = decomposer.reconstruct_2(delta_list, phi_mat_n)

    W_recon = torch.from_numpy(W_recon).to(torch.float32).to(output_device)

    # res = check_equal_tensor(W, W_recon)
    # print("[I] checkEqual: ", res)
    # print("S:", Sigma)
    # print("delta_U:", delta_list_U)
    # print("delta_V:", delta_list_V)
    # print("W:", W)
    # print("W_rec:", W_recon)

    # bin=2**v_bit
    # plt.figure()
    # plt.subplot(2,1,1)
    # plt.hist(W_numpy.reshape([-1]), bins=bin, range=[-1, 1])
    # plt.subplot(2,1,2)
    # plt.hist(W_recon.detach().cpu().numpy().reshape([-1]), bins=bin, range=[-1, 1])

    # plt.figure()
    # plt.subplot(2,1,1)
    # plt.hist(phi_list_U.reshape([-1]), bins=bin, range=[-np.pi, np.pi])
    # plt.subplot(2,1,2)
    # plt.hist(phi_list_U_q.reshape([-1]), bins=bin, range=[-np.pi, np.pi])

    # v_max = np.sqrt(2*np.pi/gamma)
    # plt.figure()
    # plt.subplot(2,1,1)
    # plt.hist(v_list_U.reshape([-1]), bins=bin, range=[0, v_max])
    # plt.subplot(2,1,2)
    # plt.hist(v_list_U_q.reshape([-1]), bins=bin, range=[0, v_max])
    # plt.show()

    return W_recon


class voltage_quantize_prune_with_gamma_noise_of_unitary_fn(torch.nn.Module):
    def __init__(self,
        v_bit,
        v_pi,
        v_max,
        gamma_noise_std=0,
        decay_rate=0,
        learning_rate=1e-3,
        clip_voltage=False,
        lower_thres=0,
        upper_thres=10,
        prune_mask=None,
        random_state=None,
        device=torch.device("cuda")
    ):
        '''
        description: Perform quantization, clipping, decay, pruning, and gamma noise injection in the decomposed domain of a unitary matrix (or a batch of unitary matrices). LTE is used for gradient BP.
        return {Tuple} quantized unitary and quantized uniatry with noises
        '''
        super().__init__()
        self.v_bit = v_bit
        self.v_pi = v_pi
        self.v_max = v_max
        self.gamma_noise_std = gamma_noise_std
        self.decay_rate = decay_rate
        self.learning_rate = learning_rate
        self.clip_voltage = clip_voltage
        self.lower_thres = lower_thres
        self.upper_thres = upper_thres
        self.prune_mask = prune_mask
        self.random_state = random_state
        self.device = device

    def set_gamma_noise(self, noise_std, random_state=None):
        self.gamma_noise_std = noise_std
        self.random_state = random_state

    def forward(self, x):
        if(self.random_state is not None):
            set_torch_deterministic(self.random_state)
        U_q, U_qn = QuantizationWithNoiseOfUnitary.apply(
            x,
            self.v_bit,
            self.v_pi,
            self.v_max,
            1,
            self.gamma_noise_std,
            self.decay_rate,
            self.learning_rate,
            self.clip_voltage,
            self.lower_thres,
            self.upper_thres,
            self.prune_mask,
            self.random_state,
            self.device
        )
        return U_q, U_qn


class QuantizationWithNoiseOfUnitary(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, v_bit, v_pi, v_max, lambda3, gamma_noise_std, decay_rate, learning_rate, clip_voltage, lower_thres, upper_thres, prune_mask, crosstalk_factor, mixedtraining_mask, random_state, output_device):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        ctx.save_for_backward(input, prune_mask)

        output_q, output_qn = conditional_update_voltage_of_unitary_cpu(
            W=input,
            v_bit=v_bit,
            v_pi=v_pi,
            v_max=v_max,
            lambda3=lambda3,
            voltage_mask=None,
            voltage_backup=None,
            gamma_noise_std=gamma_noise_std,
            weight_decay_rate=decay_rate,
            learning_rate=learning_rate,
            clip_voltage=clip_voltage,
            lower_thres=lower_thres,
            upper_thres=upper_thres,
            return_ori=True,
            crosstalk_factor=crosstalk_factor,
            mixedtraining_mask=mixedtraining_mask,
            random_state=random_state,
            output_device=output_device)
        output_q[prune_mask, :, :], output_qn[prune_mask, :, :] = 0, 0
        return output_q.data, output_qn

    @staticmethod
    def backward(ctx, grad_output_q, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        input, prune_mask = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[prune_mask, :, :] = 0
        mean, sigma = grad_input.mean(), grad_input.std()
        grad_input = grad_input.clamp_(mean-3*sigma, mean+3*sigma)
        return grad_input, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None


def usv(U, S, V):
    '''
    description: Inverse SVD which builds matrix W from decomposed uintary matrices. Batched operation is supported\\
    U {torch.Tensor or np.ndarray} Square unitary matrix [..., M, M]\\
    S {torch.Tensor or np.ndarray} Diagonal vector [..., min(M, N)]\\
    V {torch.Tensor or np.ndarray} Square transposed unitary matrix [..., N, N]\\
    return W {torch.Tensor or np.ndarray} constructed MxN matrix [..., M, N]
    '''
    if(isinstance(U, torch.Tensor)):
        if(U.size(-1) == V.size(-1)):
            W = torch.matmul(U, S.unsqueeze(-1)*V)
        elif(U.size(-1) > V.size(-1)):
            W = torch.matmul(U[..., :V.size(-1)], S.unsqueeze(-1)*V)
        else:
            W = torch.matmul(U*S.unsqueeze(-2), V[..., :U.size(-1), :])
    elif(isinstance(U, np.ndarray)):
        if(U.shape[-1] == V.shape[-1]):
            W = np.matmul(U, S[..., np.newaxis]*V)
        elif(U.shape[-1] > V.shape[-1]):
            W = np.matmul(U[..., :V.shape[-1]], S[..., np.newaxis]*V)
        else:
            W = np.matmul(U * S[..., np.newaxis,:], V[..., :U.shape[-1], :])
    else:
        raise NotImplementedError
    return W


class voltage_quantize_prune_with_gamma_noise_of_matrix_fn(torch.nn.Module):
    def __init__(self,
        v_bit,
        v_pi,
        v_max,
        gamma_noise_std=0,
        decay_rate=0,
        learning_rate=1e-3,
        clip_voltage=False,
        lower_thres=0,
        upper_thres=10,
        prune_mask=None,
        random_state=None,
        device=torch.device("cuda")
    ):
        '''
        description: Perform quantization, clipping, decay, pruning, and gamma noise injection in the decomposed domain of a unitary matrix (or a batch of unitary matrices). LTE is used for gradient BP.
        return {Tuple} quantized unitary and quantized uniatry with noises
        '''
        super().__init__()
        self.v_bit = v_bit
        self.v_pi = v_pi
        self.v_max = v_max
        self.gamma_noise_std = gamma_noise_std
        self.decay_rate = decay_rate
        self.learning_rate = learning_rate
        self.clip_voltage = clip_voltage
        self.lower_thres = lower_thres
        self.upper_thres = upper_thres
        self.prune_mask = prune_mask
        self.random_state = random_state
        self.device = device

    def set_gamma_noise(self, noise_std, random_state=None):
        self.gamma_noise_std = noise_std
        self.random_state = random_state

    def set_crosstalk_factor(self, crosstalk_factor):
        self.phase_quantizer.set_crosstalk_factor(crosstalk_factor)

    def forward(self, x, mixedtraining_mask):
        ### svd is differentiable here, we want V* as numpy does
        U, S, V = torch.svd(x, some=False)
        V = V.t()
        if(self.random_state is not None):
            set_torch_deterministic(self.random_state)
        U_q, U_qn = QuantizationWithNoiseOfUnitary.apply(
            U,
            self.v_bit,
            self.v_pi,
            self.v_max,
            1,
            self.gamma_noise_std,
            self.decay_rate,
            self.learning_rate,
            self.clip_voltage,
            self.lower_thres,
            self.upper_thres,
            self.prune_mask,
            self.crosstalk_factor,
            mixedtraining_mask,
            self.random_state,
            self.device
        )
        if(self.random_state is not None):
            set_torch_deterministic(self.random_state)
        V_q, V_qn = QuantizationWithNoiseOfUnitary.apply(
            V,
            self.v_bit,
            self.v_pi,
            self.v_max,
            1,
            self.gamma_noise_std,
            self.decay_rate,
            self.learning_rate,
            self.clip_voltage,
            self.lower_thres,
            self.upper_thres,
            self.prune_mask,
            self.crosstalk_factor,
            mixedtraining_mask,
            self.random_state,
            self.device
        )
        W_qn = usv(U_qn, S, V_qn)
        return None, W_qn


class voltage_quantize_prune_with_gamma_noise_of_diag_fn(torch.nn.Module):
    def __init__(self,
        v_bit,
        v_pi,
        v_max,
        S_scale=3,
        gamma_noise_std=0,
        random_state=None,
        device=torch.device("cuda")
    ):
        '''
        description: Perform quantization and gamma noise injection in the decomposed domain of a diagonal vector of a matrix (or a batch of diagonals). LTE is used for gradient BP.
        return {Tuple} quantized diagonal(s) with noises
        '''
        super().__init__()
        self.v_bit = v_bit
        self.v_pi = v_pi
        self.v_max = v_max
        self.gamma = np.pi / v_pi**2
        self.S_scale = S_scale
        self.gamma_noise_std = gamma_noise_std

        self.phase_quantizer = phase_quantize_fn(v_bit, v_pi, v_max, gamma_noise_std, random_state=random_state)

        self.device = device

    def set_gamma_noise(self, noise_std, random_state=None):
        self.phase_quantizer.set_gamma_noise(noise_std, random_state)

    def set_crosstalk_factor(self, crosstalk_factor):
        self.phase_quantizer.set_crosstalk_factor(crosstalk_factor)

    def forward(self, x, mixedtraining_mask=None):
        phase = x.clamp(min=-self.S_scale, max=self.S_scale).mul(1/self.S_scale).arccos()
        phase = self.phase_quantizer(phase, mixedtraining_mask)
        S = phase.cos().mul(self.S_scale)
        return S


class PhaseQuantizer(torch.nn.Module):
    def __init__(self,
        bit,
        phase_noise_std=0,
        random_state=None,
        device=torch.device("cuda")) -> None:
        """2021/02/18: Phase quantization, considering the MZI architecture used in [David AB Miller, Optica'20]. Quantization of phases honoring hardware constraints in theta and phi. Not differentiable.
        Args:
            bit (int): bitwidth
            phase_onise_std (float, optional): std dev of Gaussian phase noise. Defaults to 0.
            random_state (None or int, optional): random_state for noise injection. Defaults to None.
            device (torch.Device, optional): torch.Device. Defaults to torch.device("cuda").
        """
        super().__init__()
        self.bit = bit
        self.phase_min = (0.5**(bit-2)-0.5)*np.pi
        self.phase_max = (1.5-0.5**(bit-1))*np.pi
        self.phase_range = self.phase_max - self.phase_min
        self.phase_noise_std = phase_noise_std
        self.random_state = random_state
        self.device = device

        self.quantizer = uniform_quantize(bit, gradient_clip=True)

    def set_phase_noise_std(self, noise_std=0, random_state=None):
        self.phase_noise_std = noise_std
        self.random_state = random_state

    def forward(self, x):
        x = x % (2*np.pi)
        mask = x > (1.5*np.pi)
        x[mask] -= 2 * np.pi
        x.clamp_(self.phase_min, self.phase_max)
        ratio = self.phase_range / (2**self.bit-1)
        x.add_(-self.phase_min).div_(ratio).round_().mul_(ratio).add_(self.phase_min)
        if(self.phase_noise_std > 1e-5):
            x.add_(gen_gaussian_noise(x, 0, self.phase_noise_std, trunc_range=(-2*self.phase_noise_std,2*self.phase_noise_std), random_state=self.random_state))
        return x


def diagonal_quantize_function(x, bit, phase_noise_std=0, random_state=None, gradient_clip=False):
    class DiagonalQuantizeFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            ### support batched diagonals. input is not a matrix, but a vector which is the diagonal entries.
            S_scale = x.abs().max(dim=-1, keepdim=True)[0]
            x = (x / S_scale).acos() # phase after acos is from [0, pi]
            ratio = np.pi / (2**bit-1)
            x.div_(ratio).round_().mul_(ratio)
            if(phase_noise_std > 1e-5):
                noise = gen_gaussian_noise(x, noise_mean=0, noise_std=phase_noise_std, trunc_range=[-2*phase_noise_std, 2*phase_noise_std], random_state=random_state)
                x.add_(noise)
            x.cos_().mul_(S_scale)
            return x

        @staticmethod
        def backward(ctx, grad_output):
            if(gradient_clip):
                grad_input = grad_output.clamp(-1, 1)
            else:
                grad_input = grad_output.clone()
            return grad_input
    return DiagonalQuantizeFunction.apply(x)


class DiagonalQuantizer(torch.nn.Module):
    def __init__(self,
        bit,
        phase_noise_std=0.0,
        random_state=None,
        device=torch.device("cuda")
    ):
        """2021/02/18: New phase quantizer for Sigma matrix in MZI-ONN. Gaussian phase noise is supported. All singular values are normalized by a TIA gain (S_scale), the normalized singular values will be achieved by cos(phi), phi will have [0, pi] uniform quantization.
        We do not consider real MZI implementation, thus voltage quantization and gamma noises are not supported.
        Args:
            bit (int): bitwidth for phase quantization.
            phase_noise_std (float, optional): Std dev for Gaussian phase noises. Defaults to 0.
            random_state (int, optional): random_state to control random noise injection. Defaults to None.
            device (torch.Device, optional): torch.Device. Defaults to torch.device("cuda").
        """

        super().__init__()
        self.bit = bit
        self.phase_noise_std = phase_noise_std
        self.random_state = random_state
        self.device = device

    def set_phase_noise_std(self, phase_noise_std=0, random_state=None):
        self.phase_noise_std = phase_noise_std
        self.random_state = random_state

    def forward(self, x):
        ### support batched diagonals. input is not a matrix, but a vector which is the diagonal entries.
        ### this function is differentiable
        x = diagonal_quantize_function(
                x,
                self.bit,
                self.phase_noise_std,
                self.random_state,
                gradient_clip=True)

        return x



class UnitaryQuantizer(torch.nn.Module):
    def __init__(self,
        bit,
        phase_noise_std=0.0,
        random_state=None,
        alg="reck",
        mode="phase",
        device=torch.device("cuda")
    ):
        """2021/02/18: New phase quantizer for Uintary matrix in MZI-ONN. Gaussian phase noise is supported. The quantization considers the MZI implementation in [David AB Miller, Optica'20], but voltage quantization and gamma noises are not supported.
        Args:
            bit (int): bitwidth for phase quantization.
            phase_noise_std (float, optional): Std dev for Gaussian phase noises. Defaults to 0.
            random_state (int, optional): random_state to control random noise injection. Defaults to None.
            device (torch.Device, optional): torch.Device. Defaults to torch.device("cuda").
        """

        super().__init__()
        self.bit = bit
        self.phase_noise_std = phase_noise_std
        self.random_state = random_state
        self.alg = alg
        self.decomposer = RealUnitaryDecomposerBatch(alg=alg)
        self.quantizer = PhaseQuantizer(bit, device=device)
        if(alg == "reck"):
            self.decomposer.m2v = upper_triangle_to_vector
            self.decomposer.v2m = vector_to_upper_triangle
        elif(alg == "clements"):
            self.decomposer.m2v = checkerboard_to_vector
            self.decomposer.v2m = vector_to_checkerboard
        else:
            raise NotImplementedError
        self.device = device
        self.phase_min = (0.5**(bit-2)-0.5)*np.pi
        self.phase_max = (1.5-0.5**(bit-1))*np.pi
        self.phase_range = self.phase_max - self.phase_min

    def set_phase_noise_std(self, phase_noise_std=0, random_state=None):
        self.phase_noise_std = phase_noise_std
        self.random_state = random_state
        self.quantizer.set_phase_noise_std(phase_noise_std, random_state)

    def forward(self, x):
        ### this function is not differentiable
        delta_list, x = self.decomposer.decompose(x.data.clone())
        x = self.decomposer.m2v(x)
        x = self.quantizer(x)
        x = self.decomposer.reconstruct(delta_list, self.decomposer.v2m(x))

        return x



class ThermalCrosstalkSimulator(object):
    def __init__(self,
                 heat_source_interval=8,  # interval bet/ heat source (um)
                 # SetPad=0,
                 grid_precision=10,  # um
                 power_density_multipier=1e-3,
                 # W/(um K) thermal conductivity
                 thermal_conductivity=1.4e-6,
                 max_iter=2000,      # max # of iterations
                 # material options
                 boundary_cond=False,
                 # plotting options
                 plotting=True,
                 display_iter=10,
                 hold_time=0.00001,
                 gaussian_filter_size=3,
                 gaussian_filter_std=0.33,
                 device=torch.device("cuda:0")):
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
        self.gaussian_filter_size = gaussian_filter_size
        self.gaussian_filter_std = gaussian_filter_std
        self.device = device
        self.power_density = None


        # self.init_phase_distribution(self.phases)
        self.init_gaussian_filter(gaussian_filter_size, gaussian_filter_std)
        self.mixedtraining_mask = None

    def init_gaussian_filter(self, gaussian_filter_size, gaussian_filter_std):
        self.gaussian_filter = gen_gaussian_filter2d(size=self.gaussian_filter_size, std=gaussian_filter_std, device=self.device).unsqueeze(0).unsqueeze(0)
        self.gaussian_filter_zero_center = self.gaussian_filter.clone()
        self.gaussian_filter_zero_center[0,0,gaussian_filter_size//2, gaussian_filter_size//2] = 0

    def init_phase_distribution(self, phases, dim):
        self.power_density = np.zeros([self.heat_source_interval * dim, self.heat_source_interval * dim])
        cnt = 0
        # for i in range(1, dim):
        #     for j in range(1, dim - i + 1):
        #         self.power_density[self.heat_source_interval*i, self.heat_source_interval*j] = phases[cnt]
        #         cnt = cnt + 1
        pointer = 0
        for i in range(1, dim):
            number_of_sources = dim - i
            interval = self.heat_source_interval
            self.power_density[interval * i, interval:number_of_sources * interval+1:interval] = phases[pointer:pointer+number_of_sources]
            pointer += number_of_sources

    def simulate(self, phases, dim):
        self.init_phase_distribution(phases, dim)
        nx = self.power_density.shape[0] #*SetSpace      # number of steps in x
        ny = self.power_density.shape[1] #*SetSpace   # number of steps in y
        dx = self.grid_precision           #nx/(nx-1) # width of step
        dy = self.grid_precision           #ny/(ny-1) # width of step

        # Initial Conditions
        p = torch.zeros((1, 1, nx, ny)).float().to(self.device)
        power_density = (torch.from_numpy(self.power_density.copy()).unsqueeze(0).unsqueeze(0)*dx*dx*dy*dy*self.thermal_conductivity / (2*(dx*dx+dy*dy))).float().to(self.device)
        kernel = torch.from_numpy(np.array([[0,    dy*dy,     0],
                                            [dx*dx,  0  , dx*dx],
                                            [0,    dy*dy,    0]], dtype=np.float32)) / (2*(dx*dx+dy*dy))
        kernel = kernel.unsqueeze(0).unsqueeze(0).to(self.device)
        mask = torch.zeros(nx, ny, dtype=torch.float32, device=self.device)
        for row in range(1, nx-2):
            mask[row,1:ny-row-1] = 1

        conv_err = []
        if self.plotting is True:
            plt.ion() # continuous SetPlotting
            fig = plt.figure()
            ax = fig.gca(projection='3d')
            x = np.linspace(dx/2, nx - dx/2, nx)
            y = np.linspace(dy/2, ny - dy/2, ny)#orig no setspace
            X, Y = np.meshgrid(x,y)

        for it in range(self.max_iter+1):
            # print(f"[I] iteration: {it}")
            out = torch.nn.functional.conv2d(p, kernel, padding=(1,1))
            out.add_(power_density).mul_(mask)

            conv_err.append((it, (out-p).abs().max().data.item()))
            p = out

            if self.plotting is True and it%(self.display_iter)== 0:
                surf = ax.plot_surface(X, Y, p.squeeze(0).squeeze(0).numpy(), cmap=cm.rainbow, linewidth=0, antialiased = False)
                #ax.set_zlim(0,80)
                #ax.set_xlim(0,0.1)
                #ax.set_ylim(0,0.1)
                plt.title('it#%d' %it, y = 1)
                ax.set_xlabel('Distance (x%d um)'%(self.grid_precision))
                ax.set_ylabel('Distance (x%d um)'%(self.grid_precision))
                ax.set_zlabel('Temperature (C)')
                # for tick in ax.xaxis.get_major_ticks():
                #     tick.label.set_fontsize(80)
                # for tick in ax.yaxis.get_major_ticks():
                #     tick.label.set_fontsize(80)

                plt.show()
                plt.pause(self.hold_time)

        return p.cpu().numpy().astype(np.float64)

    def set_crosstalk_factor(self, crosstalk_factor):
        self.gaussian_filter_std = crosstalk_factor
        self.init_gaussian_filter(self.gaussian_filter_size, crosstalk_factor)

    def simple_simulate_triangle(self, phases, mixedtraining_mask):
        phases = phases % (2 * np.pi)
        if(mixedtraining_mask is None):
            phase_mat = vector_to_upper_triangle(phases).unsqueeze(0).unsqueeze(0)
            gaussian_filter = self.gaussian_filter
            padding = self.gaussian_filter_size // 2
            phase_mat = torch.nn.functional.conv2d(phase_mat, gaussian_filter, padding=(padding, padding))
            phase_mat = phase_mat[0,0,...]
            phase_list = upper_triangle_to_vector(phase_mat)
        else:
            ### only active devices marked as 1/True in the mixed training mask will influcence others
            ### poassive devices will be influenced by active devices, but will not incluence others
            phase_mat_active = vector_to_upper_triangle(phases * mixedtraining_mask.float()).unsqueeze(0).unsqueeze(0)
            gaussian_filter = self.gaussian_filter_zero_center
            padding = self.gaussian_filter_size // 2
            ### influence map
            phase_mat_active = torch.nn.functional.conv2d(phase_mat_active, gaussian_filter, padding=(padding, padding))
            ### add influence map and original phases together
            phase_list = upper_triangle_to_vector(phase_mat_active[0,0,...]) + phases

        return phase_list

    def simple_simulate_diagonal(self, phases, mixedtraining_mask):
        return phases

    def simple_simulate_butterfly(self, phases, mixedtraining_mask):
        phases = phases % (2 * np.pi)
        ## [n_level, k/2, 2]
        size = phases.size()


        if(mixedtraining_mask is None):
            phases = phases.view([1,1] + list(size)[:-2] + [phases.size(-1)*phases.size(-2)]) # [1, 1, n_level, k]
            gaussian_filter = self.gaussian_filter
            padding = self.gaussian_filter_size // 2
            phases = torch.nn.functional.conv2d(phases, gaussian_filter, padding=(padding, padding))
            phases = phases.view(size)

        else:
            ### only active devices marked as 1/True in the mixed training mask will influcence others
            ### poassive devices will be influenced by active devices, but will not incluence others

            phases_active = phases * mixedtraining_mask.float()
            gaussian_filter = self.gaussian_filter_zero_center
            padding = self.gaussian_filter_size // 2
            ### influence map
            phases_active = torch.nn.functional.conv2d(phases_active.view([1,1] + list(size)[:-2] + [phases.size(-1)*phases.size(-2)]), gaussian_filter, padding=(padding, padding))
            ### add influence map and original phases together
            phases = phases_active.view_as(phases) + phases

        return phases


    def simple_simulate(self, phases, mixedtraining_mask=None, mode="triangle"):
        assert mode in {"triangle", "diagonal", "butterfly"}, logging.error(f"Only support triangle, diagonal, and butterfly. But got {mode}")
        if(mode == "triangle"):
            return self.simple_simulate_triangle(phases, mixedtraining_mask)
        elif(mode == "diagonal"):
            return self.simple_simulate_diagonal(phases, mixedtraining_mask)
        elif(mode == "butterfly"):
            return self.simple_simulate_butterfly(phases, mixedtraining_mask)
        else:
            return phases


if __name__ == "__main__":
    print(gen_gaussian_filter2d(3, std=0.33))
    exit(0)
    mat = np.random.normal(size=(5,5))
    print(mat)
    print(checkerboard_to_vector(mat))
    exit(0)
    checkerboard_to_vector(mat)
    a = np.ones((512,512))
    b = np.ones([(1+511)*511//2])
    with TimerCtx() as t:
        for _ in range(100):
            # b = upper_triangle_to_vector_cpu(a)
            a = vector_to_upper_triangle_cpu(b)
    print(t.interval)
