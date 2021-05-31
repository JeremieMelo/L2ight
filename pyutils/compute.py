##########################
#       computation      #
##########################
import logging
from functools import lru_cache
from typing import List, Optional

import numpy as np
import torch
from scipy.stats import truncnorm
from torch.autograd import grad
from torch.tensor import Tensor
from torch.types import Device, _size

from .torch_train import set_torch_deterministic

__all__ = ["shift", "Krylov", "circulant", "toeplitz", "complex_circulant", "complex_mult", "expi", "complex_matvec_mult", "complex_matmul", "real_to_complex", "get_complex_magnitude", "get_complex_energy", "complex_to_polar", "polar_to_complex", "absclamp", "absclamp_", "im2col_2d", "check_identity_matrix",
           "check_unitary_matrix", "check_equal_tensor", "batch_diag", "batch_eye_cpu", "batch_eye", "merge_chunks", "partition_chunks", "clip_by_std", "percentile", "gen_boolean_mask_cpu", "gen_boolean_mask", "fftshift_cpu", "ifftshift_cpu", "gen_gaussian_noise", "gen_gaussian_filter2d_cpu", "gen_gaussian_filter2d", "add_gaussian_noise_cpu", "add_gaussian_noise", "add_gaussian_noise_", "circulant_multiply", "calc_diagonal_hessian", "calc_jacobian", "polynomial", "gaussian", "lowrank_decompose", "get_conv2d_flops"]


def shift(v, f=1):
    return torch.cat((f * v[..., -1:], v[..., :-1]), dim=-1)


def Krylov(linear_map, v, n=None):
    if n is None:
        n = v.size(-1)
    cols = [v]
    for _ in range(n - 1):
        v = linear_map(v)
        cols.append(v)
    return torch.stack(cols, dim=-2)


def circulant(eigens):
    circ = Krylov(shift, eigens).transpose(-1, -2)  # .t()
    return circ


@lru_cache(maxsize=4)
def _get_toeplitz_indices(n, device):
    # cached toeplitz indices. avoid repeatedly generate the indices.
    indices = circulant(torch.arange(n, device=device))
    return indices


def toeplitz(col):
    '''
    Efficient Toeplitz matrix generation from the first column. The column vector must in the last dimension. Batch generation is supported. Suitable for AutoGrad. Circulant matrix multiplication is ~4x faster than rfft-based implementation!\\
    @col {torch.Tensor} (Batched) column vectors.\\
    return out {torch.Tensor} (Batched) circulant matrices
    '''
    n = col.size(-1)
    indices = _get_toeplitz_indices(n, device=col.device)
    return col[..., indices]


def complex_circulant(eigens):
    circ = Krylov(shift, eigens).transpose(-1, -2)  # .t()
    return circ


def complex_mult(X, Y):
    assert X.shape[-1] == 2 and Y.shape[-1] == 2, 'Last dimension must be 2'
    if(hasattr(torch, "view_as_complex")):
        return torch.view_as_real(torch.view_as_complex(X) * torch.view_as_complex(Y))
    return torch.stack(
        (X[..., 0] * Y[..., 0] - X[..., 1] * Y[..., 1],
         X[..., 0] * Y[..., 1] + X[..., 1] * Y[..., 0]),
        dim=-1)


def complex_matvec_mult(W, X):
    return torch.sum(complex_mult(W, X.unsqueeze(0).repeat(W.size(0), 1, 1)), dim=1)


def complex_matmul(X, Y):
    assert X.shape[-1] == 2 and Y.shape[-1] == 2, 'Last dimension must be 2'
    if(torch.__version__ >= "1.8" or (torch.__version__ >= "1.7" and X.shape[:-3] == Y.shape[:-3])):
        return torch.view_as_real(torch.matmul(torch.view_as_complex(X), torch.view_as_complex(Y)))

    return torch.stack([X[..., 0].matmul(Y[..., 0]) - X[..., 1].matmul(Y[..., 1]), X[..., 0].matmul(Y[..., 1]) + X[..., 1].matmul(Y[..., 0])], dim=-1)


def expi(x):
    if(torch.__version__ >= "1.8" or (torch.__version__ >= "1.7" and not x.requires_grad)):
        return torch.exp(1j*x)
    else:
        return x.cos().type(torch.cfloat) + 1j*x.sin().type(torch.cfloat)


def real_to_complex(x):
    if(torch.__version__ < "1.7"):
        return torch.stack((x, torch.zeros_like(x).to(x.device)), dim=-1)
    else:
        return torch.view_as_real(x.to(torch.complex64))


def get_complex_magnitude(x):
    assert x.size(-1) == 2, "[E] Input must be complex Tensor"
    return torch.sqrt(x[..., 0] * x[..., 0] + x[..., 1] * x[..., 1])


def complex_to_polar(x):
    # real and imag to magnitude and angle
    if(isinstance(x, torch.Tensor)):
        mag = x.norm(p=2, dim=-1)
        angle = torch.view_as_complex(x).angle()
        x = torch.stack([mag, angle], dim=-1)
    elif(isinstance(x, np.ndarray)):
        x = x.astype(np.complex64)
        mag = np.abs(x)
        angle = np.angle(x)
        x = np.stack([mag, angle], axis=-1)
    else:
        raise NotImplementedError
    return x


def polar_to_complex(mag, angle):
    # magnitude and angle to real and imag
    if(angle is None):
        return real_to_complex(angle)
    if(mag is None):
        if(isinstance(angle, torch.Tensor)):
            x = torch.stack([angle.cos(), angle.sin()], dim=-1)
        elif(isinstance(angle, np.ndarray)):
            x = np.stack([np.cos(angle), np.sin(angle)], axis=-1)
        else:
            raise NotImplementedError
    else:
        if(isinstance(angle, torch.Tensor)):
            x = torch.stack([mag * angle.cos(), mag * angle.sin()], dim=-1)
        elif(isinstance(angle, np.ndarray)):
            x = np.stack([mag * np.cos(angle), mag * np.sin(angle)], axis=-1)
        else:
            raise NotImplementedError
    return x


def get_complex_energy(x):
    assert x.size(-1) == 2, "[E] Input must be complex Tensor"
    return x[..., 0] * x[..., 0] + x[..., 1] * x[..., 1]


def absclamp(x, min=None, max=None):
    if(isinstance(x, torch.Tensor)):
        mag = x.norm(p=2, dim=-1).clamp(min=min, max=max)
        angle = torch.view_as_complex(x).angle()
        x = polar_to_complex(mag, angle)
    elif(isinstance(x, np.ndarray)):
        x = x.astype(np.complex64)
        mag = np.clip(np.abs(x), a_min=min, a_max=max)
        angle = np.angle(x)
        x = polar_to_complex(mag, angle)
    else:
        raise NotImplementedError
    return x


def absclamp_(x, min=None, max=None):
    if(isinstance(x, torch.Tensor)):
        y = torch.view_as_complex(x)
        mag = y.abs().clamp(min=min, max=max)
        angle = y.angle()
        x.data.copy_(polar_to_complex(mag, angle))
    elif(isinstance(x, np.ndarray)):
        y = x.astype(np.complex64)
        mag = np.clip(np.abs(y), a_min=min, a_max=max)
        angle = np.angle(y)
        x[:] = polar_to_complex(mag, angle)
    else:
        raise NotImplementedError
    return x


def im2col_2d(W=None, X=None, stride=1, padding=0, w_size=None):
    if(W is not None):
        W_col = W.view(W.size(0), -1)
    else:
        W_col = None

    if(X is not None):
        n_filters, d_filter, h_filter, w_filter = W.size() if W is not None else w_size
        n_x, d_x, h_x, w_x = X.size()

        h_out = (h_x - h_filter + 2 * padding) / stride + 1
        w_out = (w_x - w_filter + 2 * padding) / stride + 1

        h_out, w_out = int(h_out), int(w_out)
        X_col = torch.nn.functional.unfold(X.view(
            1, -1, h_x, w_x), h_filter, dilation=1, padding=padding, stride=stride).view(n_x, -1, h_out*w_out)
        X_col = X_col.permute(1, 2, 0).contiguous().view(X_col.size(1), -1)
    else:
        X_col, h_out, w_out = None, None, None

    return W_col, X_col, h_out, w_out


def check_identity_matrix(W):
    if(isinstance(W, np.ndarray)):
        W_numpy = W.copy().astype(np.float64)
    elif(isinstance(W, torch.Tensor)):
        W_numpy = W.detach().cpu().numpy().copy().astype(np.float64)
    else:
        assert 0, "[E] Array type not supported, must be numpy.ndarray or torch.Tensor"

    return (W_numpy.shape[0] == W_numpy.shape[1]) and np.allclose(W_numpy, np.eye(W_numpy.shape[0]))


def check_unitary_matrix(W):
    if(isinstance(W, np.ndarray)):
        W_numpy = W.copy().astype(np.float64)
    elif(isinstance(W, torch.Tensor)):
        W_numpy = W.detach().cpu().numpy().copy().astype(np.float64)
    else:
        assert 0, "[E] Array type not supported, must be numpy.ndarray or torch.Tensor"
    M = np.dot(W_numpy, W_numpy.T)
    # print(M)
    return check_identity_matrix(M)


def check_equal_tensor(W1, W2):
    if(isinstance(W1, np.ndarray)):
        W1_numpy = W1.copy().astype(np.float64)
    elif(isinstance(W1, torch.Tensor)):
        W1_numpy = W1.detach().cpu().numpy().copy().astype(np.float64)
    else:
        assert 0, "[E] Array type not supported, must be numpy.ndarray or torch.Tensor"

    if(isinstance(W2, np.ndarray)):
        W2_numpy = W2.copy().astype(np.float64)
    elif(isinstance(W2, torch.Tensor)):
        W2_numpy = W2.detach().cpu().numpy().copy().astype(np.float64)
    else:
        assert 0, "[E] Array type not supported, must be numpy.ndarray or torch.Tensor"
    return (W1_numpy.shape == W2_numpy.shape) and np.allclose(W1_numpy, W2_numpy)


def batch_diag(x):
    # x[..., N, N] -> [..., N]
    assert len(
        x.shape) >= 2, f"At least 2-D array/tensor is expected, but got shape {x.shape}"
    if isinstance(x, np.ndarray):
        size = list(x.shape)
        x = x.reshape(size[:-2]+[size[-2]*size[-1]])
        x = x[..., ::size[-1]+1]
    elif isinstance(x, torch.Tensor):
        size = list(x.size())
        x = x.flatten(-2, -1)
        x = x[..., ::size[-1]+1]
    else:
        raise NotImplementedError
    return x


def batch_eye_cpu(N: int, batch_shape: List[int], dtype: np.dtype) -> np.ndarray:
    x = np.zeros(list(batch_shape)+[N, N], dtype=dtype)
    x.reshape(-1, N*N)[..., ::N+1] = 1
    return x


def batch_eye(N: int, batch_shape: List[int], dtype: torch.dtype, device: Device = torch.device("cuda")) -> torch.Tensor:
    x = torch.zeros(list(batch_shape)+[N, N], dtype=dtype, device=device)
    x.view(-1, N*N)[..., ::N+1] = 1
    return x


def merge_chunks(x, complex=False):
    # x = [H, W, B, B] or [H, W, B, B, 2]
    if(isinstance(x, torch.Tensor)):
        if(not complex):
            h, w, bs = x.size(0), x.size(1), x.size(2)
            x = x.permute(0, 2, 1, 3).contiguous()  # x = [h, bs, w, bs]
            x = x.view(h * bs, w * bs)
        else:
            h, w, bs = x.size(0), x.size(1), x.size(2)
            x = x.permute(0, 2, 1, 3, 4).contiguous()  # x = [h, bs, w, bs, 2]
            x = x.view(h * bs, w * bs, 2)

    elif(isinstance(x, np.ndarray)):
        if(not complex):
            h, w, bs = x.shape[0], x.shape[1], x.shape[2]
            x = np.transpose(x, [0, 2, 1, 3])
            x = np.reshape(x, [h * bs, w * bs])
        else:
            h, w, bs = x.shape[0], x.shape[1], x.shape[2]
            x = np.transpose(x, [0, 2, 1, 3, 4])
            x = np.reshape(x, [h * bs, w * bs, 2])
    else:
        raise NotImplementedError
    return x


def partition_chunks(x, bs, complex=False):
    # x = [H, W] or [H, W, 2]
    if(isinstance(x, torch.Tensor)):
        h, w = x.size(0), x.size(1)
        new_h, new_w = h // bs, w // bs
        if(not complex):
            x = x.view(new_h, bs, new_w, bs)  # x = (h // bs, bs, w // bs, bs)
            # (h // bs, w // bs, bs, bs)
            x = x.permute(0, 2, 1, 3).contiguous()
        else:
            # x = (h // bs, bs, w // bs, bs, 2)
            x = x.view(new_h, bs, new_w, bs, 2)
            # (h // bs, w // bs, bs, bs, 2)
            x = x.permute(0, 2, 1, 3, 4).contiguous()
    elif(isinstance(x, np.ndarray)):
        h, w = x.shape[0], x.shape[1]
        new_h, new_w = h // bs, w // bs
        if(not complex):
            x = np.reshape(x, [new_h, bs, new_w, bs])
            x = np.transpose(x, [0, 2, 1, 3])
        else:
            x = np.reshape(x, [new_h, bs, new_w, bs, 2])
            x = np.transpose(x, [0, 2, 1, 3, 2])
    else:
        raise NotImplementedError

    return x


def clip_by_std(x, n_std_neg=3, n_std_pos=3):
    if(isinstance(x, np.ndarray)):
        std = np.std(x)
        mean = np.mean(x)
        out = np.clip(x, a_min=mean-n_std_neg*std, a_max=mean+n_std_pos*std)
    elif(isinstance(x, torch.Tensor)):
        std = x.data.std()
        mean = x.data.mean()
        out = x.clamp(min=mean-n_std_neg*std, max=mean+n_std_pos*std)
    else:
        raise NotImplementedError
    return out


def percentile(t, q: float):
    """
    Return the ``q``-th percentile of the flattened input tensor's data.

    CAUTION:
     * Needs PyTorch >= 1.1.0, as ``torch.kthvalue()`` is used.
     * Values are not interpolated, which corresponds to
       ``numpy.percentile(..., interpolation="nearest")``.

    :param t: Input tensor.
    :param q: Percentile to compute, which must be between 0 and 100 inclusive.
    :return: Resulting value (scalar).
    """
    # Note that ``kthvalue()`` works one-based, i.e. the first sorted value
    # indeed corresponds to k=1, not k=0! Use float(q) instead of q directly,
    # so that ``round()`` returns an integer, even if q is a np.float32.
    if(isinstance(t, torch.Tensor)):
        k = 1 + round(.01 * float(q) * (t.numel() - 1))
        result = t.view(-1).kthvalue(k).values.item()
    elif(isinstance(t, np.ndarray)):
        result = np.percentile(t, q=q)
    else:
        raise NotImplementedError
    return result


def gen_boolean_mask_cpu(size, true_prob):
    assert 0 <= true_prob <= 1, f"[E] Wrong probability for True"
    return np.random.choice(a=[False, True], size=size, p=[1-true_prob, true_prob])


def gen_boolean_mask(size: _size, true_prob: float, random_state: Optional[int] = None, device: Device = torch.device("cuda")) -> Tensor:
    assert 0 <= true_prob <= 1, f"[E] Wrong probability for True"
    if(true_prob > 1 - 1e-9):
        return torch.ones(size, device=device, dtype=torch.bool)
    elif(true_prob < 1e-9):
        return torch.zeros(size, device=device, dtype=torch.bool)
    if(random_state is not None):
        with torch.random.fork_rng():
            torch.random.manual_seed(random_state)
            return torch.empty(size, dtype=torch.bool, device=device).bernoulli_(true_prob)
    else:
        return torch.empty(size, dtype=torch.bool, device=device).bernoulli_(true_prob)


def fftshift_cpu(x, batched=True, dim=None):
    if(isinstance(x, np.ndarray)):
        if(dim is None):
            if(batched):
                dim = tuple(range(1, len(x.shape)))
            else:
                dim = tuple(range(0, len(x.shape)))
        out = np.fft.fftshift(x, axes=dim)
    elif(isinstance(x, torch.Tensor)):
        device = x.device
        x = x.cpu().detach().numpy()
        if(dim is None):
            if(batched):
                dim = tuple(range(1, len(x.shape)))
            else:
                dim = tuple(range(0, len(x.shape)))
        out = np.fft.fftshift(x, axes=dim)
        out = torch.from_numpy(out).to(device)
    return out


def ifftshift_cpu(x, batched=True, dim=None):
    if(isinstance(x, np.ndarray)):
        if(dim is None):
            if(batched):
                dim = tuple(range(1, len(x.shape)))
            else:
                dim = tuple(range(0, len(x.shape)))
        out = np.fft.ifftshift(x, axes=dim)
    elif(isinstance(x, torch.Tensor)):
        device = x.device
        x = x.cpu().detach().numpy()
        if(dim is None):
            if(batched):
                dim = tuple(range(1, len(x.shape)))
            else:
                dim = tuple(range(0, len(x.shape)))
        out = np.fft.ifftshift(x, axes=dim)
        out = torch.from_numpy(out).to(device)
    return out


def gen_gaussian_noise(W, noise_mean=0, noise_std=0.002, trunc_range=(), random_state=None):
    if(random_state is not None):
        set_torch_deterministic(random_state)
    if(isinstance(W, np.ndarray)):
        if(not trunc_range):
            noises = np.random.normal(noise_mean, noise_std, W.shape)
        else:
            a = (trunc_range[0] - noise_mean) / noise_std
            b = (trunc_range[1] - noise_mean) / noise_std
            noises = truncnorm.rvs(
                a, b, loc=noise_mean, scale=noise_std, size=W.shape, random_state=None)
    elif(isinstance(W, torch.Tensor)):
        if(not trunc_range):
            noises = torch.zeros_like(W).normal_(
                mean=noise_mean, std=noise_std)
        else:

            size = W.shape
            tmp = W.new_empty(size + (4,)).normal_()
            a = (trunc_range[0] - noise_mean) / noise_std
            b = (trunc_range[1] - noise_mean) / noise_std
            valid = (tmp < b) & (tmp > a)
            ind = valid.max(-1, keepdim=True)[1]
            noises = tmp.gather(-1,
                                ind).squeeze(-1).mul_(noise_std).add_(noise_mean)
            # noises = truncated_normal(W, mean=noise_mean, std=noise_std, a=trunc_range[0], b=trunc_range[1])
    else:
        assert 0, logging.error(
            f"Array type not supported, must be numpy.ndarray or torch.Tensor, but got {type(W)}")
    return noises


def gen_gaussian_filter2d_cpu(size=3, std=0.286):
    assert size % 2 == 1, f"Gaussian filter can only be odd size, but size={size} is given."
    ax = np.linspace(-(size - 1) / 2., (size - 1) / 2., size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-0.5 / np.square(std) * (np.square(xx) + np.square(yy)))
    kernel = kernel / np.sum(kernel)
    kernel[size//2, size//2] = 1
    return kernel


def gen_gaussian_filter2d(size=3, std=0.286, device=torch.device("cuda")):
    assert size % 2 == 1, f"Gaussian filter can only be odd size, but size={size} is given."
    if(std > 1e-8):
        ax = torch.linspace(-(size - 1) / 2., (size - 1) /
                            2., size, dtype=torch.float32, device=device)
        xx, yy = torch.meshgrid(ax, ax)
        kernel = torch.exp(-0.5 / (std**2) * (xx.square() + yy.square()))
        kernel = kernel.div_(kernel.sum())
        kernel[size//2, size//2] = 1
    else:
        kernel = torch.zeros(size, size, dtype=torch.float32, device=device)
        kernel[size//2, size//2] = 1

    return kernel


def add_gaussian_noise(W, noise_mean=0, noise_std=0.002, trunc_range=(), random_state=None):
    noises = gen_gaussian_noise(W, noise_mean=noise_mean, noise_std=noise_std,
                                trunc_range=trunc_range, random_state=random_state)
    output = W + noises
    return output


def add_gaussian_noise_(W, noise_mean=0, noise_std=0.002, trunc_range=(), random_state=None):
    noises = gen_gaussian_noise(W, noise_mean=noise_mean, noise_std=noise_std,
                                trunc_range=trunc_range, random_state=random_state)
    if(isinstance(W, np.ndarray)):
        W += noises
    elif(isinstance(W, torch.Tensor)):
        W.data += noises
    else:
        assert 0, logging.error(
            f"Array type not supported, must be numpy.ndarray or torch.Tensor, but got {type(W)}")
    return W


def add_gaussian_noise_cpu(W, noise_mean=0, noise_std=0.002, trunc_range=()):
    if(isinstance(W, np.ndarray)):
        W_numpy = W.copy().astype(np.float64)
    elif(isinstance(W, torch.Tensor)):
        W_numpy = W.detach().cpu().numpy().copy().astype(np.float64)
    else:
        assert 0, "[E] Array type not supported, must be numpy.ndarray or torch.Tensor"
    if(not trunc_range):
        noises = np.random.normal(noise_mean, noise_std, W_numpy.shape)
    else:
        a = (trunc_range[0] - noise_mean) / noise_std
        b = (trunc_range[1] - noise_mean) / noise_std
        noises = truncnorm.rvs(
            a, b, loc=noise_mean, scale=noise_std, size=W_numpy.shape, random_state=None)
    return W_numpy + noises


def circulant_multiply(c, x):
    """ Multiply circulant matrix with first column c by x
    Parameters:
        c: (n, )
        x: (batch_size, n) or (n, )
    Return:
        prod: (batch_size, n) or (n, )
    """
    return torch.irfft(complex_mult(torch.rfft(c, 1), torch.rfft(x, 1)), 1, signal_sizes=(c.shape[-1], ))


def calc_diagonal_hessian(weight_dict, loss, model):
    model.zero_grad()
    hessian_dict = {}
    for name, weight in weight_dict.items():
        first_gradient = grad(loss, weight, create_graph=True)[0]
        second_gradient = grad(first_gradient.sum(),
                               weight, create_graph=True)[0]
        hessian_dict[name] = second_gradient.clone()
    model.zero_grad()
    return hessian_dict


def calc_jacobian(weight_dict, loss, model):
    model.zero_grad()
    jacobian_dict = {}
    for name, weight in weight_dict.items():
        first_gradient = grad(loss, weight, create_graph=True)[0]
        jacobian_dict[name] = first_gradient.clone()
    model.zero_grad()
    return jacobian_dict


def polynomial(x, coeff):
    # xs = [x]
    # for i in range(2, coeff.size(0)):
    #     xs.append(xs[-1]*x)
    # xs.reverse()
    # x = torch.stack(xs, dim=-1)
    x = torch.stack([x**i for i in range(coeff.size(0)-1, 0, -1)], dim=-1)
    out = (x*coeff[:-1]).sum(dim=-1) + coeff[-1].data.item()
    return out


def gaussian(x, coeff):
    # coeff : [n, 3], includes a, b, c
    ## a * exp(-((x-b)/c)^2) + ...
    size = x.size()
    x = x.view(-1).unsqueeze(0)
    x = (coeff[:, 0:1] * torch.exp(-((x - coeff[:, 1:2]) /
                                     coeff[:, 2:3]).square())).sum(dim=0).view(size)
    return x


def lowrank_decompose(x, r, u_ortho=False, out_u=None, out_v=None):
    ### x [..., m, n]
    # r rank
    u, s, v = x.data.svd(some=True)
    v = v.transpose(-2, -1).contiguous()
    u = u[..., :, :r]
    s = s[..., :r]
    v = v[..., :r, :]
    if(u_ortho == False):
        u.mul_(s.unsqueeze(-2))
    else:
        v.mul_(s.unsqueeze(-1))
    if(out_u is not None):
        out_u.data.copy_(u)
    if(out_v is not None):
        out_v.data.copy_(v)
    return u, v


def get_conv2d_flops(input_shape, conv_filter, stride=(1, 1), padding=(1, 1)):
    # input_shape = (4, 3,300,300) # Format:(batch, channels, rows,cols)
    # conv_filter = (64,3,3,3)  # Format: (num_filters, channels, rows, cols)
    # stride = (1, 1) in (height, width)
    # padding = (1, 1) in (height, width)
    if(type(stride) not in {list, tuple}):
        stride = [stride, stride]
    if(type(padding) not in {list, tuple}):
        padding = [padding, padding]
    n = conv_filter[1] * conv_filter[2] * conv_filter[3]  # vector_length
    # general defination for number of flops (n: multiplications and n-1: additions)
    flops_per_instance = n + 1

    num_instances_per_filter = (
        (input_shape[2] - conv_filter[2] + 2*padding[0]) / stride[0]) + 1  # for rows
    # multiplying with cols
    num_instances_per_filter *= ((input_shape[3] -
                                  conv_filter[3] + 2*padding[1]) / stride[1]) + 1

    flops_per_filter = num_instances_per_filter * flops_per_instance
    # multiply with number of filters adn batch
    total_flops_per_layer = flops_per_filter * conv_filter[0] * input_shape[0]
    return total_flops_per_layer


"""
def mrr_modulator(T, a=0.9, r=0.8):
    '''
    @description: map from the field intensity of through port transmission to coherent light with phase reponse
    @T {torch.Tensor or np.ndarray} field intensity modulation factor
    @a {float} attenuation factor from [0,1]. Default: 0.9
    @r {float} transmission/self-coupling factor from [0,1]. Default: 0.8
    @return: complexed light signal
    '''
    if(isinstance(T, np.ndarray)):
        T = torch.from_numpy(T)
    cos_phi = (a**2 + r**2 - T * (1 + r**2 * a**2)) / 2 * (1 - T) * a * r
    phi = torch.acos(cos_phi)
    sin_phi = torch.sin(phi)
    phase = np.pi + phi + \
        torch.atan((r*sin_phi-2*r**2*a*sin_phi*cos_phi+r*a **
                    2*sin_phi)/(a-r*cos_phi)*(1-r*a*cos_phi))
    cos_phase, sin_phase = torch.cos(phase), torch.sin(phase)
    output_real = T * cos_phase - T * sin_phase
    output_imag = T * sin_phase + T * cos_phase
    output = torch.stack([output_real, output_imag], dim=-1)
    return output
"""
