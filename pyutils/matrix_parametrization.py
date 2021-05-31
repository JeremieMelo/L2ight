#!/usr/bin/env python
# coding=UTF-8
import logging
import time
from functools import partial
from multiprocessing.dummy import Pool
from typing import List

import numpy as np

# from .general import disable_tf_warning

# disable_tf_warning()
# import tensorflow as tf
import torch
from numba import jit
from scipy.stats import ortho_group, unitary_group
from torch import nn
from torch.types import Device

from .general import TimerCtx

try:
    import matrix_parametrization_cuda
except:
    logging.warning("Cannot import matrix_parametrization_cuda")

__all__ = ["profile", "RealUnitaryDecomposer", "RealUnitaryDecomposerBatch", "sparsifyDecomposition"]

# try:
#     config = tf.ConfigProto()
#     config.gpu_options.allow_growth = True
#     tf.enable_eager_execution(config=config)
# except:
#     # physical_devices = tf.config.list_physical_devices('GPU')
#     [tf.config.experimental.set_memory_growth(device, True) for device in tf.config.list_physical_devices('GPU')]


def profile(func=None, timer=True):
    import time
    from functools import partial, wraps
    if (func == None):
        return partial(profile, timer=timer)

    @wraps(func)
    def wrapper(*args, **kw):
        if (timer):
            local_time = time.time()
            res = func(*args, **kw)
            end_time = time.time()
            print('[I] <%s> runtime: %.3f ms' %
                  (func.__name__, (end_time - local_time) * 1000))
        else:
            res = func(*args, **kw)
        return res

    return wrapper


def batch_diag(x):
    # x[..., N, N] -> [..., N]
    assert len(x.shape) >= 2, f"At least 2-D array/tensor is expected, but got shape {x.shape}"
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
    x = np.zeros(list(batch_shape)+[N,N], dtype=dtype)
    x.reshape(-1, N*N)[..., ::N+1] = 1
    return x

def batch_eye(N: int, batch_shape: List[int], dtype: torch.dtype, device: Device = torch.device("cuda")) -> torch.Tensor:
    x = torch.zeros(list(batch_shape)+[N,N], dtype=dtype, device=device)
    x.view(-1, N*N)[..., ::N+1] = 1
    return x

class RealUnitaryDecomposer(object):
    timer = False

    def __init__(self, min_err=1e-7, timer=False, determine=False):
        self.min_err = min_err
        self.timer = timer
        self.determine = determine

    def buildPlaneUnitary(self, p, q, phi, N, transpose=True):
        assert N > 0 and isinstance(
            N, int), "[E] Matrix size must be positive integer"
        assert isinstance(p, int) and isinstance(q,
                                                 int) and 0 <= p < q < N, "[E] Integer value p and q must satisfy p < q"
        assert (isinstance(phi, float) or isinstance(phi, int)
                ), "[E] Value phi must be of type float or int"

        U = np.eye(N)
        c = np.cos(phi)
        s = np.sin(phi)

        U[p, p] = U[q, q] = c
        U[q, p] = s if not transpose else -s
        U[p, q] = -s if not transpose else s

        return U

    def calPhi_determine(self, u1, u2, is_first_col=False):
        u1_abs, u2_abs = np.abs(u1), np.abs(u2)
        min_err = self.min_err

        if u1_abs < min_err and u2_abs < min_err:
            phi = 0
        elif u1_abs >= min_err and u2_abs < min_err:
            phi = 0 if u1 > min_err else -np.pi
        elif u1_abs < min_err and u2_abs >= min_err:
            phi = -0.5 * np.pi if u2 > min_err else 0.5 * np.pi
        else:
            # solve the equation: u'_1n=0
            if(is_first_col):
                phi = np.arctan2(-u2, u1)  # 4 quadrant
            else:
                phi = np.arctan(-u2/u1)

    def calPhi_nondetermine(self, u1, u2, is_first_col=False):
        u1_abs, u2_abs = np.abs(u1), np.abs(u2)
        min_err = self.min_err

        if u1_abs < min_err and u2_abs < min_err:
            phi = 0
        elif u1_abs >= min_err and u2_abs < min_err:
            phi = 0 if u1 > min_err else -np.pi
        elif u1_abs < min_err and u2_abs >= min_err:
            phi = -0.5 * np.pi if u2 > min_err else 0.5 * np.pi
        else:
            # solve the equation: u'_1n=0
            phi = np.arctan2(-u2, u1)  # 4 quadrant

        # print(phi, u1, u2)

        return phi

    # @profile(timer=timer)
    def decomposeKernel(self, U, dim, phi_list=None):
        '''return U^(N-1); (phi_1,...,phi_N-2); (sigma_1,...,sigma_N-2)'''
        N = U.shape[0]
        if(phi_list is None):
            phi_list = np.zeros(dim, dtype=np.float64)

        calPhi = self.calPhi_determine if self.determine else self.calPhi_nondetermine
        for i in range(N - 1):
            # with TimerCtx() as t:
            u1, u2 = U[0, 0], U[0, N - 1 - i]
            phi = calPhi(u1, u2, is_first_col=(i == 0))

            # print("calPhi:", t.interval)
            # with TimerCtx() as t:
            phi_list[i] = phi
            p, q = 0, N - i - 1
            # U = np.dot(U, self.buildPlaneUnitary(p=p, q=q, phi=phi, N=N, transpose=True))
            c, s = np.cos(phi), np.sin(phi)
            row_p, row_q = U[:, p], U[:, q]
            U[:, p], U[:, q] = row_p * c - row_q * s, row_p * s + row_q * c
            # print(U)
            # U[:, p], U[:, q] = U[:, p] * c - U[:, q] * s, U[:, p] * s + U[:, q] * c
            # print("rotate:", t.interval)

        # print(f'[I] Decomposition kernel done')
        return U, phi_list

    @profile(timer=timer)
    def decompose(self, U_ori):
        U = U_ori.copy()
        N = U.shape[0]
        assert N > 0 and U.shape[0] == U.shape[1], '[E] Input matrix must be square and N > 0'

        phi_mat = np.zeros([N, N], dtype=np.float64)
        sigma_mat = np.zeros([N, N], dtype=np.float64)
        delta_list = np.zeros(N, dtype=np.float64)

        for i in range(N - 1):
            # U, phi_list = self.decomposeKernel(U, dim=N)
            # phi_mat[i, :] = phi_list
            U, _ = self.decomposeKernel(U, dim=N, phi_list=phi_mat[i, :])
            delta_list[i] = U[0, 0]
            U = U[1:, 1:]
        else:
            delta_list[-1] = U[-1, -1]

        # print(f'[I] Decomposition done')
        return delta_list, phi_mat

    @profile(timer=timer)
    def reconstruct(self, delta_list, phi_mat):
        N = delta_list.shape[0]
        Ur = np.identity(N)

        # reconstruct from right to left as in the book chapter
        for i in range(N):
            for j in range(N - i - 1):
                phi = phi_mat[i, j]
                Ur = np.dot(self.buildPlaneUnitary(
                    i, N - j - 1, phi, N, transpose=False), Ur)

        D = np.diag(delta_list)
        Ur = np.dot(D, Ur)
        print(f'[I] Reconstruct {N}*{N} unitary matrix done')

        return Ur

    @profile(timer=timer)
    def reconstruct_2(self, delta_list, phi_mat):
        N = delta_list.shape[0]
        Ur = np.identity(N)

        # reconstruct from right to left as in the book chapter
        # cos_phi = np.cos(phi_list)
        # sin_phi = np.sin(phi_list)
        # print(phi_list)
        # print(cos_phi)
        # count = 0
        phi_mat_cos = np.cos(phi_mat)
        phi_mat_sin = np.sin(phi_mat)
        for i in range(N-1):
            for j in range(N - i - 1):
                # with TimerCtx() as t:
                # phi = phi_mat[i, j]
                # c = np.cos(phi)
                # s = np.sin(phi)
                # index = int((2 * N - i - 1) * i / 2 + j)
                # phi = phi_list[index]
                c, s = phi_mat_cos[i, j], phi_mat_sin[i, j]

                # print("cos:", t.interval)
                # c = cos_phi[count]
                # s = sin_phi[count]
                # count += 1
                # with TimerCtx() as t:
                p = i
                q = N - j - 1
                # Ur_new = Ur_old.clone()
                Ur[p, :], Ur[q, :] = Ur[p, :] * c - \
                    Ur[q, :] * s, Ur[p, :] * s + Ur[q, :] * c
                # print("rotate:", t.interval)
        D = np.diag(delta_list)
        Ur = np.dot(D, Ur)
        # print(f'[I] Reconstruct {N}*{N} unitary matrix done')

        return Ur

    def checkIdentity(self, M):
        return (M.shape[0] == M.shape[1]) and np.allclose(M, np.eye(M.shape[0]))

    def checkUnitary(self, U):
        M = np.dot(U, U.T)
        # print(M)
        return self.checkIdentity(M)

    def checkEqual(self, M1, M2):
        return (M1.shape[0] == M2.shape[0]) and (M1.shape[1] == M2.shape[1]) and np.allclose(M1, M2)

    def genRandomOrtho(self, N):
        U = ortho_group.rvs(N)
        print(
            f'[I] Generate random {N}*{N} unitary matrix, check unitary: {self.checkUnitary(U)}')
        return U

    def toDegree(self, M):
        return np.degrees(M)


class RealUnitaryDecomposerPyTorch(object):
    timer = True

    def __init__(self, device="cuda", min_err=1e-6, timer=False, use_multithread=False, n_thread=8):
        self.min_err = min_err
        self.timer = timer
        self.device = torch.device(device)
        self.use_multithread = use_multithread
        self.n_thread = n_thread
        self.pi = torch.Tensor([np.pi]).to(self.device)
        self.zero = torch.Tensor([0]).to(self.device)
        if(self.use_multithread):
            self.pool = Pool(self.n_thread)
        else:
            self.pool = None

    def buildPlaneUnitary(self, p, q, phi, N, transpose=True):

        U = torch.eye(N, device=self.device)
        c = torch.cos(phi)
        s = torch.sin(phi)

        U[p, p] = U[q, q] = c
        U[q, p] = s if not transpose else -s
        U[p, q] = -s if not transpose else s

        return U

    def calPhi(self, u1, u2):
        u1_abs, u2_abs = torch.abs(u1), torch.abs(u2)
        min_err = self.min_err

        if u1_abs < min_err and u2_abs < min_err:
            phi = self.zero
        elif u1_abs >= min_err and u2_abs < min_err:
            phi = self.zero if u1 > min_err else -self.pi
        elif u1_abs < min_err and u2_abs >= min_err:
            phi = torch.Tensor([-0.5]).to(self.device) * \
                self.pi if u2 > min_err else 0.5 * self.pi
        else:
            # solve the equation: u'_1n=0
            phi = torch.atan2(-u2, u1)  # 4 quadrant

        if len(phi.size()) == 0:
            phi = phi.unsqueeze(0)
        return phi

    # @profile(timer=timer)
    def decomposeKernel(self, U, dim):
        '''return U^(N-1); (phi_1,...,phi_N-2); (sigma_1,...,sigma_N-2)'''
        N = U.size(0)
        phi_list = torch.zeros((dim)).to(self.device)

        for i in range(N - 1):
            u1, u2 = U[0, 0], U[0, N - 1 - i]
            phi = self.calPhi(u1, u2)
            assert len(phi.size()) > 0, f"{phi}"
            phi_list[i] = phi
            p, q = 0, N - i - 1
            U = torch.mm(U, self.buildPlaneUnitary(
                p=p, q=q, phi=phi, N=N, transpose=True))

        # print(f'[I] Decomposition kernel done')
        return U, phi_list

    # @profile(timer=timer)
    def decompose(self, U):
        N = U.size(0)

        phi_mat = torch.zeros((N, N)).to(self.device)
        delta_list = torch.zeros((N)).to(self.device)

        for i in range(N - 1):
            U, phi_list = self.decomposeKernel(U, dim=N)
            phi_mat[i, :] = phi_list
            delta_list[i] = U[0, 0]
            U = U[1:, 1:]
        else:
            delta_list[-1] = U[-1, -1]

        # print(f'[I] Decomposition done')
        return delta_list, phi_mat

    @profile(timer=timer)
    def reconstruct(self, delta_list, phi_mat):
        N = delta_list.size(0)
        Ur = nn.init.eye_(torch.empty((N, N))).to(self.device)

        # reconstruct from right to left as in the book chapter
        for i in range(N):
            for j in range(N - i - 1):
                phi = phi_mat[i, j]
                Ur = torch.mm(self.buildPlaneUnitary(
                    i, N - j - 1, phi, N, transpose=False), Ur)

        D = torch.diag(delta_list).to(self.device)
        Ur = torch.mm(D, Ur)
        # print(f'[I] Reconstruct {N}*{N} unitary matrix done')

        return Ur

    @profile(timer=timer)
    def reconstruct_2(self, delta_list, phi_list):
        N = delta_list.size(0)
        Ur = nn.init.eye_(torch.empty((N, N))).to(self.device)
        Ur_new = nn.init.eye_(torch.empty((N, N))).to(self.device)

        # reconstruct from right to left as in the book chapter
        if(self.use_multithread == False):
            for i in range(N):
                for j in range(N - i - 1):
                    # phi = phi_mat[i, j]
                    index = int((2 * N - i - 1) * i / 2 + j)
                    phi = phi_list[index]
                    c = torch.cos(phi)
                    s = torch.sin(phi)
                    p = i
                    q = N - j - 1
                    # Ur_new = Ur_old.clone()
                    Ur_new[p, ...], Ur_new[q, ...] = Ur[p, ...] * c - \
                        Ur[q, ...] * s, Ur[p, ...] * s + Ur[q, ...] * c
                    Ur = Ur_new.clone()
                    # Ur = torch.mm(self.buildPlaneUnitary(i, N - j - 1, phi, N, transpose=False), Ur)

        else:

            PlaneUnitary_list = [(i, j, phi_list[int((2 * N - i - 1) * i / 2 + j)])
                                 for i in range(N) for j in range(N - i - 1)]

            PlaneUnitary_list = self.pool.map(lambda args: self.buildPlaneUnitary(
                args[0], N - args[1] - 1, args[2], N, transpose=False), PlaneUnitary_list)

            PlaneUnitary_list = torch.stack(PlaneUnitary_list, dim=0)
            n_planes = PlaneUnitary_list.size(0)
            log2_n_planes = int(np.log2(n_planes))
            n_iters = log2_n_planes if(
                2**log2_n_planes == n_planes) else log2_n_planes + 1

            for _ in range(n_iters):
                even_batch = PlaneUnitary_list[::2]
                odd_batch = PlaneUnitary_list[1::2]
                if(odd_batch.size(0) < even_batch.size(0)):
                    odd_batch = torch.cat([odd_batch, torch.eye(
                        N).to(self.device).unsqueeze(0)], dim=0)
                PlaneUnitary_list = torch.bmm(odd_batch, even_batch)

            Ur = PlaneUnitary_list.squeeze(0)

        D = torch.diag(delta_list).to(self.device)
        Ur = torch.mm(D, Ur)
        # print(f'[I] Reconstruct {N}*{N} unitary matrix done')

        return Ur

    def prunePhases(self, phi_mat, epsilon=1e-4):
        N = phi_mat.size(0)
        is_close_to_0 = phi_mat.abs() < epsilon
        is_close_to_90 = torch.abs(phi_mat - self.pi/2) < epsilon
        is_close_to_180 = torch.abs(phi_mat - self.pi) < epsilon
        is_close_to_270 = torch.abs(phi_mat + self.pi/2) < epsilon
        print(is_close_to_0.sum()-N*(N+1)/2)
        print(is_close_to_90.sum())
        print(is_close_to_180.sum())
        print(is_close_to_270.sum())
        phi_mat[is_close_to_0] = self.zero
        phi_mat[is_close_to_90] = self.pi/2
        phi_mat[is_close_to_180] = self.pi
        phi_mat[is_close_to_270] = -self.pi/2
        n_prune = is_close_to_0.sum() + is_close_to_90.sum() + is_close_to_180.sum() + \
            is_close_to_270.sum() - N*(N+1)/2
        return n_prune

    def checkIdentity(self, M):
        I = nn.init.eye_(torch.empty((N, N))).to(self.device)
        return (M.size(0) == M.size(1)) and torch.allclose(M, I, rtol=1e-04, atol=1e-07)

    def checkUnitary(self, U):
        M = torch.mm(U, U.t())
        print(M)
        return self.checkIdentity(M)

    def checkEqual(self, M1, M2):
        return (M1.size() == M2.size()) and torch.allclose(M1, M2, rtol=1e-04, atol=1e-07)

    def genRandomOrtho(self, N):
        U = torch.Tensor(ortho_group.rvs(N)).to(self.device)
        print(
            f'[I] Generate random {N}*{N} unitary matrix, check unitary: {self.checkUnitary(U)}')
        return U

    def toDegree(self, M):
        return M * 180 / self.pi


class RealUnitaryDecomposerBatch(object):
    timer = False

    def __init__(self, min_err=1e-7, timer=False, determine=False, alg="reck", dtype=np.float64):
        self.min_err = min_err
        self.timer = timer
        self.determine = determine
        self.alg = alg
        assert alg.lower() in {"reck", "clements", "francis"}, logging.error(f"Unitary decomposition algorithm can only be [reck, clements, francis], but got {alg}")
        self.dtype = dtype

    def set_alg(self, alg):
        assert alg.lower() in {"reck", "clements", "francis"}, logging.error(f"Unitary decomposition algorithm can only be [reck, clements, francis], but got {alg}")
        self.alg = alg

    def buildPlaneUnitary(self, p, q, phi, N, transpose=True):
        assert N > 0 and isinstance(
            N, int), "[E] Matrix size must be positive integer"
        assert isinstance(p, int) and isinstance(q,
                                                 int) and 0 <= p < q < N, "[E] Integer value p and q must satisfy p < q"
        assert (isinstance(phi, float) or isinstance(phi, int)
                ), "[E] Value phi must be of type float or int"

        U = np.eye(N)
        c = np.cos(phi)
        s = np.sin(phi)

        U[p, p] = U[q, q] = c
        U[q, p] = s if not transpose else -s
        U[p, q] = -s if not transpose else s

        return U

    def calPhi_batch_determine(self, u1: np.ndarray, u2: np.ndarray, is_first_col=False) -> np.ndarray:
        pi = np.pi
        u1_abs, u2_abs = np.abs(u1), np.abs(u2)
        min_err = self.min_err
        cond1 = u1_abs < min_err
        cond2 = u2_abs < min_err
        cond1_n = ~cond1
        cond2_n = ~cond2
        if(is_first_col):
            phi = np.where(cond1 & cond2, 0,
                           np.where(cond1_n & cond2, np.where(u1 > min_err, 0, -pi),
                                    np.where(cond1 & cond2_n, np.where(u2 > min_err, -0.5*pi, 0.5*pi), np.arctan2(-u2, u1))))
        else:
            phi = np.where(cond1 & cond2, 0,
                           np.where(cond1_n & cond2, np.where(u1 > min_err, 0, -pi),
                                    np.where(cond1 & cond2_n, np.where(u2 > min_err, -0.5*pi, 0.5*pi), np.arctan(-u2/u1))))
        return phi

    def calPhi_batch_nondetermine(self, u1: np.ndarray, u2: np.ndarray, is_first_col=False) -> np.ndarray:
        pi = np.pi
        u1_abs, u2_abs = np.abs(u1), np.abs(u2)
        min_err = self.min_err
        cond1 = u1_abs < min_err
        cond2 = u2_abs < min_err
        cond1_n = ~cond1
        cond2_n = ~cond2
        phi = np.where(cond1 & cond2, 0,
                       np.where(cond1_n & cond2, np.where(u1 > min_err, 0, -pi),
                                np.where(cond1 & cond2_n, np.where(u2 > min_err, -0.5*pi, 0.5*pi), np.arctan2(-u2, u1))))

        return phi

    def calPhi_determine(self, u1, u2, is_first_col=False):
        pi = np.pi
        u1_abs, u2_abs = np.abs(u1), np.abs(u2)
        min_err = self.min_err

        if u1_abs < min_err and u2_abs < min_err:
            phi = 0
        elif u1_abs >= min_err and u2_abs < min_err:
            phi = 0 if u1 > min_err else -pi
        elif u1_abs < min_err and u2_abs >= min_err:
            phi = -0.5 * pi if u2 > min_err else 0.5 * pi
        else:
            # solve the equation: u'_1n=0
            if(is_first_col):
                phi = np.arctan2(-u2, u1)  # 4 quadrant4
            else:
                phi = np.arctan(-u2/u1)
            # phi = np.arctan(-u2/u1)
        # print(phi, u1, u2)

        # cond = ((u1_abs < min_err) << 1) | (u2_abs < min_err)
        # if(cond == 0):
        #     phi = np.arctan2(-u2, u1)
        # elif(cond == 1):
        #     phi = 0 if u1 > min_err else -np.pi
        # elif(cond == 2):
        #     phi = -0.5 * np.pi if u2 > min_err else 0.5 * np.pi
        # else:
        #     phi = 0
        # phi = [np.arctan2(-u2, u1), 0 if u1 > min_err else -np.pi, -0.5 * np.pi if u2 > min_err else 0.5 * np.pi, 0][cond]

        return phi

    def calPhi_nondetermine(self, u1, u2):
        pi = np.pi
        u1_abs, u2_abs = np.abs(u1), np.abs(u2)
        min_err = self.min_err

        if u1_abs < min_err and u2_abs < min_err:
            phi = 0
        elif u1_abs >= min_err and u2_abs < min_err:
            phi = 0 if u1 > min_err else -pi
        elif u1_abs < min_err and u2_abs >= min_err:
            phi = -0.5 * pi if u2 > min_err else 0.5 * pi
        else:
            # solve the equation: u'_1n=0
            phi = np.arctan2(-u2, u1)  # 4 quadrant4

        return phi

    def decomposeKernel_batch(self, U: np.ndarray, dim, phi_list=None):
        '''return U^(N-1); (phi_1,...,phi_N-2); (sigma_1,...,sigma_N-2)'''
        N = U.shape[-1]
        if(phi_list is None):
            phi_list = np.zeros(list(U.shape[:-2])+[dim], dtype=np.float64)

        calPhi_batch = self.calPhi_batch_determine if self.determine else self.calPhi_batch_nondetermine
        for i in range(N - 1):
            # with TimerCtx() as t:
            u1, u2 = U[..., 0, 0], U[..., 0, N - 1 - i]
            phi = calPhi_batch(u1, u2, is_first_col=(i == 0))

            # print("calPhi:", t.interval)
            # with TimerCtx() as t:
            phi_list[..., i] = phi

            p, q = 0, N - i - 1
            # U = np.dot(U, self.buildPlaneUnitary(p=p, q=q, phi=phi, N=N, transpose=True))
            c, s = np.cos(phi)[..., np.newaxis], np.sin(phi)[..., np.newaxis]
            col_p, col_q = U[..., :, p], U[..., :, q]
            U[..., :, p], U[..., :, q] = col_p * \
                c - col_q * s, col_p * s + col_q * c
            # U[:, p], U[:, q] = U[:, p] * c - U[:, q] * s, U[:, p] * s + U[:, q] * c
            # print("rotate:", t.interval)

        # print(f'[I] Decomposition kernel done')
        return U, phi_list

    def decomposeKernel_determine(self, U, phi_list):
        '''return U^(N-1); (phi_1,...,phi_N-2); (sigma_1,...,sigma_N-2)'''
        N = U.shape[0]
        # if(phi_list is None):
        #     phi_list = np.zeros(dim, dtype=self.dtype)

        # calPhi = self.calPhi_determine
        # cos, sin = np.cos, np.sin
        for i in range(N - 1):
            # with TimerCtx() as t:
            u1, u2 = U[0, 0], U[0, N - 1 - i]
            # phi = calPhi(u1, u2, is_first_col=(i == 0))
            pi = np.pi
            u1_abs, u2_abs = np.abs(u1), np.abs(u2)
            min_err = self.min_err

            if u1_abs < min_err and u2_abs < min_err:
                phi = 0
            elif u1_abs >= min_err and u2_abs < min_err:
                phi = 0 if u1 > min_err else -pi
            elif u1_abs < min_err and u2_abs >= min_err:
                phi = -0.5 * pi if u2 > min_err else 0.5 * pi
            else:
                # solve the equation: u'_1n=0
                if(i==0):
                    phi = np.arctan2(-u2, u1)  # 4 quadrant4
                else:
                    phi = np.arctan(-u2/u1)

            # print("calPhi:", t.interval)
            # with TimerCtx() as t:
            phi_list[i] = phi
            p, q = 0, N - i - 1
            # U = np.dot(U, self.buildPlaneUnitary(p=p, q=q, phi=phi, N=N, transpose=True))
            c, s = np.cos(phi), np.sin(phi)
            row_p, row_q = U[:, p], U[:, q]
            row_p_cos, row_p_sin = row_p * c, row_p * s
            row_q_cos, row_q_sin = row_q * c, row_q * s
            U[:, p], U[:, q] = row_p_cos - row_q_sin, row_q_cos + row_p_sin
            # U[:, p], U[:, q] = row_p * c - row_q * s, row_q*c + row_p*s
            # U[:, p], U[:, q] = U[:, p] * c - U[:, q] * s, U[:, p] * s + U[:, q] * c
            # print("rotate:", t.interval)

        # print(f'[I] Decomposition kernel done')
        return U, phi_list

    def decomposeKernel_nondetermine(self, U, phi_list):
        '''return U^(N-1); (phi_1,...,phi_N-2); (sigma_1,...,sigma_N-2)'''
        N = U.shape[0]
        # if(phi_list is None):
        #     phi_list = np.zeros(dim, dtype=self.dtype)

        # calPhi = self.calPhi_nondetermine
        # cos, sin = np.cos, np.sin
        pi = np.pi
        half_pi = np.pi / 2
        min_err = self.min_err
        for i in range(N - 1):
            # with TimerCtx() as t:
            u1, u2 = U[0, 0], U[0, N - 1 - i]
            u1_abs, u2_abs = np.abs(u1), np.abs(u2)
            cond1, cond2 = u1_abs >= min_err, u2_abs >= min_err
            if(cond1 & cond2):
                phi = np.arctan2(-u2, u1)
            elif(~cond1 & cond2):
                phi = -half_pi if u2 > min_err else half_pi
            elif(cond1 & ~cond2):
                phi = 0 if u1 > min_err else -pi
            else:
                phi = 0

            # print("calPhi:", t.interval)
            # with TimerCtx() as t:
            phi_list[i] = phi
            p, q = 0, N - i - 1
            # U = np.dot(U, self.buildPlaneUnitary(p=p, q=q, phi=phi, N=N, transpose=True))
            # c, s = np.cos(phi), np.sin(phi)
            c = np.cos(phi)
            s = (1 - c*c)**0.5 if phi > 0 else -((1 - c*c)**0.5)
            row_p, row_q = U[:, p], U[:, q]
            row_p_cos, row_p_sin = row_p * c, row_p * s
            row_q_cos, row_q_sin = row_q * c, row_q * s
            U[:, p], U[:, q] = row_p_cos - row_q_sin, row_q_cos + row_p_sin
            # U[:, p], U[:, q] = row_p * c - row_q * s, row_q*c + row_p*s
            # U[:, p], U[:, q] = U[:, p] * c - U[:, q] * s, U[:, p] * s + U[:, q] * c
            # print("rotate:", t.interval)

        # print(f'[I] Decomposition kernel done')
        return U, phi_list

    @profile(timer=timer)
    def decompose_francis_batch(self, U: np.ndarray):
        N = U.shape[-1]
        assert N > 0 and U.shape[-1] == U.shape[-2], '[E] Input matrix must be square and N > 0'

        phi_mat = np.zeros(U.shape, dtype=np.float64)
        delta_list = np.zeros(U.shape[:-1], dtype=np.float64)

        for i in range(N - 1):
            # U, phi_list = self.decomposeKernel(U, dim=N)
            # phi_mat[i, :] = phi_list
            U, _ = self.decomposeKernel_batch(
                U, dim=N, phi_list=phi_mat[..., i, :])
            delta_list[..., i] = U[..., 0, 0]
            U = U[..., 1:, 1:]
        else:
            delta_list[..., -1] = U[..., -1, -1]

        # print(f'[I] Decomposition done')
        return delta_list, phi_mat

    @profile(timer=timer)
    def decompose_francis_cpu(self, U):
        #### This decomposition has follows the natural reflection of MZIs. Thus the circuit will give a reversed output.
        ### Francis style, 1962
        N = U.shape[0]
        assert N > 0 and U.shape[0] == U.shape[1], '[E] Input matrix must be square and N > 0'

        phi_mat = np.zeros([N, N], dtype=self.dtype)
        delta_list = np.zeros(N, dtype=self.dtype)
        decompose_kernel = self.decomposeKernel_determine if self.determine else self.decomposeKernel_nondetermine

        for i in range(N - 1):
            # U, phi_list = self.decomposeKernel(U, dim=N)
            # phi_mat[i, :] = phi_list
            U, _ = decompose_kernel(U, phi_list=phi_mat[i, :])
            # U, _ = self.decomposeKernel(U, dim=N, phi_list=phi_mat[i, :])
            delta_list[i] = U[0, 0]
            U = U[1:, 1:]
        else:
            delta_list[-1] = U[-1, -1]

        # print(f'[I] Decomposition done')
        return delta_list, phi_mat

    def decompose(self, U):
        if(self.alg == "reck"):
            decompose_cpu = self.decompose_reck_cpu
            decompose_batch = self.decompose_reck_batch
            decompose_cuda = matrix_parametrization_cuda.decompose_reck
        elif(self.alg == "francis"):
            decompose_cpu = self.decompose_francis_cpu
            decompose_batch = self.decompose_francis_batch
            decompose_cuda = matrix_parametrization_cuda.decompose_francis
        elif(self.alg == "clements"):
            decompose_cpu = self.decompose_clements_cpu
            decompose_batch = self.decompose_clements_batch
            decompose_cuda = matrix_parametrization_cuda.decompose_clements
        else:
            raise NotImplementedError

        if(isinstance(U, np.ndarray)):
            if(len(U.shape) == 2):
                return decompose_cpu(U)
            else:
                return decompose_batch(U)
        else:
            if(U.is_cuda):
                N = U.size(-1)
                size = U.size()
                U = U.view(-1, N, N).contiguous()
                delta_list = torch.zeros(list(U.size())[:-1], dtype=U.dtype, device=U.device).contiguous()
                phi_mat = torch.zeros_like(U).contiguous()
                decompose_cuda(U, delta_list, phi_mat)
                delta_list = delta_list.view(list(size)[:-1])
                phi_mat = phi_mat.view(size)
                return delta_list, phi_mat
            else:
                if(U.dim() == 2):
                    delta_list, phi_mat = decompose_cpu(U.cpu().numpy())
                else:
                    delta_list, phi_mat = decompose_batch(U.cpu().numpy())
                return torch.from_numpy(delta_list), torch.from_numpy(phi_mat)

    def decompose_reck_cpu(self, U):
        '''Reck decomposition implemented by Neurophox. Triangular mesh, input and output have no mirroring effects, i.e, [x1, ..., xn] -> Y = U x X -> [y1, ..., yn]
        Rmn: [ cos(phi)   -sin(phi)] -> MZI achieves counter-clock-wise rotation with phi (reconstruction, left mul)
             [ sin(phi)    cos(phi)]
        Rmn*:[ cos(phi)    sin(phi)] -> column-wise clock-wise rotation (decompose, right mul)
             [-sin(phi)    cos(phi)]

        U = D R43 R32 R43 R21 R32 R43
        '''
        N = U.shape[0]
        assert N > 0 and U.shape[0] == U.shape[1], '[E] Input matrix must be square and N > 0'

        phi_mat = np.zeros([N, N], dtype=self.dtype) ## left upper triangular array.
        '''
        the bottom-left phase corresponds to the MZI at the bottom-left corner.
        The decomposition ordering follows from bottom to top, from left to right.
        R21 R32 R43  0
        R32 R43 0    0
        R43 0   0    0
        0   0   0    0
        '''
        # theta_checkerboard = np.zeros_like(U, dtype=self.dtype)
        delta_list = np.zeros(N, dtype=self.dtype) ## D
        '''
            x x x 0     x x 0 0
            x x x x  -> x x x 0
            x x x x     x x x x
            x x x x     x x x x
        '''

        for i in range(N-1):
            ### each outer loop deals with one off-diagonal, nullification starts from top-right
            ### even loop for column rotation
            for j in range(i + 1):
                ### let p, q be the indices for the nullified '0'
                p = j ## row
                q = N - 1 - i + j ## col
                ### rotate two columns such that u2 is nullified to 0
                pi = np.pi
                half_pi = np.pi / 2
                min_err = self.min_err
                ### col q-1 nullifies col q
                u1, u2 = U[p, q-1], U[p, q]
                u1_abs, u2_abs = np.abs(u1), np.abs(u2)
                cond1, cond2 = u1_abs >= min_err, u2_abs >= min_err
                if(cond1 & cond2):
                    phi = np.arctan2(-u2, u1)
                elif(~cond1 & cond2):
                    phi = -half_pi if u2 > min_err else half_pi
                elif(cond1 & ~cond2):
                    phi = 0 if u1 > min_err else -pi
                else:
                    phi = 0
                # phi_mat[p,q] = phi
                # theta_checkerboard[pairwise_index, -j - 1] = phi
                phi_mat[N-i-2, j] = phi
                c, s = np.cos(phi), np.sin(phi)
                ## q_m1 means q-1; right multiply by R*
                col_q_m1, col_q = U[p:, q-1], U[p:, q]
                col_q_m1_cos, col_q_m1_sin = col_q_m1 * c, col_q_m1 * s
                col_q_cos, col_q_sin = col_q * c, col_q * s
                U[p:, q-1], U[p:, q] = col_q_m1_cos - col_q_sin, col_q_cos + col_q_m1_sin

        delta_list = np.diag(U) ## only the first and last element can be 1 or -1, the rest elements are all 1. This feature can be used in fast forward/reconstruction

        return delta_list, phi_mat

    def decompose_reck_batch(self, U):
        '''Reck decomposition implemented by Neurophox. Triangular mesh, input and output have no mirroring effects, i.e, [x1, ..., xn] -> Y = U x X -> [y1, ..., yn]
        Rmn: [ cos(phi)   -sin(phi)] -> MZI achieves counter-clock-wise rotation with phi (reconstruction, left mul)
             [ sin(phi)    cos(phi)]
        Rmn*:[ cos(phi)    sin(phi)] -> column-wise clock-wise rotation (decompose, right mul)
             [-sin(phi)    cos(phi)]

        U = D R43 R32 R43 R21 R32 R43
        '''
        N = U.shape[-1]
        assert N > 0 and U.shape[-1] == U.shape[-2], '[E] Input matrix must be square and N > 0'

        phi_mat = np.zeros(U.shape, dtype=self.dtype) ## left upper triangular array.
        '''
        the bottom-left phase corresponds to the MZI at the bottom-left corner.
        The decomposition ordering follows from bottom to top, from left to right.
        R21 R32 R43  0
        R32 R43 0    0
        R43 0   0    0
        0   0   0    0
        '''
        # theta_checkerboard = np.zeros_like(U, dtype=self.dtype)
        delta_list = np.zeros(U.shape[:-1], dtype=self.dtype) ## D
        '''
            x x x 0     x x 0 0
            x x x x  -> x x x 0
            x x x x     x x x x
            x x x x     x x x x
        '''

        for i in range(N-1):
            ### each outer loop deals with one off-diagonal, nullification starts from top-right
            ### even loop for column rotation
            for j in range(i + 1):
                ### let p, q be the indices for the nullified '0'
                p = j ## row
                q = N - 1 - i + j ## col
                ### rotate two columns such that u2 is nullified to 0
                ### col q-1 nullifies col q

                u1, u2 = U[..., p, q-1], U[..., p, q]
                phi = self.calPhi_batch_nondetermine(u1, u2)

                phi_mat[..., N-i-2, j] = phi
                c, s = np.cos(phi)[..., np.newaxis], np.sin(phi)[..., np.newaxis]
                ## q_m1 means q-1; right multiply by R*
                col_q_m1, col_q = U[..., p:, q-1], U[..., p:, q]
                col_q_m1_cos, col_q_m1_sin = col_q_m1 * c, col_q_m1 * s
                col_q_cos, col_q_sin = col_q * c, col_q * s
                U[..., p:, q-1], U[..., p:, q] = col_q_m1_cos - col_q_sin, col_q_cos + col_q_m1_sin

        # delta_list = batch_diag(U)
        delta_list = batch_diag(U)

        return delta_list, phi_mat

    def reconstruct_reck_cpu(self, delta_list, phi_mat):
        N = delta_list.shape[0]
        Ur = np.identity(N)
        ### left multiply by a counter-clock-wise rotation
        '''
        cos, -sin
        sin, cos
        '''

        phi_mat_cos = np.cos(phi_mat)
        phi_mat_sin = np.sin(phi_mat)

        ## totally 2n-3 stage
        for i in range(N-1):
            lower = N - 2 - i
            for j in range(i+1):
                c, s = phi_mat_cos[lower, j], phi_mat_sin[lower, j]
                p = N - 2 - i + j
                q = p + 1

                row_p, row_q = Ur[p, lower:], Ur[q, lower:]
                # row_p_cos, row_p_sin = row_p * c, row_p * s
                # row_q_cos, row_q_sin = row_q * c, row_q * s
                # Ur[p, lower:], Ur[q, lower:] = row_p_cos - row_q_sin, row_p_sin + row_q_cos
                res = (c + 1j * s) * (row_p + 1j * row_q)
                Ur[p, lower:], Ur[q, lower:] = res.real, res.imag
        Ur = delta_list[:, np.newaxis] * Ur
        return Ur

    def reconstruct_reck_batch(self, delta_list, phi_mat):
        N = delta_list.shape[-1]
        Ur = batch_eye_cpu(N, batch_shape=delta_list.shape[:-1], dtype=delta_list.dtype)
        ### left multiply by a counter-clock-wise rotation
        '''
        cos, -sin
        sin, cos
        '''

        phi_mat_cos = np.cos(phi_mat)
        phi_mat_sin = np.sin(phi_mat)
        # e_jphi = np.exp(1j*phi_mat)

        for i in range(N-1):
            lower = N - 2 - i
            for j in range(i+1):
                c, s = phi_mat_cos[..., lower, j:j+1], phi_mat_sin[..., lower, j:j+1]
                p = N - 2 - i + j
                q = p + 1
                row_p, row_q = Ur[..., p, lower:], Ur[..., q, lower:]
                # row_p_cos, row_p_sin = row_p * c, row_p * s
                # row_q_cos, row_q_sin = row_q * c, row_q * s
                # Ur[..., p, lower:], Ur[..., q, lower:] = row_p_cos - row_q_sin, row_p_sin + row_q_cos
                ### this rotation is equivalent to complex number multiplication as an acceleration.
                res = (c + 1j * s) * (row_p + 1j * row_q)
                Ur[..., p, lower:], Ur[..., q, lower:] = res.real, res.imag
        Ur = delta_list[..., np.newaxis] * Ur
        return Ur

    def reconstruct_reck_batch_par(self, delta_list, phi_mat):
        N = delta_list.shape[-1]
        Ur = batch_eye_cpu(N, batch_shape=delta_list.shape[:-1], dtype=delta_list.dtype)
        ### left multiply by a counter-clock-wise rotation
        '''
        cos, -sin
        sin, cos
        '''

        phi_mat_cos = np.cos(phi_mat)
        phi_mat_sin = np.sin(phi_mat)

        ### 2n-3 stages
        for i in range(2 * N - 3):
            lower = N - 2 - i
            for j in range(i + 1):
                c, s = phi_mat_cos[..., lower, j:j+1], phi_mat_sin[..., lower, j:j+1]
                p = N - 2 - i + j
                q = p + 1
                row_p, row_q = Ur[..., p, lower:], Ur[..., q, lower:]
                row_p_cos, row_p_sin = row_p * c, row_p * s
                row_q_cos, row_q_sin = row_q * c, row_q * s
                Ur[..., p, lower:], Ur[..., q, lower:] = row_p_cos - row_q_sin, row_p_sin + row_q_cos
        Ur = delta_list[..., np.newaxis] * Ur
        return Ur

    @profile(timer=timer)
    def reconstruct_slow(self, delta_list, phi_mat):
        N = delta_list.shape[0]
        Ur = np.identity(N)

        # reconstruct from right to left as in the book chapter
        for i in range(N):
            for j in range(N - i - 1):
                phi = phi_mat[i, j]
                Ur = np.dot(self.buildPlaneUnitary(
                    i, N - j - 1, phi, N, transpose=False), Ur)

        D = np.diag(delta_list)
        Ur = np.dot(D, Ur)
        print(f'[I] Reconstruct {N}*{N} unitary matrix done')

        return Ur

    @profile(timer=timer)
    def reconstruct_francis_batch(self, delta_list: np.ndarray, phi_mat: np.ndarray) -> np.ndarray:
        N = delta_list.shape[-1]
        Ur = batch_eye_cpu(N, batch_shape=delta_list.shape[:-1], dtype=delta_list.dtype)

        # reconstruct from right to left as in the book chapter
        # cos_phi = np.cos(phi_list)
        # sin_phi = np.sin(phi_list)
        # print(phi_list)
        # print(cos_phi)
        # count = 0
        phi_mat_cos = np.cos(phi_mat)
        phi_mat_sin = np.sin(phi_mat)
        for i in range(N):
            for j in range(N - i - 1):
                # with TimerCtx() as t:
                # phi = phi_mat[i, j]
                # c = np.cos(phi)
                # s = np.sin(phi)
                # index = int((2 * N - i - 1) * i / 2 + j)
                # phi = phi_list[index]
                c, s = phi_mat_cos[..., i, j:j+1], phi_mat_sin[..., i, j:j+1]

                # print("cos:", t.interval)
                # c = cos_phi[count]
                # s = sin_phi[count]
                # count += 1
                # with TimerCtx() as t:
                p = i
                q = N - j - 1
                # Ur_new = Ur_old.clone()
                Ur[..., p, :], Ur[..., q, :] = Ur[..., p, :] * c - \
                    Ur[..., q, :] * s, Ur[..., p, :] * s + Ur[..., q, :] * c
                # res = (c + 1j * s) * (Ur[..., p, :] + 1j * Ur[..., q, :])
                # Ur[..., p, :], Ur[..., q, :] = res.real, res.imag
                # print("rotate:", t.interval)

        # D = tf.linalg.diag(
        #     diagonal=delta_list
        # )

        # # Ur = np.dot(D, Ur)
        # Ur = tf.matmul(D, Ur).numpy()
        Ur = delta_list[..., np.newaxis] * Ur
        # print(f'[I] Reconstruct {N}*{N} unitary matrix done')

        return Ur

    @profile(timer=timer)
    def reconstruct_francis(self, delta_list, phi_mat):
        ### Francis style, 1962
        N = delta_list.shape[0]
        Ur = np.identity(N)

        # reconstruct from right to left as in the book chapter
        # count = 0
        phi_mat_cos = np.cos(phi_mat)
        phi_mat_sin = np.sin(phi_mat)
        ### cannot gaurantee the phase range, so this will be slower
        # phi_mat_sin = (1 - phi_mat_cos*phi_mat_cos)**0.5
        # phi_mat_sin[phi_mat <= 0] *= -1
        for i in range(N):
            for j in range(N - i - 1):
                # with TimerCtx() as t:
                # phi = phi_mat[i, j]
                # c = np.cos(phi)
                # s = np.sin(phi)
                # index = int((2 * N - i - 1) * i / 2 + j)
                # phi = phi_list[index]
                c, s = phi_mat_cos[i, j], phi_mat_sin[i, j]

                # print("cos:", t.interval)
                # c = cos_phi[count]
                # s = sin_phi[count]
                # count += 1
                # with TimerCtx() as t:
                p = i
                q = N - j - 1
                # Ur_new = Ur_old.clone()
                row_p, row_q = Ur[p, :], Ur[q, :]
                row_p_cos, row_p_sin = row_p * c, row_p * s
                row_q_cos, row_q_sin = row_q * c, row_q * s
                Ur[p, :], Ur[q, :] = row_p_cos - row_q_sin, row_p_sin + row_q_cos
                # res = (c + 1j * s) * (row_p + 1j * row_q)
                # Ur[p, :], Ur[q, :] = res.real, res.imag
                # Ur[p, :], Ur[q, :] = Ur[p, :] * c - \
                #     Ur[q, :] * s, Ur[p, :] * s + Ur[q, :] * c
                # print("rotate:", t.interval)
        # D = np.diag(delta_list)
        # Ur = np.dot(D, Ur)
        Ur = delta_list[:, np.newaxis] * Ur
        # print(f'[I] Reconstruct {N}*{N} unitary matrix done')

        return Ur

    def reconstruct_2_batch(self, delta_list: np.ndarray, phi_mat: np.ndarray):
        logging.warning("This API is deprecated. Please use reconstruct_francis_batch instead")
        return self.reconstruct_francis_batch(delta_list, phi_mat)

    def reconstruct_2(self, delta_list, phi_mat):
        logging.warning("This API is deprecated. Please use reconstruct_francis instead")
        return self.reconstruct_francis(delta_list, phi_mat)

    def reconstruct(self, delta_list, phi_mat):
        if(self.alg == "francis"):
            reconstruct_cpu = self.reconstruct_francis
            reconstruct_batch = self.reconstruct_francis_batch
            reconstruct_cuda = matrix_parametrization_cuda.reconstruct_francis
        elif(self.alg == "reck"):
            reconstruct_cpu = self.reconstruct_reck_cpu
            reconstruct_batch = self.reconstruct_reck_batch
            reconstruct_cuda = matrix_parametrization_cuda.reconstruct_reck
        elif(self.alg == "clements"):
            reconstruct_cpu = self.reconstruct_clements_cpu
            reconstruct_batch = self.reconstruct_clements_batch
            reconstruct_cuda = matrix_parametrization_cuda.reconstruct_clements
        else:
            raise NotImplementedError

        if(isinstance(phi_mat, np.ndarray)):
            if(len(delta_list.shape) == 1):
                return reconstruct_cpu(delta_list, phi_mat)
            else:
                return reconstruct_batch(delta_list, phi_mat)
        else:
            if(phi_mat.is_cuda):
                size = phi_mat.size()
                N = phi_mat.size(-1)
                delta_list = delta_list.view(-1, N).to(phi_mat.device).contiguous()
                phi_mat = phi_mat.view(-1, N, N).contiguous()
                # U = torch.zeros_like(phi_mat).contiguous()
                U = reconstruct_cuda(delta_list, phi_mat)

                U = U.view(size)
                return U
            else:
                if(phi_mat.ndim == 2):
                    return torch.from_numpy(reconstruct_cpu(delta_list.cpu().numpy(), phi_mat.cpu().numpy()))
                else:
                    return torch.from_numpy(reconstruct_batch(delta_list.cpu().numpy(), phi_mat.cpu().numpy()))

    def decompose_clements_cpu(self, U):
        '''clements Optica 2018 unitary decomposition
        Tmn: [e^iphi x cos(theta)   -sin(theta)]
             [e^iphi x sin(theta)    cos(theta)]
        phi  DC   2 theta  DC ---
        ---  DC   -------  DC ---
        T45 T34 T23 T12 T45 T34 U T12* T34* T23* T12 = D
        U=D T34 T45 T12 T23 T34 T45 T12 T23 T34 T12'''
        N = U.shape[0]
        assert N > 0 and U.shape[0] == U.shape[1], '[E] Input matrix must be square and N > 0'

        phi_mat = np.zeros([N, N], dtype=self.dtype) ## theta checkerboard that maps to the real MZI mesh layout, which is efficient for parallel reconstruction col-by-col.
        # theta_checkerboard = np.zeros_like(U, dtype=self.dtype)
        # delta_list = np.zeros(N, dtype=self.dtype) ## D

        for i in range(N-1):
            ### each outer loop deals with one off-diagonal
            ## even loop for column rotation
            if(i % 2 == 0):
                for j in range(i + 1):
                    ### let p, q be the indices for the nullified '0'
                    p = N - 1 - j ## row
                    q = i - j ## col
                    ### rotate two columns such that u2 is nullified to 0
                    pi = np.pi
                    half_pi = np.pi / 2
                    min_err = self.min_err
                    u1, u2 = U[p, q+1], U[p, q]
                    u1_abs, u2_abs = np.abs(u1), np.abs(u2)
                    cond1, cond2 = u1_abs >= min_err, u2_abs >= min_err
                    if(cond1 & cond2):
                        phi = np.arctan2(-u2, u1)
                    elif(~cond1 & cond2):
                        phi = -half_pi if u2 > min_err else half_pi
                    elif(cond1 & ~cond2):
                        phi = 0 if u1 > min_err else -pi
                    else:
                        phi = 0
                    phi = -phi ### simply convert the solved theta from T to T*, it is easier than changing the solving procedure
                    # phi_mat[p,q] = phi
                    pairwise_index = i - j
                    # theta_checkerboard[pairwise_index, -j - 1] = phi
                    phi_mat[pairwise_index, j] = phi
                    c, s = np.cos(phi), np.sin(phi)
                    ## q_p1 means q+1; right multiply by T*
                    col_q_p1, col_q = U[:p+1, q+1], U[:p+1, q]
                    col_q_p1_cos, col_q_p1_sin = col_q_p1 * c, col_q_p1 * s
                    col_q_cos, col_q_sin = col_q * c, col_q * s
                    U[:p+1, q+1], U[:p+1, q] = col_q_p1_cos + col_q_sin, col_q_cos - col_q_p1_sin
            else:
                ## odd loop for row rotation
                for j in range(i+1):
                    p = N - 1 - i + j
                    q = j
                    ### rotate two rows such that u2 is nullified to 0
                    pi = np.pi
                    half_pi = np.pi / 2
                    min_err = self.min_err
                    u1, u2 = U[p-1, q], U[p, q]
                    u1_abs, u2_abs = np.abs(u1), np.abs(u2)
                    cond1, cond2 = u1_abs >= min_err, u2_abs >= min_err
                    if(cond1 & cond2):
                        phi = np.arctan2(-u2, u1)
                    elif(~cond1 & cond2):
                        phi = -half_pi if u2 > min_err else half_pi
                    elif(cond1 & ~cond2):
                        phi = 0 if u1 > min_err else -pi
                    else:
                        phi = 0
                    # phi_mat[p,q] = phi

                    pairwise_index = N + j - i - 2
                    # theta_checkerboard[pairwise_index, j] = phi
                    phi_mat[pairwise_index, N - 1 - j] = -phi ### from T* to T, consistent with propogation through MZI (T) see clements paper Eq.(4)
                    c, s = np.cos(phi), np.sin(phi)
                    ## p_1 means p - 1; left multiply by T
                    row_p_1, row_p = U[p-1, j:], U[p, j:]
                    row_p_1_cos, row_p_1_sin = row_p_1 * c, row_p_1 * s
                    row_p_cos, row_p_sin = row_p * c, row_p * s
                    U[p-1, j:], U[p, j:] = row_p_1_cos - row_p_sin, row_p_cos + row_p_1_sin
            # print(U)
        delta_list = np.diag(U) ## only the first and last element can be 1 or -1, the rest elements are all 1. This feature can be used in fast forward/reconstruction
        delta_list.setflags(write=True)
        # print("before flip:\n", theta_checkerboard)
        # theta_checkerboard = self.checkerboard_to_param(theta_checkerboard, N)
        # print("after flip:\n", theta_checkerboard)

        return delta_list, phi_mat

    def decompose_clements_batch(self, U):
        N = U.shape[-1]
        assert N > 0 and U.shape[-1] == U.shape[-2], '[E] Input matrix must be square and N > 0'

        phi_mat = np.zeros(U.shape, dtype=np.float64)
        delta_list = np.zeros(U.shape[:-1], dtype=np.float64)
        for i in range(N-1):
            ### each outer loop deals with one off-diagonal
            ## even loop for column rotation
            if(i % 2 == 0):
                for j in range(i + 1):
                    ### let p, q be the indices for the nullified '0'
                    p = N - 1 - j ## row
                    q = i - j ## col
                    ### rotate two columns such that u2 is nullified to 0
                    pi = np.pi
                    # half_pi = np.pi / 2
                    min_err = self.min_err
                    u1, u2 = U[..., p:p+1, q+1], U[..., p:p+1, q]
                    pi = np.pi
                    u1_abs, u2_abs = np.abs(u1), np.abs(u2)
                    min_err = self.min_err
                    cond1 = u1_abs < min_err
                    cond2 = u2_abs < min_err
                    cond1_n = ~cond1
                    cond2_n = ~cond2
                    phi = np.where(cond1 & cond2, 0,
                                np.where(cond1_n & cond2, np.where(u1 > min_err, 0, -pi),
                                            np.where(cond1 & cond2_n, np.where(u2 > min_err, -0.5*pi, 0.5*pi), np.arctan2(-u2, u1))))
                    phi = -phi ### simply convert the solved theta from T to T*, it is easier than changing the solving procedure
                    # phi_mat[p,q] = phi
                    pairwise_index = i - j
                    # theta_checkerboard[pairwise_index, -j - 1] = phi
                    phi_mat[..., pairwise_index, j] = phi[..., 0]
                    c, s = np.cos(phi), np.sin(phi)
                    ## q_p1 means q+1; right multiply by T*
                    col_q_p1, col_q = U[..., :p+1, q+1], U[..., :p+1, q]
                    col_q_p1_cos, col_q_p1_sin = col_q_p1 * c, col_q_p1 * s
                    col_q_cos, col_q_sin = col_q * c, col_q * s
                    U[..., :p+1, q+1], U[..., :p+1, q] = col_q_p1_cos + col_q_sin, col_q_cos - col_q_p1_sin
            else:
                ## odd loop for row rotation
                for j in range(i+1):
                    p = N - 1 - i + j
                    q = j
                    ### rotate two rows such that u2 is nullified to 0
                    pi = np.pi
                    # half_pi = np.pi / 2
                    min_err = self.min_err
                    u1, u2 = U[..., p-1, q:q+1], U[..., p, q:q+1]
                    pi = np.pi
                    u1_abs, u2_abs = np.abs(u1), np.abs(u2)
                    min_err = self.min_err
                    cond1 = u1_abs < min_err
                    cond2 = u2_abs < min_err
                    cond1_n = ~cond1
                    cond2_n = ~cond2
                    phi = np.where(cond1 & cond2, 0,
                                np.where(cond1_n & cond2, np.where(u1 > min_err, 0, -pi),
                                            np.where(cond1 & cond2_n, np.where(u2 > min_err, -0.5*pi, 0.5*pi), np.arctan2(-u2, u1))))

                    pairwise_index = N + j - i - 2
                    # theta_checkerboard[pairwise_index, j] = phi
                    phi_mat[..., pairwise_index, N - 1 - j] = -phi[..., 0] ### from T* to T, consistent with propogation through MZI (T) see clements paper Eq.(4)
                    c, s = np.cos(phi), np.sin(phi)
                    ## p_1 means p - 1; left multiply by T
                    row_p_1, row_p = U[..., p-1, j:], U[..., p, j:]
                    row_p_1_cos, row_p_1_sin = row_p_1 * c, row_p_1 * s
                    row_p_cos, row_p_sin = row_p * c, row_p * s
                    U[..., p-1, j:], U[..., p, j:] = row_p_1_cos - row_p_sin, row_p_cos + row_p_1_sin
            # print(U)
        # for i in range(N):
        #     delta_list[..., i] = U[..., i, i]## only the first and last element can be 1 or -1, the rest elements are all 1. This feature can be used in fast forward/reconstruction
        # delta_list = batch_diag(U)
        delta_list = batch_diag(U)
        return delta_list, phi_mat

    def decompose_clements(self, U):
        if(isinstance(U, np.ndarray)):
            if(len(U.shape) == 2):
                return self.decompose_clements_cpu(U)
            else:
                return self.decompose_clements_batch(U)
        else:
            if(U.is_cuda):
                N = U.size(-1)
                size = U.size()
                U = U.view(-1, N, N).contiguous()
                delta_list = torch.zeros(list(U.size())[:-1], dtype=U.dtype, device=U.device).contiguous()
                phi_mat = torch.zeros_like(U).contiguous()
                matrix_parametrization_cuda.decompose_clements(U, delta_list, phi_mat)
                delta_list = delta_list.view(list(size)[:-1])
                phi_mat = phi_mat.view(size)
                return delta_list, phi_mat
            else:
                if(U.ndim == 2):
                    return torch.from_numpy(self.decompose_clements_cpu(U.cpu().numpy()))
                else:
                    return torch.from_numpy(self.decompose_clements_batch(U.cpu().numpy()))

    def checkerboard_to_param(self,checkerboard: np.ndarray, units: int):
        param = np.zeros((units, units // 2))
        if units % 2:
            param[::2, :] = checkerboard.T[::2, :-1:2]
        else:
            param[::2, :] = checkerboard.T[::2, ::2]
        param[1::2, :] = checkerboard.T[1::2, 1::2]
        return param

    def reconstruct_clements_cpu(self, delta_list, phi_mat):
        N = delta_list.shape[0]
        Ur = np.identity(N)

        # parallelly reconstruct col by col based on the checkerboard (phi_mat)
        # count = 0
        phi_mat_cos = np.cos(phi_mat)
        phi_mat_sin = np.sin(phi_mat)


        for i in range(N): ## N layers
            max_len = 2 * (i + 1)
            ### in odd N, address delta_list[-1] before the first column
            if(i == 0 and N % 2 == 1 and delta_list[-1] < 0):
                Ur[-1, :] *= -1
            for j in range((i%2), N-1, 2):
                c, s = phi_mat_cos[j, i], phi_mat_sin[j, i]
                ## not the entire row needs to be rotated, only a small working set is used
                # row_p, row_q = Ur[j, :], Ur[j+1, :]
                lower = j - i
                upper = lower + max_len
                lower = max(0, lower)
                upper = min(upper, N)
                row_p, row_q = Ur[j, lower:upper], Ur[j+1, lower:upper]
                # row_p_cos, row_p_sin = row_p * c, row_p * s
                # row_q_cos, row_q_sin = row_q * c, row_q * s
                # Ur[j, lower:upper], Ur[j+1, lower:upper] = row_p_cos - row_q_sin, row_p_sin + row_q_cos
                res = (c+1j*s)*(row_p+1j*row_q)
                Ur[j, lower:upper], Ur[j+1, lower:upper] = res.real, res.imag
            if(i == 0 and N % 2 == 0 and delta_list[-1] < 0):
                Ur[-1, :] *= -1
            if(i == N-2 and N % 2 == 1 and delta_list[0] < 0): ## consider diagonal[0]= {-1,1} before the last layer when N odd
                Ur[0,:] *= -1
        if(N % 2 == 0 and delta_list[0] < 0): ## consider diagonal[0]= {-1,1} after the last layer when N even
            Ur[0,:] *= -1
        return Ur

    def reconstruct_clements_batch(self, delta_list, phi_mat):
        N = delta_list.shape[-1]
        Ur = batch_eye_cpu(N, batch_shape=delta_list.shape[:-1], dtype=delta_list.dtype)
        # parallelly reconstruct col by col based on the checkerboard (phi_mat)
        # count = 0
        phi_mat_cos = np.cos(phi_mat)
        phi_mat_sin = np.sin(phi_mat)
        for i in range(N): ## N layers
            max_len = 2 * (i + 1)
            ### in odd N, address delta_list[-1] before the first column
            if(i == 0 and N % 2 == 1):
                Ur[..., -1, :] *= delta_list[..., -1:]
            for j in range((i%2), N-1, 2):
                ## not the entire row needs to be rotated, only a small working set is used
                lower = j - i
                upper = lower + max_len
                lower = max(0, lower)
                upper = min(upper, N)
                c, s = phi_mat_cos[..., j, i:i+1], phi_mat_sin[..., j, i:i+1]
                # row_p, row_q = Ur[..., j, :], Ur[..., j+1, :]
                row_p, row_q = Ur[..., j, lower:upper], Ur[..., j+1, lower:upper]
                row_p_cos, row_p_sin = row_p * c, row_p * s
                row_q_cos, row_q_sin = row_q * c, row_q * s
                Ur[..., j, lower:upper], Ur[..., j+1, lower:upper] = row_p_cos - row_q_sin, row_p_sin + row_q_cos
                # res = (c+1j*s)*(row_p+1j*row_q)
                # Ur[..., j, lower:upper], Ur[..., j+1, lower:upper] = res.real, res.imag
            if(i == 0 and N % 2 == 0 ):
                Ur[..., -1, :] *= delta_list[..., -1:]
            if(i == N-2 and N % 2 == 1): ## consider diagonal[0]= {-1,1} before the last layer when N odd
                Ur[..., 0, :] *= delta_list[..., 0:1]
        if(N % 2 == 0): ## consider diagonal[0]= {-1,1} after the last layer when N even
            Ur[..., 0, :] *= delta_list[..., 0:1]

        return Ur

    def reconstruct_clements(self, delta_list, phi_mat):
        if(isinstance(phi_mat, np.ndarray)):
            if(len(delta_list.shape) == 1):
                return self.reconstruct_clements_cpu(delta_list, phi_mat)
            else:
                return self.reconstruct_clements_batch(delta_list, phi_mat)
        else:
            if(phi_mat.is_cuda):
                size = phi_mat.size()
                N = phi_mat.size(-1)
                delta_list = delta_list.view(-1, N).to(phi_mat.device).contiguous()
                phi_mat = phi_mat.view(-1, N, N).contiguous()
                # U = torch.zeros_like(phi_mat).contiguous()
                U = matrix_parametrization_cuda.reconstruct_clements(delta_list, phi_mat)

                U = U.view(size)
                return U
            else:
                if(phi_mat.dim() == 2):
                    return torch.from_numpy(self.reconstruct_clements(delta_list.cpu().numpy(), phi_mat.cpu().numpy()))
                else:
                    return torch.from_numpy(self.reconstruct_clements_batch(delta_list.cpu().numpy(), phi_mat.cpu().numpy()))

    def checkIdentity(self, M):
        return (M.shape[0] == M.shape[1]) and np.allclose(M, np.eye(M.shape[0]))

    def checkUnitary(self, U):
        M = np.dot(U, U.T)
        # print(M)
        return self.checkIdentity(M)

    def checkEqual(self, M1, M2):
        return (M1.shape == M2.shape) and np.allclose(M1, M2)

    def genRandomOrtho(self, N):
        U = ortho_group.rvs(N)
        print(
            f'[I] Generate random {N}*{N} unitary matrix, check unitary: {self.checkUnitary(U)}')
        return U

    def toDegree(self, M):
        return np.degrees(M)



class ComplexUnitaryDecomposerBatch(object):
    timer = False

    def __init__(self, min_err=1e-7, timer=False, determine=False, alg="reck", dtype=np.float64):
        self.min_err = min_err
        self.timer = timer
        self.determine = determine
        self.alg = alg
        assert alg.lower() in {"reck", "clements", "francis"}, logging.error(f"Unitary decomposition algorithm can only be [reck, clements, francis], but got {alg}")
        self.dtype = dtype

    def set_alg(self, alg):
        assert alg.lower() in {"reck", "clements", "francis"}, logging.error(f"Unitary decomposition algorithm can only be [reck, clements, francis], but got {alg}")
        self.alg = alg

    def buildPlaneUnitary(self, p, q, phi, N, transpose=True):
        assert N > 0 and isinstance(
            N, int), "[E] Matrix size must be positive integer"
        assert isinstance(p, int) and isinstance(q,
                                                 int) and 0 <= p < q < N, "[E] Integer value p and q must satisfy p < q"
        assert (isinstance(phi, float) or isinstance(phi, int)
                ), "[E] Value phi must be of type float or int"

        U = np.eye(N)
        c = np.cos(phi)
        s = np.sin(phi)

        U[p, p] = U[q, q] = c
        U[q, p] = s if not transpose else -s
        U[p, q] = -s if not transpose else s

        return U

    def calPhi_batch_determine(self, u1: np.ndarray, u2: np.ndarray, is_first_col=False) -> np.ndarray:
        pi = np.pi
        u1_abs, u2_abs = np.abs(u1), np.abs(u2)
        min_err = self.min_err
        cond1 = u1_abs < min_err
        cond2 = u2_abs < min_err
        cond1_n = ~cond1
        cond2_n = ~cond2
        if(is_first_col):
            phi = np.where(cond1 & cond2, 0,
                           np.where(cond1_n & cond2, np.where(u1 > min_err, 0, -pi),
                                    np.where(cond1 & cond2_n, np.where(u2 > min_err, -0.5*pi, 0.5*pi), np.arctan2(-u2, u1))))
        else:
            phi = np.where(cond1 & cond2, 0,
                           np.where(cond1_n & cond2, np.where(u1 > min_err, 0, -pi),
                                    np.where(cond1 & cond2_n, np.where(u2 > min_err, -0.5*pi, 0.5*pi), np.arctan(-u2/u1))))
        return phi

    def calPhi_batch_nondetermine(self, u1: np.ndarray, u2: np.ndarray, is_first_col=False) -> np.ndarray:
        pi = np.pi
        u1_abs, u2_abs = np.abs(u1), np.abs(u2)
        min_err = self.min_err
        cond1 = u1_abs < min_err
        cond2 = u2_abs < min_err
        cond1_n = ~cond1
        cond2_n = ~cond2
        phi = np.where(cond1 & cond2, 0,
                       np.where(cond1_n & cond2, np.where(u1 > min_err, 0, -pi),
                                np.where(cond1 & cond2_n, np.where(u2 > min_err, -0.5*pi, 0.5*pi), np.arctan2(-u2, u1))))

        return phi

    def calPhi_determine(self, u1, u2, is_first_col=False):
        pi = np.pi
        u1_abs, u2_abs = np.abs(u1), np.abs(u2)
        min_err = self.min_err

        if u1_abs < min_err and u2_abs < min_err:
            phi = 0
        elif u1_abs >= min_err and u2_abs < min_err:
            phi = 0 if u1 > min_err else -pi
        elif u1_abs < min_err and u2_abs >= min_err:
            phi = -0.5 * pi if u2 > min_err else 0.5 * pi
        else:
            # solve the equation: u'_1n=0
            if(is_first_col):
                phi = np.arctan2(-u2, u1)  # 4 quadrant4
            else:
                phi = np.arctan(-u2/u1)
            # phi = np.arctan(-u2/u1)
        # print(phi, u1, u2)

        # cond = ((u1_abs < min_err) << 1) | (u2_abs < min_err)
        # if(cond == 0):
        #     phi = np.arctan2(-u2, u1)
        # elif(cond == 1):
        #     phi = 0 if u1 > min_err else -np.pi
        # elif(cond == 2):
        #     phi = -0.5 * np.pi if u2 > min_err else 0.5 * np.pi
        # else:
        #     phi = 0
        # phi = [np.arctan2(-u2, u1), 0 if u1 > min_err else -np.pi, -0.5 * np.pi if u2 > min_err else 0.5 * np.pi, 0][cond]

        return phi

    def calPhi_nondetermine(self, u1, u2):
        pi = np.pi
        u1_abs, u2_abs = np.abs(u1), np.abs(u2)
        min_err = self.min_err

        if u1_abs < min_err and u2_abs < min_err:
            phi = 0
        elif u1_abs >= min_err and u2_abs < min_err:
            phi = 0 if u1 > min_err else -pi
        elif u1_abs < min_err and u2_abs >= min_err:
            phi = -0.5 * pi if u2 > min_err else 0.5 * pi
        else:
            # solve the equation: u'_1n=0
            phi = np.arctan2(-u2, u1)  # 4 quadrant4

        return phi

    def decomposeKernel_batch(self, U: np.ndarray, dim, phi_list=None):
        '''return U^(N-1); (phi_1,...,phi_N-2); (sigma_1,...,sigma_N-2)'''
        N = U.shape[-1]
        if(phi_list is None):
            phi_list = np.zeros(list(U.shape[:-2])+[dim], dtype=np.float64)

        calPhi_batch = self.calPhi_batch_determine if self.determine else self.calPhi_batch_nondetermine
        for i in range(N - 1):
            # with TimerCtx() as t:
            u1, u2 = U[..., 0, 0], U[..., 0, N - 1 - i]
            phi = calPhi_batch(u1, u2, is_first_col=(i == 0))

            # print("calPhi:", t.interval)
            # with TimerCtx() as t:
            phi_list[..., i] = phi

            p, q = 0, N - i - 1
            # U = np.dot(U, self.buildPlaneUnitary(p=p, q=q, phi=phi, N=N, transpose=True))
            c, s = np.cos(phi)[..., np.newaxis], np.sin(phi)[..., np.newaxis]
            col_p, col_q = U[..., :, p], U[..., :, q]
            U[..., :, p], U[..., :, q] = col_p * \
                c - col_q * s, col_p * s + col_q * c
            # U[:, p], U[:, q] = U[:, p] * c - U[:, q] * s, U[:, p] * s + U[:, q] * c
            # print("rotate:", t.interval)

        # print(f'[I] Decomposition kernel done')
        return U, phi_list

    def decomposeKernel_determine(self, U, phi_list):
        '''return U^(N-1); (phi_1,...,phi_N-2); (sigma_1,...,sigma_N-2)'''
        N = U.shape[0]
        # if(phi_list is None):
        #     phi_list = np.zeros(dim, dtype=self.dtype)

        # calPhi = self.calPhi_determine
        # cos, sin = np.cos, np.sin
        for i in range(N - 1):
            # with TimerCtx() as t:
            u1, u2 = U[0, 0], U[0, N - 1 - i]
            # phi = calPhi(u1, u2, is_first_col=(i == 0))
            pi = np.pi
            u1_abs, u2_abs = np.abs(u1), np.abs(u2)
            min_err = self.min_err

            if u1_abs < min_err and u2_abs < min_err:
                phi = 0
            elif u1_abs >= min_err and u2_abs < min_err:
                phi = 0 if u1 > min_err else -pi
            elif u1_abs < min_err and u2_abs >= min_err:
                phi = -0.5 * pi if u2 > min_err else 0.5 * pi
            else:
                # solve the equation: u'_1n=0
                if(i==0):
                    phi = np.arctan2(-u2, u1)  # 4 quadrant4
                else:
                    phi = np.arctan(-u2/u1)

            # print("calPhi:", t.interval)
            # with TimerCtx() as t:
            phi_list[i] = phi
            p, q = 0, N - i - 1
            # U = np.dot(U, self.buildPlaneUnitary(p=p, q=q, phi=phi, N=N, transpose=True))
            c, s = np.cos(phi), np.sin(phi)
            row_p, row_q = U[:, p], U[:, q]
            row_p_cos, row_p_sin = row_p * c, row_p * s
            row_q_cos, row_q_sin = row_q * c, row_q * s
            U[:, p], U[:, q] = row_p_cos - row_q_sin, row_q_cos + row_p_sin
            # U[:, p], U[:, q] = row_p * c - row_q * s, row_q*c + row_p*s
            # U[:, p], U[:, q] = U[:, p] * c - U[:, q] * s, U[:, p] * s + U[:, q] * c
            # print("rotate:", t.interval)

        # print(f'[I] Decomposition kernel done')
        return U, phi_list

    def decomposeKernel_nondetermine(self, U, phi_list):
        '''return U^(N-1); (phi_1,...,phi_N-2); (sigma_1,...,sigma_N-2)'''
        N = U.shape[0]
        # if(phi_list is None):
        #     phi_list = np.zeros(dim, dtype=self.dtype)

        # calPhi = self.calPhi_nondetermine
        # cos, sin = np.cos, np.sin
        pi = np.pi
        half_pi = np.pi / 2
        min_err = self.min_err
        for i in range(N - 1):
            # with TimerCtx() as t:
            u1, u2 = U[0, 0], U[0, N - 1 - i]
            u1_abs, u2_abs = np.abs(u1), np.abs(u2)
            cond1, cond2 = u1_abs >= min_err, u2_abs >= min_err
            if(cond1 & cond2):
                phi = np.arctan2(-u2, u1)
            elif(~cond1 & cond2):
                phi = -half_pi if u2 > min_err else half_pi
            elif(cond1 & ~cond2):
                phi = 0 if u1 > min_err else -pi
            else:
                phi = 0

            # print("calPhi:", t.interval)
            # with TimerCtx() as t:
            phi_list[i] = phi
            p, q = 0, N - i - 1
            # U = np.dot(U, self.buildPlaneUnitary(p=p, q=q, phi=phi, N=N, transpose=True))
            # c, s = np.cos(phi), np.sin(phi)
            c = np.cos(phi)
            s = (1 - c*c)**0.5 if phi > 0 else -((1 - c*c)**0.5)
            row_p, row_q = U[:, p], U[:, q]
            row_p_cos, row_p_sin = row_p * c, row_p * s
            row_q_cos, row_q_sin = row_q * c, row_q * s
            U[:, p], U[:, q] = row_p_cos - row_q_sin, row_q_cos + row_p_sin
            # U[:, p], U[:, q] = row_p * c - row_q * s, row_q*c + row_p*s
            # U[:, p], U[:, q] = U[:, p] * c - U[:, q] * s, U[:, p] * s + U[:, q] * c
            # print("rotate:", t.interval)

        # print(f'[I] Decomposition kernel done')
        return U, phi_list

    @profile(timer=timer)
    def decompose_francis_batch(self, U: np.ndarray):
        N = U.shape[-1]
        assert N > 0 and U.shape[-1] == U.shape[-2], '[E] Input matrix must be square and N > 0'

        phi_mat = np.zeros(U.shape, dtype=np.float64)
        delta_list = np.zeros(U.shape[:-1], dtype=np.float64)

        for i in range(N - 1):
            # U, phi_list = self.decomposeKernel(U, dim=N)
            # phi_mat[i, :] = phi_list
            U, _ = self.decomposeKernel_batch(
                U, dim=N, phi_list=phi_mat[..., i, :])
            delta_list[..., i] = U[..., 0, 0]
            U = U[..., 1:, 1:]
        else:
            delta_list[..., -1] = U[..., -1, -1]

        # print(f'[I] Decomposition done')
        return delta_list, phi_mat

    @profile(timer=timer)
    def decompose_francis_cpu(self, U):
        #### This decomposition has follows the natural reflection of MZIs. Thus the circuit will give a reversed output.
        ### Francis style, 1962
        N = U.shape[0]
        assert N > 0 and U.shape[0] == U.shape[1], '[E] Input matrix must be square and N > 0'

        phi_mat = np.zeros([N, N], dtype=self.dtype)
        delta_list = np.zeros(N, dtype=self.dtype)
        decompose_kernel = self.decomposeKernel_determine if self.determine else self.decomposeKernel_nondetermine

        for i in range(N - 1):
            # U, phi_list = self.decomposeKernel(U, dim=N)
            # phi_mat[i, :] = phi_list
            U, _ = decompose_kernel(U, phi_list=phi_mat[i, :])
            # U, _ = self.decomposeKernel(U, dim=N, phi_list=phi_mat[i, :])
            delta_list[i] = U[0, 0]
            U = U[1:, 1:]
        else:
            delta_list[-1] = U[-1, -1]

        # print(f'[I] Decomposition done')
        return delta_list, phi_mat

    def decompose(self, U):
        if(self.alg == "reck"):
            decompose_cpu = self.decompose_reck_cpu
            decompose_batch = self.decompose_reck_batch
            decompose_cuda = matrix_parametrization_cuda.decompose_reck
        elif(self.alg == "francis"):
            decompose_cpu = self.decompose_francis_cpu
            decompose_batch = self.decompose_francis_batch
            decompose_cuda = matrix_parametrization_cuda.decompose_francis
        elif(self.alg == "clements"):
            decompose_cpu = self.decompose_clements_cpu
            decompose_batch = self.decompose_clements_batch
            decompose_cuda = matrix_parametrization_cuda.decompose_clements
        else:
            raise NotImplementedError

        if(isinstance(U, np.ndarray)):
            if(len(U.shape) == 2):
                return decompose_cpu(U)
            else:
                return decompose_batch(U)
        else:
            if(U.is_cuda):
                N = U.size(-1)
                size = U.size()
                U = U.view(-1, N, N).contiguous()
                delta_list = torch.zeros(list(U.size())[:-1], dtype=U.dtype, device=U.device).contiguous()
                phi_mat = torch.zeros_like(U).contiguous()
                decompose_cuda(U, delta_list, phi_mat)
                delta_list = delta_list.view(list(size)[:-1])
                phi_mat = phi_mat.view(size)
                return delta_list, phi_mat
            else:
                if(U.dim() == 2):
                    return torch.from_numpy(decompose_cpu(U.cpu().numpy()))
                else:
                    return torch.from_numpy(decompose_batch(U.cpu().numpy()))

    def decompose_reck_cpu(self, U):
        '''Reck decomposition implemented by Neurophox. Triangular mesh, input and output have no mirroring effects, i.e, [x1, ..., xn] -> Y = U x X -> [y1, ..., yn]
        Rmn: [ cos(phi)   -sin(phi)] -> MZI achieves counter-clock-wise rotation with phi (reconstruction, left mul)
             [ sin(phi)    cos(phi)]
        Rmn*:[ cos(phi)    sin(phi)] -> column-wise clock-wise rotation (decompose, right mul)
             [-sin(phi)    cos(phi)]

        U = D R43 R32 R43 R21 R32 R43
        '''
        N = U.shape[0]
        assert N > 0 and U.shape[0] == U.shape[1], '[E] Input matrix must be square and N > 0'

        phi_mat = np.zeros([N, N, 4], dtype=self.dtype) ## phase shifter, theta_t, theta_l, omega_p, omega_w
        # theta_t_mat = np.zeros([N, N], dtype=self.dtype) ## top external phase shifter
        # theta_l_mat = np.zeros([N, N], dtype=self.dtype) ## left external phase shifter
        # omega_p_mat = np.zeros([N, N], dtype=self.dtype) ## upper internal phase shifter
        # omega_w_mat = np.zeros([N, N], dtype=self.dtype) ## lower internal phase shifter
        '''
        the bottom-left phase corresponds to the MZI at the bottom-left corner.
        The decomposition ordering follows from bottom to top, from left to right.
        R21 R32 R43  0
        R32 R43 0    0
        R43 0   0    0
        0   0   0    0
        '''
        # theta_checkerboard = np.zeros_like(U, dtype=self.dtype)
        # delta_list = np.zeros(N, dtype=self.dtype) ## D
        '''
            x x x 0     x x 0 0
            x x x x  -> x x x 0
            x x x x     x x x x
            x x x x     x x x x
        '''

        for i in range(N-1):
            ### each outer loop deals with one off-diagonal, nullification starts from top-right
            ### even loop for column rotation
            for j in range(i + 1):
                ### let p, q be the indices for the nullified '0'
                p = j ## row
                q = N - 1 - i + j ## col
                ### rotate two columns such that u2 is nullified to 0
                pi = np.pi
                half_pi = np.pi / 2
                min_err = self.min_err
                ### col q-1 nullifies col q
                u1, u2 = U[p, q-1], U[p, q]
                u1_abs, u2_abs = np.abs(u1), np.abs(u2)
                cond1, cond2 = u1_abs >= min_err, u2_abs >= min_err
                if(cond1 & cond2):
                    phi = np.arctan2(-u2, u1)
                elif(~cond1 & cond2):
                    phi = -half_pi if u2 > min_err else half_pi
                elif(cond1 & ~cond2):
                    phi = 0 if u1 > min_err else -pi
                else:
                    phi = 0
                if (phi < -pi/2):
                    phi += 2*np.pi ## [-pi/2, 3pi/2]

                phi_mat[N-i-2, j, 0] = np.pi/2 ## this absorbs the global phase theta_tot
                phi_mat[N-i-2, j, 1] = 3*np.pi/2
                phi_mat[N-i-2, j, 2] = 1.5*np.pi - phi
                phi_mat[N-i-2, j, 3] = 0.5*np.pi + phi
                c, s = np.cos(phi), np.sin(phi)
                ## q_m1 means q-1; right multiply by R*
                col_q_m1, col_q = U[p:, q-1], U[p:, q]
                col_q_m1_cos, col_q_m1_sin = col_q_m1 * c, col_q_m1 * s
                col_q_cos, col_q_sin = col_q * c, col_q * s
                U[p:, q-1], U[p:, q] = col_q_m1_cos - col_q_sin, col_q_cos + col_q_m1_sin

        delta_list = np.angle(np.diag(U)) ## only the first and last element can be 1 or -1, the rest elements are all 1. This feature can be used in fast forward/reconstruction

        return delta_list, phi_mat

    def decompose_reck_batch(self, U):
        '''Reck decomposition implemented by Neurophox. Triangular mesh, input and output have no mirroring effects, i.e, [x1, ..., xn] -> Y = U x X -> [y1, ..., yn]
        Rmn: [ cos(phi)   -sin(phi)] -> MZI achieves counter-clock-wise rotation with phi (reconstruction, left mul)
             [ sin(phi)    cos(phi)]
        Rmn*:[ cos(phi)    sin(phi)] -> column-wise clock-wise rotation (decompose, right mul)
             [-sin(phi)    cos(phi)]

        U = D R43 R32 R43 R21 R32 R43
        U is real matrix
        '''
        N = U.shape[-1]
        assert N > 0 and U.shape[-1] == U.shape[-2], '[E] Input matrix must be square and N > 0'

        phi_mat = np.zeros(list(U.shape) + [4], dtype=self.dtype) ## left upper triangular array.
        '''
        the bottom-left phase corresponds to the MZI at the bottom-left corner.
        The decomposition ordering follows from bottom to top, from left to right.
        R21 R32 R43  0
        R32 R43 0    0
        R43 0   0    0
        0   0   0    0
        '''
        # theta_checkerboard = np.zeros_like(U, dtype=self.dtype)
        # delta_list = np.zeros(U.shape[:-1], dtype=self.dtype) ## D
        '''
            x x x 0     x x 0 0
            x x x x  -> x x x 0
            x x x x     x x x x
            x x x x     x x x x
        '''

        for i in range(N-1):
            ### each outer loop deals with one off-diagonal, nullification starts from top-right
            ### even loop for column rotation
            for j in range(i + 1):
                ### let p, q be the indices for the nullified '0'
                p = j ## row
                q = N - 1 - i + j ## col
                ### rotate two columns such that u2 is nullified to 0
                ### col q-1 nullifies col q

                u1, u2 = U[..., p, q-1], U[..., p, q]
                phi = self.calPhi_batch_nondetermine(u1, u2)
                phi[phi<-np.pi/2] += 2*np.pi ## [-pi/2, 3pi/2]

                phi_mat[..., N-i-2, j, 0] = np.pi/2 ## this absorbs the global phase theta_tot
                phi_mat[..., N-i-2, j, 1] = 3*np.pi/2
                phi_mat[..., N-i-2, j, 2] = 1.5*np.pi - phi
                phi_mat[..., N-i-2, j, 3] = 0.5*np.pi + phi
                c, s = np.cos(phi)[..., np.newaxis], np.sin(phi)[..., np.newaxis]
                ## q_m1 means q-1; right multiply by R*
                col_q_m1, col_q = U[..., p:, q-1], U[..., p:, q]
                col_q_m1_cos, col_q_m1_sin = col_q_m1 * c, col_q_m1 * s
                col_q_cos, col_q_sin = col_q * c, col_q * s
                U[..., p:, q-1], U[..., p:, q] = col_q_m1_cos - col_q_sin, col_q_cos + col_q_m1_sin

        delta_list = np.angle(batch_diag(U))

        return delta_list, phi_mat

    def reconstruct_reck_cpu(self, delta_list, phi_mat):
        N = delta_list.shape[0]
        Ur = np.identity(N, dtype=np.complex64)
        ### left multiply by a counter-clock-wise rotation
        '''
        cos, -sin
        sin, cos
        '''

        for i in range(N-1):
            lower = N - 2 - i
            for j in range(i+1):
                phi = phi_mat[lower, j, :]
                p = N - 2 - i + j
                q = p + 1

                row_p, row_q = Ur[p, lower:], Ur[q, lower:]

                row_p *= np.exp(1j*(phi[0]+(phi[2]+phi[3])/2+np.pi/2))
                row_q *= np.exp(1j*(phi[1]+(phi[2]+phi[3])/2+np.pi/2))
                half_delta_theta = (phi[2]-phi[3])/2
                c, s = np.cos(half_delta_theta), np.sin(half_delta_theta)
                row_p_cos, row_p_sin = row_p * c, row_p * s
                row_q_cos, row_q_sin = row_q * c, row_q * s
                Ur[p, lower:], Ur[q, lower:] = row_p_sin + row_q_cos, row_p_cos - row_q_sin

        Ur = np.exp(1j*delta_list[:, np.newaxis]) * Ur
        return Ur

    def reconstruct_reck_batch(self, delta_list, phi_mat):
        N = delta_list.shape[-1]
        Ur = batch_eye_cpu(N, batch_shape=delta_list.shape[:-1], dtype=np.complex64)

        ### left multiply by a counter-clock-wise rotation
        '''
        cos, -sin
        sin, cos
        '''

        # phi_mat_cos = np.cos(phi_mat)
        # phi_mat_sin = np.sin(phi_mat)
        # e_jphi = np.exp(1j*phi_mat)

        for i in range(N-1):
            lower = N - 2 - i
            for j in range(i+1):
                phi = phi_mat[..., lower, j:j+1, :]
                # c, s = phi_mat_cos[..., lower, j:j+1], phi_mat_sin[..., lower, j:j+1]
                p = N - 2 - i + j
                q = p + 1
                row_p, row_q = Ur[..., p, lower:], Ur[..., q, lower:]
                row_p *= np.exp(1j*(phi[..., 0]+(phi[..., 2]+phi[..., 3])/2+np.pi/2))
                row_q *= np.exp(1j*(phi[..., 1]+(phi[..., 2]+phi[..., 3])/2+np.pi/2))
                half_delta_theta = (phi[..., 2]-phi[..., 3])/2
                c, s = np.cos(half_delta_theta), np.sin(half_delta_theta)
                row_p_cos, row_p_sin = row_p * c, row_p * s
                row_q_cos, row_q_sin = row_q * c, row_q * s
                Ur[..., p, lower:], Ur[..., q, lower:] = row_p_sin + row_q_cos, row_p_cos - row_q_sin
                ### this rotation is equivalent to complex number multiplication as an acceleration.

        Ur = np.exp(1j*delta_list[..., np.newaxis]) * Ur
        return Ur

    def reconstruct_reck_batch_par(self, delta_list, phi_mat):
        N = delta_list.shape[-1]
        Ur = batch_eye_cpu(N, batch_shape=delta_list.shape[:-1], dtype=delta_list.dtype)
        ### left multiply by a counter-clock-wise rotation
        '''
        cos, -sin
        sin, cos
        '''

        phi_mat_cos = np.cos(phi_mat)
        phi_mat_sin = np.sin(phi_mat)

        ### 2n-3 stages
        for i in range(2 * N - 3):
            lower = N - 2 - i
            for j in range(i + 1):
                c, s = phi_mat_cos[..., lower, j:j+1], phi_mat_sin[..., lower, j:j+1]
                p = N - 2 - i + j
                q = p + 1
                row_p, row_q = Ur[..., p, lower:], Ur[..., q, lower:]
                row_p_cos, row_p_sin = row_p * c, row_p * s
                row_q_cos, row_q_sin = row_q * c, row_q * s
                Ur[..., p, lower:], Ur[..., q, lower:] = row_p_cos - row_q_sin, row_p_sin + row_q_cos
        Ur = delta_list[..., np.newaxis] * Ur
        return Ur

    @profile(timer=timer)
    def reconstruct_slow(self, delta_list, phi_mat):
        N = delta_list.shape[0]
        Ur = np.identity(N)

        # reconstruct from right to left as in the book chapter
        for i in range(N):
            for j in range(N - i - 1):
                phi = phi_mat[i, j]
                Ur = np.dot(self.buildPlaneUnitary(
                    i, N - j - 1, phi, N, transpose=False), Ur)

        D = np.diag(delta_list)
        Ur = np.dot(D, Ur)
        print(f'[I] Reconstruct {N}*{N} unitary matrix done')

        return Ur

    @profile(timer=timer)
    def reconstruct_francis_batch(self, delta_list: np.ndarray, phi_mat: np.ndarray) -> np.ndarray:
        N = delta_list.shape[-1]
        Ur = batch_eye_cpu(N, batch_shape=delta_list.shape[:-1], dtype=delta_list.dtype)

        # reconstruct from right to left as in the book chapter
        # cos_phi = np.cos(phi_list)
        # sin_phi = np.sin(phi_list)
        # print(phi_list)
        # print(cos_phi)
        # count = 0
        phi_mat_cos = np.cos(phi_mat)
        phi_mat_sin = np.sin(phi_mat)
        for i in range(N):
            for j in range(N - i - 1):
                # with TimerCtx() as t:
                # phi = phi_mat[i, j]
                # c = np.cos(phi)
                # s = np.sin(phi)
                # index = int((2 * N - i - 1) * i / 2 + j)
                # phi = phi_list[index]
                c, s = phi_mat_cos[..., i, j:j+1], phi_mat_sin[..., i, j:j+1]

                # print("cos:", t.interval)
                # c = cos_phi[count]
                # s = sin_phi[count]
                # count += 1
                # with TimerCtx() as t:
                p = i
                q = N - j - 1
                # Ur_new = Ur_old.clone()
                Ur[..., p, :], Ur[..., q, :] = Ur[..., p, :] * c - \
                    Ur[..., q, :] * s, Ur[..., p, :] * s + Ur[..., q, :] * c
                # res = (c + 1j * s) * (Ur[..., p, :] + 1j * Ur[..., q, :])
                # Ur[..., p, :], Ur[..., q, :] = res.real, res.imag
                # print("rotate:", t.interval)

        # D = tf.linalg.diag(
        #     diagonal=delta_list
        # )

        # # Ur = np.dot(D, Ur)
        # Ur = tf.matmul(D, Ur).numpy()
        Ur = delta_list[..., np.newaxis] * Ur
        # print(f'[I] Reconstruct {N}*{N} unitary matrix done')

        return Ur

    @profile(timer=timer)
    def reconstruct_francis(self, delta_list, phi_mat):
        ### Francis style, 1962
        N = delta_list.shape[0]
        Ur = np.identity(N)

        # reconstruct from right to left as in the book chapter
        # count = 0
        phi_mat_cos = np.cos(phi_mat)
        phi_mat_sin = np.sin(phi_mat)
        ### cannot gaurantee the phase range, so this will be slower
        # phi_mat_sin = (1 - phi_mat_cos*phi_mat_cos)**0.5
        # phi_mat_sin[phi_mat <= 0] *= -1
        for i in range(N):
            for j in range(N - i - 1):
                # with TimerCtx() as t:
                # phi = phi_mat[i, j]
                # c = np.cos(phi)
                # s = np.sin(phi)
                # index = int((2 * N - i - 1) * i / 2 + j)
                # phi = phi_list[index]
                c, s = phi_mat_cos[i, j], phi_mat_sin[i, j]

                # print("cos:", t.interval)
                # c = cos_phi[count]
                # s = sin_phi[count]
                # count += 1
                # with TimerCtx() as t:
                p = i
                q = N - j - 1
                # Ur_new = Ur_old.clone()
                row_p, row_q = Ur[p, :], Ur[q, :]
                row_p_cos, row_p_sin = row_p * c, row_p * s
                row_q_cos, row_q_sin = row_q * c, row_q * s
                Ur[p, :], Ur[q, :] = row_p_cos - row_q_sin, row_p_sin + row_q_cos
                # res = (c + 1j * s) * (row_p + 1j * row_q)
                # Ur[p, :], Ur[q, :] = res.real, res.imag
                # Ur[p, :], Ur[q, :] = Ur[p, :] * c - \
                #     Ur[q, :] * s, Ur[p, :] * s + Ur[q, :] * c
                # print("rotate:", t.interval)
        # D = np.diag(delta_list)
        # Ur = np.dot(D, Ur)
        Ur = delta_list[:, np.newaxis] * Ur
        # print(f'[I] Reconstruct {N}*{N} unitary matrix done')

        return Ur

    def reconstruct_2_batch(self, delta_list: np.ndarray, phi_mat: np.ndarray):
        logging.warning("This API is deprecated. Please use reconstruct_francis_batch instead")
        return self.reconstruct_francis_batch(delta_list, phi_mat)

    def reconstruct_2(self, delta_list, phi_mat):
        logging.warning("This API is deprecated. Please use reconstruct_francis instead")
        return self.reconstruct_francis(delta_list, phi_mat)

    def reconstruct(self, delta_list, phi_mat):
        if(self.alg == "francis"):
            reconstruct_cpu = self.reconstruct_francis
            reconstruct_batch = self.reconstruct_francis_batch
            reconstruct_cuda = matrix_parametrization_cuda.reconstruct_francis
        elif(self.alg == "reck"):
            reconstruct_cpu = self.reconstruct_reck_cpu
            reconstruct_batch = self.reconstruct_reck_batch
            reconstruct_cuda = matrix_parametrization_cuda.reconstruct_reck
        elif(self.alg == "clements"):
            reconstruct_cpu = self.reconstruct_clements_cpu
            reconstruct_batch = self.reconstruct_clements_batch
            reconstruct_cuda = matrix_parametrization_cuda.reconstruct_clements
        else:
            raise NotImplementedError

        if(isinstance(phi_mat, np.ndarray)):
            if(len(delta_list.shape) == 1):
                return reconstruct_cpu(delta_list, phi_mat)
            else:
                return reconstruct_batch(delta_list, phi_mat)
        else:
            if(phi_mat.is_cuda):
                size = phi_mat.size()
                N = phi_mat.size(-1)
                delta_list = delta_list.view(-1, N).to(phi_mat.device).contiguous()
                phi_mat = phi_mat.view(-1, N, N).contiguous()
                # U = torch.zeros_like(phi_mat).contiguous()
                U = reconstruct_cuda(delta_list, phi_mat)

                U = U.view(size)
                return U
            else:
                if(phi_mat.dim() == 2):
                    return torch.from_numpy(reconstruct_cpu(delta_list.cpu().numpy(), phi_mat.cpu().numpy()))
                else:
                    return torch.from_numpy(reconstruct_batch(delta_list.cpu().numpy(), phi_mat.cpu().numpy()))

    def decompose_clements_cpu(self, U):
        '''clements Optica 2018 unitary decomposition
        Tmn: [e^iphi x cos(theta)   -sin(theta)]
             [e^iphi x sin(theta)    cos(theta)]
        phi  DC   2 theta  DC ---
        ---  DC   -------  DC ---
        T45 T34 T23 T12 T45 T34 U T12* T34* T23* T12 = D
        U=D T34 T45 T12 T23 T34 T45 T12 T23 T34 T12'''
        N = U.shape[0]
        assert N > 0 and U.shape[0] == U.shape[1], '[E] Input matrix must be square and N > 0'

        phi_mat = np.zeros([N, N, 4], dtype=self.dtype) ## theta checkerboard that maps to the real MZI mesh layout, which is efficient for parallel reconstruction col-by-col.
        # theta_checkerboard = np.zeros_like(U, dtype=self.dtype)

        pi = np.pi
        half_pi = np.pi / 2
        min_err = self.min_err

        for i in range(N-1):
            ### each outer loop deals with one off-diagonal
            ## even loop for column rotation
            if(i % 2 == 0):
                for j in range(i + 1):
                    ### let p, q be the indices for the nullified '0'
                    p = N - 1 - j ## row
                    q = i - j ## col
                    ### rotate two columns such that u2 is nullified to 0
                    u1, u2 = U[p, q+1], U[p, q]
                    u1_abs, u2_abs = np.abs(u1), np.abs(u2)
                    cond1, cond2 = u1_abs >= min_err, u2_abs >= min_err
                    if(cond1 & cond2):
                        phi = np.arctan2(-u2, u1)
                    elif(~cond1 & cond2):
                        phi = -half_pi if u2 > min_err else half_pi
                    elif(cond1 & ~cond2):
                        phi = 0 if u1 > min_err else -pi
                    else:
                        phi = 0
                    phi = -phi ### simply convert the solved theta from T to T*, it is easier than changing the solving procedure
                    if(phi < -pi/2):
                        phi += 2*pi
                    # phi_mat[p,q] = phi
                    pairwise_index = i - j
                    # theta_checkerboard[pairwise_index, -j - 1] = phi
                    # phi_mat[pairwise_index, j] = phi
                    phi_mat[pairwise_index, j, 0] = np.pi/2 ## this absorbs the global phase theta_tot
                    phi_mat[pairwise_index, j, 1] = 3*np.pi/2
                    phi_mat[pairwise_index, j, 2] = 1.5*np.pi - phi
                    phi_mat[pairwise_index, j, 3] = 0.5*np.pi + phi
                    c, s = np.cos(phi), np.sin(phi)
                    ## q_p1 means q+1; right multiply by T*
                    col_q_p1, col_q = U[:p+1, q+1], U[:p+1, q]
                    col_q_p1_cos, col_q_p1_sin = col_q_p1 * c, col_q_p1 * s
                    col_q_cos, col_q_sin = col_q * c, col_q * s
                    U[:p+1, q+1], U[:p+1, q] = col_q_p1_cos + col_q_sin, col_q_cos - col_q_p1_sin
            else:
                ## odd loop for row rotation
                for j in range(i+1):
                    p = N - 1 - i + j
                    q = j
                    ### rotate two rows such that u2 is nullified to 0
                    pi = np.pi
                    half_pi = np.pi / 2
                    min_err = self.min_err
                    u1, u2 = U[p-1, q], U[p, q]
                    u1_abs, u2_abs = np.abs(u1), np.abs(u2)
                    cond1, cond2 = u1_abs >= min_err, u2_abs >= min_err
                    if(cond1 & cond2):
                        phi = np.arctan2(-u2, u1)
                    elif(~cond1 & cond2):
                        phi = -half_pi if u2 > min_err else half_pi
                    elif(cond1 & ~cond2):
                        phi = 0 if u1 > min_err else -pi
                    else:
                        phi = 0
                    phi = -phi
                    if(phi < -pi/2):
                        phi += 2*pi

                    pairwise_index = N + j - i - 2
                    # theta_checkerboard[pairwise_index, j] = phi
                    # phi_mat[pairwise_index, N - 1 - j] = phi ### from T* to T, consistent with propogation through MZI (T) see clements paper Eq.(4)
                    phi_mat[pairwise_index, N - 1 - j, 0] = np.pi/2 ## this absorbs the global phase theta_tot
                    phi_mat[pairwise_index, N - 1 - j, 1] = 3*np.pi/2
                    phi_mat[pairwise_index, N - 1 - j, 2] = 1.5*np.pi - phi
                    phi_mat[pairwise_index, N - 1 - j, 3] = 0.5*np.pi + phi
                    c, s = np.cos(phi), np.sin(phi)
                    ## p_1 means p - 1; left multiply by T
                    row_p_1, row_p = U[p-1, j:], U[p, j:]
                    row_p_1_cos, row_p_1_sin = row_p_1 * c, row_p_1 * s
                    row_p_cos, row_p_sin = row_p * c, row_p * s
                    U[p-1, j:], U[p, j:] = row_p_1_cos + row_p_sin, row_p_cos - row_p_1_sin
            # print(U)
        delta_list = np.angle(np.diag(U))

        ### efficiently absorb delta_list into theta_t and theta_l and move delta_list to the last phase shifter column
        # if(N % 2 == 1):
        #     ### can completely absorb delta_list
        #     phi_mat[N-2, 1, 1] += delta_list[-1]
        #     for i in range(1, N):
        #         phi_mat[N-i-1, i, 0] += delta_list[N-i-1]
        # else:
        #     ## cannot absorb delta_list[0] because we do not have right and bottom phase shifters
        #     ## delta_list[0] has to be addressed during reconstruction
        #     phi_mat[N-2, 2, 1] += delta_list[-1]
        #     for i in range(2, N):
        #         phi_mat[N-i, i, 0] += delta_list[N-i]

        ### efficiently absorb delta_list into theta_t and theta_l and move delta_list to the last phase shifter column
        ### since U is real matrix, only delta_list[0] and delta_list[-1] can be -1.
        if(N % 2 == 1):
            phi_mat[0, -1, 0] += delta_list[0]
            delta_list[0] = 0
            phi_mat[N-2, 1, 1] += delta_list[-1]
            delta_list[-1] = 0
        else:
            phi_mat[N-2, 2, 1] += delta_list[-1]
            delta_list[-1] = 0


        return delta_list, phi_mat

    def decompose_clements_batch(self, U):
        N = U.shape[-1]
        assert N > 0 and U.shape[-1] == U.shape[-2], '[E] Input matrix must be square and N > 0'

        phi_mat = np.zeros(list(U.shape)+[4], dtype=np.float64)
        # delta_list = np.zeros(U.shape[:-1], dtype=np.float64)
        for i in range(N-1):
            ### each outer loop deals with one off-diagonal
            ## even loop for column rotation
            if(i % 2 == 0):
                for j in range(i + 1):
                    ### let p, q be the indices for the nullified '0'
                    p = N - 1 - j ## row
                    q = i - j ## col
                    ### rotate two columns such that u2 is nullified to 0
                    pi = np.pi
                    # half_pi = np.pi / 2
                    min_err = self.min_err
                    u1, u2 = U[..., p:p+1, q+1], U[..., p:p+1, q]
                    pi = np.pi
                    u1_abs, u2_abs = np.abs(u1), np.abs(u2)
                    min_err = self.min_err
                    cond1 = u1_abs < min_err
                    cond2 = u2_abs < min_err
                    cond1_n = ~cond1
                    cond2_n = ~cond2
                    phi = np.where(cond1 & cond2, 0,
                                np.where(cond1_n & cond2, np.where(u1 > min_err, 0, -pi),
                                            np.where(cond1 & cond2_n, np.where(u2 > min_err, -0.5*pi, 0.5*pi), np.arctan2(-u2, u1))))
                    phi = -phi ### simply convert the solved theta from T to T*, it is easier than changing the solving procedure
                    phi[phi < -pi/2] += 2*pi
                    # phi_mat[p,q] = phi
                    pairwise_index = i - j
                    # theta_checkerboard[pairwise_index, -j - 1] = phi
                    # phi_mat[pairwise_index, j] = phi
                    phi_mat[..., pairwise_index, j, 0] = np.pi/2 ## this absorbs the global phase theta_tot
                    phi_mat[..., pairwise_index, j, 1] = 3*np.pi/2
                    phi_mat[..., pairwise_index, j, 2] = 1.5*np.pi - phi[..., 0]
                    phi_mat[..., pairwise_index, j, 3] = 0.5*np.pi + phi[..., 0]

                    # theta_checkerboard[pairwise_index, -j - 1] = phi

                    c, s = np.cos(phi), np.sin(phi)
                    ## q_p1 means q+1; right multiply by T*
                    col_q_p1, col_q = U[..., :p+1, q+1], U[..., :p+1, q]
                    col_q_p1_cos, col_q_p1_sin = col_q_p1 * c, col_q_p1 * s
                    col_q_cos, col_q_sin = col_q * c, col_q * s
                    U[..., :p+1, q+1], U[..., :p+1, q] = col_q_p1_cos + col_q_sin, col_q_cos - col_q_p1_sin
            else:
                ## odd loop for row rotation
                for j in range(i+1):
                    p = N - 1 - i + j
                    q = j
                    ### rotate two rows such that u2 is nullified to 0
                    pi = np.pi
                    # half_pi = np.pi / 2
                    min_err = self.min_err
                    u1, u2 = U[..., p-1, q:q+1], U[..., p, q:q+1]
                    pi = np.pi
                    u1_abs, u2_abs = np.abs(u1), np.abs(u2)
                    min_err = self.min_err
                    cond1 = u1_abs < min_err
                    cond2 = u2_abs < min_err
                    cond1_n = ~cond1
                    cond2_n = ~cond2
                    phi = np.where(cond1 & cond2, 0,
                                np.where(cond1_n & cond2, np.where(u1 > min_err, 0, -pi),
                                            np.where(cond1 & cond2_n, np.where(u2 > min_err, -0.5*pi, 0.5*pi), np.arctan2(-u2, u1))))

                    phi = -phi ### simply convert the solved theta from T to T*, it is easier than changing the solving procedure
                    phi[phi < -pi/2] += 2*pi
                    # phi_mat[p,q] = phi
                    pairwise_index = N + j - i - 2
                    # theta_checkerboard[pairwise_index, -j - 1] = phi
                    # phi_mat[pairwise_index, j] = phi
                    phi_mat[..., pairwise_index, N - 1 - j, 0] = np.pi/2 ## this absorbs the global phase theta_tot
                    phi_mat[..., pairwise_index, N - 1 - j, 1] = 3*np.pi/2
                    phi_mat[..., pairwise_index, N - 1 - j, 2] = 1.5*np.pi - phi[..., 0]
                    phi_mat[..., pairwise_index, N - 1 - j, 3] = 0.5*np.pi + phi[..., 0]


                    # theta_checkerboard[pairwise_index, j] = phi
                    # phi_mat[..., pairwise_index, N - 1 - j] = -phi[..., 0] ### from T* to T, consistent with propogation through MZI (T) see clements paper Eq.(4)
                    c, s = np.cos(phi), np.sin(phi)
                    ## p_1 means p - 1; left multiply by T
                    row_p_1, row_p = U[..., p-1, j:], U[..., p, j:]
                    row_p_1_cos, row_p_1_sin = row_p_1 * c, row_p_1 * s
                    row_p_cos, row_p_sin = row_p * c, row_p * s
                    U[..., p-1, j:], U[..., p, j:] = row_p_1_cos + row_p_sin, row_p_cos - row_p_1_sin
            # print(U)
        # for i in range(N):
        #     delta_list[..., i] = U[..., i, i]## only the first and last element can be 1 or -1, the rest elements are all 1. This feature can be used in fast forward/reconstruction
        delta_list = np.angle(batch_diag(U))
        ### efficiently absorb delta_list into theta_t and theta_l and move delta_list to the last phase shifter column
        ### since U is real matrix, only delta_list[0] and delta_list[-1] can be -1.
        if(N % 2 == 1):
            phi_mat[..., 0, -1, 0] += delta_list[..., 0]
            delta_list[..., 0] = 0
            phi_mat[..., N-2, 1, 1] += delta_list[..., -1]
            delta_list[..., -1] = 0
        else:
            phi_mat[..., N-2, 2, 1] += delta_list[..., -1]
            delta_list[..., -1] = 0
        return delta_list, phi_mat

    def decompose_clements(self, U):
        if(isinstance(U, np.ndarray)):
            if(len(U.shape) == 2):
                return self.decompose_clements_cpu(U)
            else:
                return self.decompose_clements_batch(U)
        else:
            if(U.is_cuda):
                N = U.size(-1)
                size = U.size()
                U = U.view(-1, N, N).contiguous()
                delta_list = torch.zeros(list(U.size())[:-1], dtype=U.dtype, device=U.device).contiguous()
                phi_mat = torch.zeros_like(U).contiguous()
                matrix_parametrization_cuda.decompose_clements(U, delta_list, phi_mat)
                delta_list = delta_list.view(list(size)[:-1])
                phi_mat = phi_mat.view(size)
                return delta_list, phi_mat
            else:
                if(U.dim() == 2):
                    return torch.from_numpy(self.decompose_clements_cpu(U.cpu().numpy()))
                else:
                    return torch.from_numpy(self.decompose_clements_batch(U.cpu().numpy()))

    def checkerboard_to_param(self,checkerboard: np.ndarray, units: int):
        param = np.zeros((units, units // 2))
        if units % 2:
            param[::2, :] = checkerboard.T[::2, :-1:2]
        else:
            param[::2, :] = checkerboard.T[::2, ::2]
        param[1::2, :] = checkerboard.T[1::2, 1::2]
        return param

    def reconstruct_clements_cpu(self, delta_list, phi_mat):
        N = delta_list.shape[0]
        Ur = np.identity(N, dtype=np.complex64)

        # parallelly reconstruct col by col based on the checkerboard (phi_mat)
        # count = 0
        # phi_mat_cos = np.cos(phi_mat)
        # phi_mat_sin = np.sin(phi_mat)
        ### efficiently absorb delta_list into theta_t and theta_l
        # if(N % 2 == 1):
        #     ### can completely absorb delta_list
        #     phi_mat[N-2, 1, 1] += delta_list[-1]
        #     for i in range(1, N):
        #         phi_mat[N-i-1, i, 0] += delta_list[N-i-1]
        # else:
        #     ## cannot absorb delta_list[0] because we do not have right and bottom phase shifters
        #     ## delta_list[0] has to be addressed during reconstruction
        #     phi_mat[N-2, 2, 1] += delta_list[-1]
        #     for i in range(2, N):
        #         phi_mat[N-i, i, 0] += delta_list[N-i]


        for i in range(N): ## N layers
            max_len = 2 * (i + 1)
            for j in range((i%2), N-1, 2):
                # c, s = phi_mat_cos[j, i], phi_mat_sin[j, i]
                phi = phi_mat[j, i, :]
                ## not the entire row needs to be rotated, only a small working set is used
                # row_p, row_q = Ur[j, :], Ur[j+1, :]
                lower = j - i
                upper = lower + max_len
                lower = max(0, lower)
                upper = min(upper, N)
                row_p, row_q = Ur[j, lower:upper], Ur[j+1, lower:upper]
                row_p *= np.exp(1j*(phi[0]+(phi[2]+phi[3])/2+np.pi/2))
                row_q *= np.exp(1j*(phi[1]+(phi[2]+phi[3])/2+np.pi/2))
                half_delta_theta = (phi[2]-phi[3])/2
                c, s = np.cos(half_delta_theta), np.sin(half_delta_theta)
                row_p_cos, row_p_sin = row_p * c, row_p * s
                row_q_cos, row_q_sin = row_q * c, row_q * s
                Ur[j, lower:upper], Ur[j+1, lower:upper] = row_p_sin + row_q_cos, row_p_cos - row_q_sin
                # row_p_cos, row_p_sin = row_p * c, row_p * s
                # row_q_cos, row_q_sin = row_q * c, row_q * s
                # Ur[j, lower:upper], Ur[j+1, lower:upper] = row_p_cos - row_q_sin, row_p_sin + row_q_cos
                # res = (c+1j*s)*(row_p+1j*row_q)
                # Ur[j, lower:upper], Ur[j+1, lower:upper] = res.real, res.imag

        if(N % 2 == 0):
            ### have to address delta_list[0]
            Ur[0,:] *= np.exp(1j*delta_list[0])
        return Ur

    def reconstruct_clements_batch(self, delta_list, phi_mat):
        N = delta_list.shape[-1]
        Ur = batch_eye_cpu(N, batch_shape=delta_list.shape[:-1], dtype=np.complex64)
        # parallelly reconstruct col by col based on the checkerboard (phi_mat)
        # count = 0
        # phi_mat_cos = np.cos(phi_mat)
        # phi_mat_sin = np.sin(phi_mat)
        for i in range(N): ## N layers
            max_len = 2 * (i + 1)
            for j in range((i%2), N-1, 2):
                ## not the entire row needs to be rotated, only a small working set is used
                lower = j - i
                upper = lower + max_len
                lower = max(0, lower)
                upper = min(upper, N)
                # c, s = phi_mat_cos[..., j, i:i+1], phi_mat_sin[..., j, i:i+1]
                phi = phi_mat[..., j, i:i+1,:]
                # row_p, row_q = Ur[..., j, :], Ur[..., j+1, :]
                row_p, row_q = Ur[..., j, lower:upper], Ur[..., j+1, lower:upper]
                row_p *= np.exp(1j*(phi[..., 0]+(phi[..., 2]+phi[..., 3])/2+np.pi/2))
                row_q *= np.exp(1j*(phi[..., 1]+(phi[..., 2]+phi[..., 3])/2+np.pi/2))
                half_delta_theta = (phi[..., 2]-phi[..., 3])/2
                c, s = np.cos(half_delta_theta), np.sin(half_delta_theta)
                row_p_cos, row_p_sin = row_p * c, row_p * s
                row_q_cos, row_q_sin = row_q * c, row_q * s
                Ur[..., j, lower:upper], Ur[..., j+1, lower:upper] = row_p_sin + row_q_cos, row_p_cos - row_q_sin

        if(N % 2 == 0):
            ### have to address delta_list[0]
            Ur[..., 0,:] *= np.exp(1j*delta_list[..., 0:1])

        return Ur

    def reconstruct_clements(self, delta_list, phi_mat):
        if(isinstance(phi_mat, np.ndarray)):
            if(len(delta_list.shape) == 1):
                return self.reconstruct_clements_cpu(delta_list, phi_mat)
            else:
                return self.reconstruct_clements_batch(delta_list, phi_mat)
        else:
            if(phi_mat.is_cuda):
                size = phi_mat.size()
                N = phi_mat.size(-1)
                delta_list = delta_list.view(-1, N).to(phi_mat.device).contiguous()
                phi_mat = phi_mat.view(-1, N, N).contiguous()
                # U = torch.zeros_like(phi_mat).contiguous()
                U = matrix_parametrization_cuda.reconstruct_clements(delta_list, phi_mat)

                U = U.view(size)
                return U
            else:
                if(phi_mat.dim() == 2):
                    return torch.from_numpy(self.reconstruct_clements(delta_list.cpu().numpy(), phi_mat.cpu().numpy()))
                else:
                    return torch.from_numpy(self.reconstruct_clements_batch(delta_list.cpu().numpy(), phi_mat.cpu().numpy()))

    def checkIdentity(self, M):
        return (M.shape[0] == M.shape[1]) and np.allclose(M, np.eye(M.shape[0]))

    def checkUnitary(self, U):
        M = np.dot(U, U.T)
        # print(M)
        return self.checkIdentity(M)

    def checkEqual(self, M1, M2):
        return (M1.shape == M2.shape) and np.allclose(M1, M2)

    def genRandomOrtho(self, N):
        U = ortho_group.rvs(N)
        print(
            f'[I] Generate random {N}*{N} unitary matrix, check unitary: {self.checkUnitary(U)}')
        return U

    def toDegree(self, M):
        return np.degrees(M)

class ComplexUnitaryDecomposer(object):
    timer = True

    def __init__(self, min_err=1e-6, timer=False):
        self.min_err = min_err
        self.timer = timer

    def buildPlaneUnitary(self, p, q, phi, sigma, N, transpose=True):
        assert N > 0 and isinstance(
            N, int), "[E] Matrix size must be positive integer"
        assert isinstance(p, int) and isinstance(q,
                                                 int) and 0 <= p < q < N, "[E] Integer value p and q must satisfy p < q"
        assert (isinstance(phi, float) or isinstance(phi, int)) and (
            isinstance(sigma, float) or isinstance(sigma,
                                                   int)), "[E] Value phi and sigma must be of type float or int"

        U = np.eye(N, dtype=complex)
        c = np.cos(phi)
        s = np.sin(phi)

        U[p, p] = U[q, q] = c
        U[q, p] = s * np.exp(1j * sigma) if not transpose else - \
            s * np.exp(1j * sigma)
        U[p, q] = -s * \
            np.exp(-1j * sigma) if not transpose else s * np.exp(-1j * sigma)

        return U

    def calPhiSigma(self, u1, u2):
        u1_abs, u2_abs = np.abs(u1), np.abs(u2)
        u1_real, u2_img = np.real(u1), np.imag(u2)
        min_err = self.min_err

        if u1_abs < min_err and u2_abs < min_err:
            phi = 0
            sigma = 0
        elif u1_abs >= min_err and u2_abs < min_err:
            phi = 0 if u1_real > -min_err else -np.pi
            sigma = 0
        elif u1_abs < min_err and u2_abs >= min_err:
            phi = -0.5 * np.pi if u2_img < min_err else 0.5 * np.pi
            sigma = 0
        else:
            # solve the equation: u'_1n=0
            # first cal sigma, make its range -pi/2 <= sigma <= pi/2
            sigma = np.angle(u2.conj() / u1.conj())
            if (sigma > np.pi / 2):  # solution sigma need to be adjusted by +pi or -pi, then case 2 for phi
                sigma -= np.pi
                if (u1_real > -min_err):
                    phi = np.arctan(u2_abs / u1_abs)  # 1 quadrant
                else:
                    phi = np.arctan(u2_abs / u1_abs) - np.pi  # 3 quadrant
            # solution sigma need to be adjusted by +pi or -pi, then case 2 for phi
            elif (sigma < -np.pi / 2):
                sigma += np.pi
                if (u1_real > -min_err):
                    phi = np.arctan(u2_abs / u1_abs)  # 1 quadrant
                else:
                    phi = np.arctan(u2_abs / u1_abs) - np.pi  # 3 quadrant
            else:  # solution sigma satisfies its range, then case 1 for phi
                if (u1_real > -min_err):
                    phi = np.arctan(-u2_abs / u1_abs)  # 4 quadrant
                else:
                    phi = np.arctan(-u2_abs / u1_abs) + np.pi  # 2 quadrant

        return phi, sigma

    @profile(timer=timer)
    def decomposeKernel(self, U, dim):
        '''return U^(N-1); (phi_1,...,phi_N-2); (sigma_1,...,sigma_N-2)'''
        N = U.shape[0]
        phi_list = np.zeros(dim, dtype=np.float64)
        sigma_list = np.zeros(dim, dtype=np.float64)

        for i in range(N - 1):
            u1, u2 = U[0, 0], U[0, N - 1 - i]
            phi, sigma = self.calPhiSigma(u1, u2)
            phi_list[i], sigma_list[i] = phi, sigma
            p, q = 0, N - i - 1
            U = np.dot(U, self.buildPlaneUnitary(
                p=p, q=q, phi=phi, sigma=sigma, N=N, transpose=True))

        print(f'[I] Decomposition kernel done')
        return U, phi_list, sigma_list

    @profile(timer=timer)
    def decompose(self, U):
        N = U.shape[0]
        assert N > 0 and U.shape[0] == U.shape[1], '[E] Input matrix must be square and N > 0'

        phi_mat = np.zeros([N, N], dtype=np.float64)
        sigma_mat = np.zeros([N, N], dtype=np.float64)
        delta_list = np.zeros(N, dtype=complex)

        for i in range(N - 1):
            U, phi_list, sigma_list = self.decomposeKernel(U, dim=N)
            phi_mat[i, :], sigma_mat[i, :] = phi_list, sigma_list
            delta_list[i] = U[0, 0]
            U = U[1:, 1:]
        else:
            delta_list[-1] = U[-1, -1]

        print(f'[I] Decomposition done')
        return delta_list, phi_mat, sigma_mat

    @profile(timer=timer)
    def reconstruct(self, delta_list, phi_mat, sigma_mat):
        N = delta_list.shape[0]
        Ur = np.identity(N)

        # reconstruct from right to left as in the book chapter
        for i in range(N):
            for j in range(N - i - 1):
                phi, sigma = phi_mat[i, j], sigma_mat[i, j]
                Ur = np.dot(self.buildPlaneUnitary(
                    i, N - j - 1, phi, sigma, N, transpose=False), Ur)

        D = np.diag(delta_list)
        Ur = np.dot(D, Ur)
        print(f'[I] Reconstruct {N}*{N} unitary matrix done')

        return Ur

    def checkIdentity(self, M):
        return (M.shape[0] == M.shape[1]) and np.allclose(M, np.eye(M.shape[0]))

    def checkUnitary(self, U):
        M = np.dot(U, U.conj().T)
        return self.checkIdentity(M)

    def checkEqual(self, M1, M2):
        return (M1.shape[0] == M2.shape[0]) and (M1.shape[1] == M2.shape[1]) and np.allclose(M1, M2)

    def genRandomUnitary(self, N):
        U = unitary_group.rvs(N)
        print(
            f'[I] Generate random {N}*{N} unitary matrix, check unitary: {self.checkUnitary(U)}')
        return U

    def toDegree(self, M):
        return np.degrees(M)


def sparsifyDecomposition(phi_mat, sigma_mat=None, row_preserve=1):
    phi_mat[row_preserve:, ...] = 0
    if(sigma_mat):
        sigma_mat[row_preserve:, ...] = 0
    print(f'[I] Sparsify decomposition')


class fullprint:
    'context manager for printing full numpy arrays'

    def __init__(self, **kwargs):
        '''linewidth=75; precision=8'''
        kwargs.setdefault('threshold', np.inf)
        self.opt = kwargs

    def __enter__(self):
        self._opt = np.get_printoptions()
        np.set_printoptions(**self.opt)

    def __exit__(self, type, value, traceback):
        np.set_printoptions(**self._opt)


if __name__ == '__main__':
    ### test clements decomposition
    ud = RealUnitaryDecomposerBatch()
    N = 5
    U = ud.genRandomOrtho(N)
    U2 = U.copy()
    delta_list, phi_mat = ud.decompose_clements(U)
    print(U)
    print(delta_list)
    print(phi_mat)
    exit(0)

    phi = np.ones(512)
    with TimerCtx() as t:
        for _ in range(512*512):
            # a = np.sin(0.678)
            a = np.sin(phi)
    print(t.interval)
    # ac = np.cos(0.678)
    ac = np.cos(phi)
    pi2 = 2*np.pi
    with TimerCtx() as t:
        for _ in range(512*512):
            # a = (1 - ac*ac)**0.5 if 0.678 > 0 else -((1 - ac*ac)**0.5)
            b = (1 - ac*ac)**0.5
            phi = phi % (2*np.pi)
            b[phi > np.pi] *= -1
            # a = (1 - ac*ac)**0.5 * (0.678 % pi2)
    print(t.interval)
    assert np.allclose(a, b)
    exit(1)
    udb = RealUnitaryDecomposerBatch(dtype=np.float64)

    U = udb.genRandomOrtho(256)
    with TimerCtx() as t:
        delta_list, phi_mat = udb.decompose(U.copy())
    print(t.interval)
    # phi_mat[phi_mat > 1] = 0
    # print(phi_mat)
    U_recon = udb.reconstruct_2(delta_list, phi_mat).astype(np.float32)
    assert np.allclose(U, U_recon)
    print(np.mean(np.abs(U-U_recon)))
    delta_list, phi_mat = udb.decompose(U_recon)
    # print(phi_mat)
    exit(1)

    W = np.random.randn(32, 32, 8, 8)
    W2 = W.copy()
    with TimerCtx() as t:
        delta_list, phi_mat = udb.decompose_batch(W)
    print(t.interval)

    with TimerCtx() as t:
        for i in range(W2.shape[0]):
            for j in range(W2.shape[1]):
                delta, phi = udb.decompose(W2[i, j, ...])
                res = udb.checkEqual(delta_list[i, j, ...], delta)
                # if(not res):
                #     print(i,j, "delta")

                # res = udb.checkEqual(phi_mat[i,j,...], phi)
                # if(not res):
                #     # pass
                #     print(phi_mat[i,j,...], phi)
    print(t.interval)
    Ur = udb.reconstruct_2_batch(delta_list, phi_mat)
    with TimerCtx() as t:
        Ur = udb.reconstruct_2_batch(delta_list, phi_mat)
    print(t.interval)

    with TimerCtx() as t:
        for i in range(W2.shape[0]):
            for j in range(W2.shape[1]):
                Ur2 = udb.reconstruct_2(
                    delta_list[i, j, ...], phi_mat[i, j, ...])
                # res = udb.checkEqual(Ur[i,j,...], Ur2)
                # if(not res):
                #     print("error")
    print(t.interval)

    exit(1)

    # u1 = np.random.randn(64*64)
    # u2 = np.random.randn(64*64)
    # phi = ud.calPhi_batch(u1,u2)
    # # print(phi)
    # phi_list = []
    # with TimerCtx() as t:
    #     for i in range(64*64):
    #         phi_list.append(ud.calPhi(u1[i], u2[i]))
    # print(t.interval*1000, "ms")
    # phi_list = np.array(phi_list)
    # print(ud.checkEqual(phi, phi_list))

    # print(phi_list)
    exit(1)
    # np.set_printoptions(threshold=np.inf)
    # N = 5
    # ud = ComplexUniaryDecomposer(timer=True)
    # U = ud.genRandomUnitary(N)
    # # print(f'[I] Original:\n{U}')

    # UP, phi_list, sigma_list = ud.decomposeKernel(U, dim=N)
    # print(f'[I] U Prime:\n{UP}')
    # print(f'[I] check U Prime is unitary:{ud.checkUnitary(UP)}')
    # print(f'[I] phi_list:\n{phi_list}')
    # print(f'[I] sigma_list:\n{sigma_list}')

    # delta_list, phi_mat, sigma_mat = ud.decompose(U)
    # print(f'[I] delta_list:\n{delta_list}')
    # print(f'[I] phi_mat:\n{ud.toDegree(phi_mat)}')
    # print(f'[I] sigma_mat:\n{sigma_mat}')

    # sparsifyDecomposition(phi_mat, sigma_mat, row_preserve=1)
    # Ur = ud.reconstruct(delta_list, phi_mat, sigma_mat)
    # #print(f'[I] Reconstructed:\n{Ur}')
    # with fullprint(threshold=None, linewidth=150, precision=2):
    #     print(Ur)
    # print(f'[I] Check reconstruction is unitary: {ud.checkUnitary(Ur)}')
    # # print(f'[I] Check reconstruction is equal to original: {ud.checkEqual(Ur, U)}')

    # real matrix parameterization
    N = 64
    ud = RealUnitaryDecomposer(timer=True)
    U = ud.genRandomOrtho(N)
    U_cuda = torch.from_numpy(U).cuda().float()
    # U = np.eye(4)
    print(U)

    delta_list, phi_mat = ud.decompose(U)

    print(f'[I] delta_list:\n{delta_list}')
    print(f'[I] phi_mat:\n{phi_mat}')

    ud_2 = RealUnitaryDecomposerPyTorch(
        timer=True, use_multithread=False, n_thread=24)
    # U = ud.genRandomOrtho(N)
    # print(f'[I] Original:\n{U}')

    # UP, phi_list, sigma_list = ud.decomposeKernel(U, dim=N)
    # print(f'[I] U Prime:\n{UP}')
    # print(f'[I] check U Prime is unitary:{ud.checkUnitary(UP)}')
    # print(f'[I] phi_list:\n{phi_list}')
    # print(f'[I] sigma_list:\n{sigma_list}')

    delta_list_2, phi_mat_2 = ud_2.decompose(U_cuda)
    print(f'[I] delta_list CUDA:\n{delta_list_2}')
    print(f'[I] phi_mat CUDA:\n{phi_mat_2}')

    # phi_list_2 = torch.zeros(N*(N-1)//2).cuda()
    # phi_list = np.zeros([N*(N-1)//2])
    # for i in range(N):
    #     for j in range(N-i-1):
    #         phi_list_2[int((2 * N - i - 1) * i / 2 + j)] = phi_mat_2[i, j]
    #         phi_list[int((2 * N - i - 1) * i / 2 + j)] = phi_mat[i, j]
    # print("phi_list:", phi_list_2)
    # print("phi_list:", phi_list)

    # sparsifyDecomposition(phi_mat, row_preserve=1)
    Ur = ud.reconstruct(delta_list, phi_mat)
    print(f'[I] Reconstructed:\n{Ur}')
    print(f'[I] Check reconstruction is unitary: {ud.checkUnitary(Ur)}')
    Ur = ud.reconstruct_2(delta_list, phi_mat)
    print(f'[I] Reconstructed:\n{Ur}')
    print(f'[I] Check reconstruction is unitary: {ud.checkUnitary(Ur)}')

    Ur_2 = ud_2.reconstruct(delta_list_2, phi_mat_2)
    print(f'[I] Check reconstruction is unitary: {ud_2.checkUnitary(Ur_2)}')
    Ur_2 = ud_2.reconstruct_2(delta_list_2, phi_list_2)
    print(f'[I] Check reconstruction is unitary: {ud_2.checkUnitary(Ur_2)}')

    # with fullprint(threshold=None, linewidth=150, precision=4):
    #     print(Ur)
    print(Ur_2)
    print(f'[I] Check reconstruction is unitary: {ud.checkUnitary(Ur)}')
    # print(f'[I] Check reconstruction is equal to original: {ud.checkEqual(Ur, U)}')
