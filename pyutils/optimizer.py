#########################
#       Optimizer       #
#########################
import torch
from torch.optim.optimizer import Optimizer, required
import math
__all__ = ["Adam_GC", "SGD_GC", "RAdam", "GLD", "SMTP"]

class Adam_GC(Optimizer):
    r"""Implements Adam algorithm.
    It has been proposed in `Adam: A Method for Stochastic Optimization`_.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False)
    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super(Adam_GC, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(Adam_GC, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                if group['weight_decay'] != 0:
                    grad.add_(p.data, alpha=group['weight_decay'])

                #GC operation for Conv layers and FC layers
                if grad.dim()>1:
                   grad.add_(-grad.mean(dim = tuple(range(1,grad.dim())), keepdim = True))

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                else:
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])

                step_size = group['lr'] / bias_correction1

                p.data.addcdiv_(exp_avg, denom, value=-step_size)

        return loss


class SGD_GC(Optimizer):

    def __init__(self, params, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(SGD_GC, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SGD_GC, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data

                if weight_decay != 0:
                    d_p.add_(p.data, alpha=weight_decay)

                #GC operation for Conv layers and FC layers
                if d_p.dim()>1:
                   d_p.add_(-d_p.mean(dim = tuple(range(1,d_p.dim())), keepdim = True))

                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                    if nesterov:
                        d_p = d_p.add(buf, alpha=momentum)
                    else:
                        d_p = buf

                p.data.add_(d_p, alpha=-group['lr'])

        return loss


class RAdam(Optimizer):
    r"""
    https://github.com/LiyuanLucasLiu/RAdam/blob/master/radam/radam.py
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, degenerated_to_sgd=True):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))

        self.degenerated_to_sgd = degenerated_to_sgd
        if isinstance(params, (list, tuple)) and len(params) > 0 and isinstance(params[0], dict):
            for param in params:
                if 'betas' in param and (param['betas'][0] != betas[0] or param['betas'][1] != betas[1]):
                    param['buffer'] = [[None, None, None] for _ in range(10)]
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, buffer=[[None, None, None] for _ in range(10)])
        super(RAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(RAdam, self).__setstate__(state)

    def step(self, closure=None):

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError('RAdam does not support sparse gradients')

                p_data_fp32 = p.data.float()

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                state['step'] += 1
                buffered = group['buffer'][int(state['step'] % 10)]
                if state['step'] == buffered[0]:
                    N_sma, step_size = buffered[1], buffered[2]
                else:
                    buffered[0] = state['step']
                    beta2_t = beta2 ** state['step']
                    N_sma_max = 2 / (1 - beta2) - 1
                    N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)
                    buffered[1] = N_sma

                    # more conservative since it's an approximated value
                    if N_sma >= 5:
                        step_size = math.sqrt((1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max / (N_sma_max - 2)) / (1 - beta1 ** state['step'])
                    elif self.degenerated_to_sgd:
                        step_size = 1.0 / (1 - beta1 ** state['step'])
                    else:
                        step_size = -1
                    buffered[2] = step_size

                # more conservative since it's an approximated value
                if N_sma >= 5:
                    if group['weight_decay'] != 0:
                        p_data_fp32.add_(p_data_fp32, alpha=-group['weight_decay'] * group['lr'])
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    p_data_fp32.addcdiv_(exp_avg, denom, value=-step_size * group['lr'])
                    p.data.copy_(p_data_fp32)
                elif step_size > 0:
                    if group['weight_decay'] != 0:
                        p_data_fp32.add_(p_data_fp32, alpha=-group['weight_decay'] * group['lr'])
                    p_data_fp32.add_(exp_avg, alpha=-step_size * group['lr'])
                    p.data.copy_(p_data_fp32)

        return loss


class GLD(Optimizer):
    r"""Implements GLD-search and GLD-fast algorithm.
    It has been proposed in `Adam: A Method for Stochastic Optimization`_.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False)
    .. _GLD\: Gradientless Descent: High-Dimensional Zeroth-Order Optimization ICLR 2020
        https://arxiv.org/abs/1911.06317

    """

    def __init__(self, params, lr=1e-3, max_r=8, min_r=1, obj_fn=required,
                 weight_decay=0, max_cond=2, mode="search"):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0 <= min_r <= max_r:
            raise ValueError("Invalid searching radius: ({},{})".format(min_r,max_r))
        self.mode = mode
        if mode not in {"search", "fast"}:
            raise ValueError("Invalid mode: {}. Only support 'search' and 'fast'".format(mode))
        if(mode == "fast" and max_cond <= 0):
            raise ValueError("Invalid condition number bound: {}.".format(max_cond))
        self.obj_fn = obj_fn
        self.search = {"search": self.GLD_search,
                       "fast": self.GLD_fast_search}[mode]

        K = (int(math.log2(max_r/min_r)) + 1) if mode == "search" else (int(math.log2(max_cond)) + 1)
        defaults = dict(lr=lr, max_r=max_r, min_r=min_r,
                        weight_decay=weight_decay, max_cond=max_cond, mode=mode, K=K)
        super(GLD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(GLD, self).__setstate__(state)
        # for group in self.param_groups:
        #     group.setdefault('amsgrad', False)

    def GLD_search(self, obj_fn, K, R, p):
        obj_start = obj_min = obj_fn(p).item()
        v_min = 0
        for k in range(K):
            r_k = 2**(-k) * R
            v_k = torch.randn_like(p)
            v_k /= v_k.norm(p=2) / r_k
            p1 = p + v_k
            obj_k = obj_fn(p1).item()
            if(obj_k <= obj_min):
                obj_min = obj_k
                v_min = v_k.clone()
                # p_min = p1.clone()
        if(obj_min <= obj_start):
            p.data.add_(v_min)
        return obj_min

    def GLD_fast_search(self, obj_fn, K, R, p):
        obj_start = obj_min = obj_fn(p).item()
        v_min = 0
        for k in range(-K, K+1):
            r_k = 2**(-k) * R
            v_k = torch.randn_like(p)
            v_k /= v_k.norm(p=2) / r_k
            p1 = p + v_k
            obj_k = obj_fn(p1).item()
            if(obj_k <= obj_min):
                obj_min = obj_k
                v_min = v_k.clone()
                # p_min = p1.clone()
        if(obj_min <= obj_start):
            p.data.add_(v_min)
        return obj_min

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0

                state['step'] += 1
                K, R = group["k"], group['max_r']
                if(self.mode == "fast"):
                    Q = group["max_cond"]
                    H = int(p.numel() * Q * math.log2(Q))
                    R /= 2**(state["step"] // H)
                loss = self.search(self.obj_fn, K, R, p)

                if group['weight_decay'] != 0:
                    grad.add_(p.data, alpha=group['weight_decay'])

        return loss


class SMTP(Optimizer):
    r"""Implements SMTP algorithm.
    It has been proposed in `Stochastic Momentum Three Points`_.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        momentum (float, optional): momentum (default: 0)
        obj_fn (callable, required): objective function
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
    .. SMTP: A STOCHASTIC DERIVATIVE FREE OPTIMIZATION METHOD WITH MOMENTUM ICLR 2020
        https://arxiv.org/pdf/1905.13278.pdf

    """

    def __init__(self, params, lr=1e-3, momentum=0, obj_fn=required,
                 weight_decay=0):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0 <= momentum < 1:
            raise ValueError("Invalid momentum: {}".format(momentum))
        self.obj_fn = obj_fn

        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay)
        super(SMTP, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SMTP, self).__setstate__(state)
        # for group in self.param_groups:
        #     group.setdefault('amsgrad', False)


    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['v_k'] = 0
                    state['z_k'] = p.data.clone()

                state['step'] += 1
                lr = state['lr']
                beta = group["momentum"]
                z_k = state['z_k']
                v_k = state['v_k']

                s_k = torch.randn_like(p)
                s_k /= s_k.norm(p=2)
                v_k_plus_1_p = beta * v_k + s_k
                v_k_plus_1_n = beta * v_k - s_k

                x_k_plus_1_p = p - lr * v_k_plus_1_p
                x_k_plus_1_n = p - lr * v_k_plus_1_n

                z_k_plus_1_p = x_k_plus_1_p - lr * beta/(1 - beta) * x_k_plus_1_p
                z_k_plus_1_n = x_k_plus_1_n - lr * beta/(1 - beta) * v_k_plus_1_n

                sorted_obj = sorted([(p.data,       v_k,          z_k,          self.obj_fn(z_k)),
                                     (x_k_plus_1_p, v_k_plus_1_p, z_k_plus_1_p, self.obj_fn(z_k_plus_1_p)),
                                     (x_k_plus_1_n, v_k_plus_1_n, z_k_plus_1_n, self.obj_fn(z_k_plus_1_n))], key=lambda x: x[3])

                p.data.copy_(sorted_obj[0][0])
                state['v_k'] = sorted_obj[0][1]
                state['z_k'] = sorted_obj[0][2]
                loss = sorted_obj[0][3]

                if group['weight_decay'] != 0:
                    p.data.add_(-lr*group['weight_decay']*p.data)
                    # grad.add_(group['weight_decay'], p.data)

        return loss



