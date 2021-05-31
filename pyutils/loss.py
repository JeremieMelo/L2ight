
##########################
#          loss          #
##########################
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.functional import Tensor

__all__ = ["distillation_loss", "CrossEntropyLossSmooth", "f_divergence", "AdaptiveLossSoft"]


def distillation_loss(y: Tensor, teacher_scores: Tensor, labels: Tensor, T: float = 6.0, alpha: float = 0.9) -> Tensor:
    '''
    description: Knowledge distillation loss function combines hard target loss and soft target loss, [https://github.com/szagoruyko/attention-transfer/blob/master/utils.py#L10]
    y {tensor.Tensor} Model output logits from the student model
    teacher_scores {tensor.Tensor} Model output logits from the teacher model
    labels {tensor.LongTensor} Hard labels, Ground truth
    T {scalar, Optional} temperature of the softmax function to make the teacher score softer. Default is 6 when accuracy is high, e.g., >95%. Typical value 1~20 [https://zhuanlan.zhihu.com/p/83456418]
    alpha {scalar, Optional} interpolation between hard and soft target loss. Default set to 0.9 for distillation mode. When hard target loss is very small (alpha=0.9), gets the best results. [https://zhuanlan.zhihu.com/p/102038521]
    return loss {tensor.Tensor} loss function
    '''
    alpha = np.clip(alpha, 0, 1)
    if(alpha == 0):
        l_ce = F.cross_entropy(y, labels)
        return l_ce
    elif(0 < alpha < 1):
        p = F.log_softmax(y/T, dim=1)
        q = F.softmax(teacher_scores/T, dim=1)
        l_kl = F.kl_div(p, q, reduction="sum") * (T**2) / y.shape[0]
        l_ce = F.cross_entropy(y, labels)
        return l_kl * alpha + l_ce * (1. - alpha)
    else:
        p = F.log_softmax(y/T, dim=1)
        q = F.softmax(teacher_scores/T, dim=1)
        l_kl = F.kl_div(p, q, reduction="sum") * (T**2) / y.shape[0]
        return l_kl


class CrossEntropyLossSmooth(torch.nn.modules.loss._Loss):
    def __init__(self, label_smoothing: float = 0.1):
        super().__init__()
        self.eps = label_smoothing

    """ label smooth """

    def forward(self, output: Tensor, target: Tensor) -> Tensor:
        n_class = output.size(1)
        one_hot = torch.zeros_like(output).scatter(1, target.view(-1, 1), 1)
        target = one_hot.mul_(1 - self.eps).add_(self.eps / n_class)
        output_log_prob = F.log_softmax(output, dim=1)
        target.unsqueeze_(1)
        output_log_prob.unsqueeze_(2)
        loss = -torch.bmm(target, output_log_prob)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


def f_divergence(q_logits: Tensor, p_logits: Tensor, alpha: float, iw_clip: float = 1e3) -> Tuple[Tensor, Tensor]:
    '''https://github.com/facebookresearch/AlphaNet/blob/master/loss_ops.py
    title={AlphaNet: Improved Training of Supernet with Alpha-Divergence},
    author={Wang, Dilin and Gong, Chengyue and Li, Meng and Liu, Qiang and Chandra, Vikas},
    '''
    assert isinstance(alpha, float)
    q_prob = F.softmax(q_logits.data, dim=1)
    p_prob = F.softmax(p_logits.data, dim=1)
    # gradient is only backpropagated here
    q_log_prob = F.log_softmax(q_logits, dim=1)

    importance_ratio = p_prob / q_prob
    if abs(alpha) < 1e-3:
        importance_ratio.clamp_(0, iw_clip).log_()
        f = -importance_ratio
        f_base = 0
        rho_f = importance_ratio.sub_(1)
    elif abs(alpha - 1.0) < 1e-3:
        # f = importance_ratio * importance_ratio.log()
        f = importance_ratio.log().mul_(importance_ratio)
        f_base = 0
        rho_f = importance_ratio
    else:
        iw_alpha = torch.pow(importance_ratio, alpha).clamp_(0, iw_clip)
        f_base = 1.0 / alpha / (alpha - 1.0)
        # f = iw_alpha / alpha / (alpha - 1.0)
        f = iw_alpha.mul(f_base)
        # f_base = 1.0 / alpha / (alpha - 1.0)
        rho_f = iw_alpha.div(alpha).add_(f_base)

    # loss = torch.sum(q_prob * (f - f_base), dim=1)
    loss = f.sub_(f_base).mul_(q_prob).sum(dim=1)
    grad_loss = -q_prob.mul_(rho_f).mul(q_log_prob).sum(dim=1)
    return loss, grad_loss


class AdaptiveLossSoft(torch.nn.modules.loss._Loss):
    '''https://github.com/facebookresearch/AlphaNet/blob/master/loss_ops.py
    title={AlphaNet: Improved Training of Supernet with Alpha-Divergence},
    author={Wang, Dilin and Gong, Chengyue and Li, Meng and Liu, Qiang and Chandra, Vikas},
    '''

    def __init__(self, alpha_min: float, alpha_max: float, iw_clip: float = 1e3) -> None:
        super().__init__()
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.iw_clip = iw_clip

    def forward(self, output: Tensor, target: Tensor, alpha_min: Optional[float] = None, alpha_max: Optional[float] = None) -> Tensor:
        alpha_min = alpha_min or self.alpha_min
        alpha_max = alpha_max or self.alpha_max

        loss_left, grad_loss_left = f_divergence(
            output, target, alpha_min, iw_clip=self.iw_clip)
        loss_right, grad_loss_right = f_divergence(
            output, target, alpha_max, iw_clip=self.iw_clip)

        ind = torch.gt(loss_left, loss_right).float()
        loss = ind * grad_loss_left + (1.0 - ind) * grad_loss_right

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss
