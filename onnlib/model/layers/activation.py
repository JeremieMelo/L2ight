

from torch import nn
import torch
from torch.nn import init

__all__ = ["ReLUN", "ModReLU"]

class ReLUN(nn.Hardtanh):
    r"""Applies the element-wise function:

    .. math::
        \text{ReLUN}(x) = \min(\max(0,x), N)

    Args:
        inplace: can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(N, *)`, same shape as the input

    .. image:: scripts/activation_images/ReLU6.png

    Examples::

        >>> m = nn.ReLUN(N)
        >>> input = torch.randn(2)
        >>> output = m(input)
    """

    def __init__(self, N, inplace=False):
        super(ReLUN, self).__init__(0., N, inplace)

    def extra_repr(self):
        inplace_str = 'inplace' if self.inplace else ''
        return inplace_str



class ModReLU(nn.Module):
    """ A modular ReLU activation function for complex-valued tensors """

    def __init__(self, bias_shape, device=torch.cuda):
        super(ModReLU, self).__init__()
        self.device = device
        if(isinstance(bias_shape, int)):
            self.bias = nn.Parameter(
                torch.Tensor(1, bias_shape).to(self.device))
        else:
            self.bias = nn.Parameter(torch.Tensor(
                1, *bias_shape).to(self.device))
        self.relu = nn.ReLU()
        self.init_bias()

    def init_bias(self):
        init.constant(self.bias, val=0)

    def forward(self, x, eps=1e-5):
        """ ModReLU forward
        Args:
            x (torch.tensor): A torch float tensor with the real and imaginary part
                stored in the last dimension of the tensor; i.e. x.shape = [a, ...,b, 2]
        Kwargs:
            eps (float): A small number added to the norm of the complex tensor for
                numerical stability.
        """
        x_re, x_im = x[..., 0], x[..., 1]
        norm = torch.sqrt(x_re ** 2 + x_im ** 2) + 1e-5
        phase_re, phase_im = x_re / norm, x_im / norm
        activated_norm = self.relu(norm + self.bias)
        modrelu = torch.stack(
            [activated_norm * phase_re, activated_norm * phase_im], -1
        )
        return modrelu
