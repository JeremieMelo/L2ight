
import torch
from torch import nn
from torch.functional import Tensor

__all__ = ["Swish"]


@torch.jit.script
def swish_fwd(x: Tensor) -> Tensor:
    # return x.mul(torch.sigmoid(x))
    return torch.sigmoid(x).mul_(x)


@torch.jit.script
def swish_bwd(x: Tensor, grad_output: Tensor) -> Tensor:
    x_sigmoid = torch.sigmoid(x)
    # return grad_output * (x_sigmoid * (1. + x * (1. - x_sigmoid)))
    output = (1-x_sigmoid).mul_(x).add_(1).mul_(x_sigmoid)
    del x_sigmoid
    return output.mul_(grad_output)


class SwishJitImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return swish_fwd(x)

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_tensors[0]
        return swish_bwd(x, grad_output)


class Swish(nn.Module):
    def __init__(self, inplace: bool = True, memory_efficient: bool = True) -> None:
        super(Swish, self).__init__()
        self.inplace = inplace
        self.swish = self.memory_efficient_swish if memory_efficient else self.original_swish

    def original_swish(self, x, inplace: bool = False) -> Tensor:
        return x.mul_(x.sigmoid()) if inplace else x.mul(x.sigmoid())

    def memory_efficient_swish(self, x, inplace: bool = False) -> Tensor:
        return SwishJitImplementation.apply(x)

    def forward(self, x):
        return self.swish(x, self.inplace)

