import logging
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
import torch
import numpy as np

from pyutils import *

sys.path.pop(0)

__all__ = ["multiport_mmi", "multiport_mmi_with_ps"]

@torch.jit.script
def multiport_mmi(n_port: int=2, device: torch.device=torch.device("cuda")):
    """ N by N mmi. Operation Principles for Optical Switches Based on Two Multimode Interference Couplers, JLT 2012

    Args:
        n_port (int): Number of input/output ports. Defaults to 2.
    """
    # assert n_port >= 2, print("[E] n_port must be at least 2.")
    x = torch.arange(1, n_port+1, device=device).double()
    x, y = torch.meshgrid(x, x)
    sign = ((-1)**(x+y)).double()
    # x = torch.exp(1j*((x - 0.5) - sign*(y-0.5)).square().mul(np.pi/(-4*n_port))).mul(sign.mul(np.sqrt(1/n_port)))
    angle = x.sub(0.5).sub_(y.sub(0.5).mul_(sign)).square_().mul_(np.pi/(-4*n_port))
    x = torch.complex(angle.cos(), angle.sin()).mul_(sign.mul_((1/n_port)**0.5)).to(torch.complex64)
    return x
    return torch.view_as_real(x)

@torch.jit.script
def multiport_mmi_with_ps(n_port: int=2, ps_loc:str="before", device: torch.device=torch.device("cuda")):
    """ N by N mmi with N ps at the input ports. Operation Principles for Optical Switches Based on Two Multimode Interference Couplers, JLT 2012

    Args:
        n_port (int): Number of input/output ports. Defaults to 2.
    """
    # assert n_port >= 2, print("[E] n_port must be at least 2.")
    x = torch.arange(1, n_port+1, device=device).double()
    x, y = torch.meshgrid(x, x)
    sign = ((-1)**(x+y)).double()
    # x = torch.exp(1j*((x - 0.5) - sign*(y-0.5)).square().mul(np.pi/(-4*n_port))).mul(sign.mul(np.sqrt(1/n_port)))
    angle = x.sub(0.5).sub_(y.sub(0.5).mul_(sign)).square_().mul_(np.pi/(-4*n_port))
    x = torch.complex(angle.cos(), angle.sin()).mul_(sign.mul_((1/n_port)**0.5))
    angle = torch.zeros(n_port, device=device).double()
    angle[1::2] += 1
    angle *= np.pi/2
    ps = torch.complex(angle.cos(), angle.sin())

    if(ps_loc == "before"):
        x = x.mul(ps.unsqueeze(0)).to(torch.complex64)
    elif(ps_loc == "after"):
        x = ps.unsqueeze(1).mul(x).to(torch.complex64)
    return x
    return torch.view_as_real(x)

if __name__ == "__main__":
    c = 100
    N = 64
    # for _ in range(10):
    #     multiport_mmi(64)
    # torch.cuda.synchronize()
    # with TimerCtx() as t:
    #     for _ in range(c):
    #         multiport_mmi(N)
    #     torch.cuda.synchronize()
    # print(t.interval / c)
    print(multiport_mmi_with_ps(4, "after"))
    print(multiport_mmi_with_ps(4, "before"))
