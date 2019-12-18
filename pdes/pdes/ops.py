import torch

import numpy as np

from torchdiffeq import odeint_adjoint as odeint

from torchfilter.util import safe_cast

class LaplacianFilter(torch.nn.Conv2d):
    def __init__(self, dx = 1, padding_mode='circular', device='cpu'):
        super(LaplacianFilter, self).__init__(
            in_channels=1, out_channels=1,
            kernel_size=3, stride=1,
            padding_mode = padding_mode,
            bias=False, padding=2
        )
        self.weight.detach()
        self.weight[0,0,:,:] = torch.tensor(
            [[0., 1., 0.,], [1., -4., 1.], [0., 1., 0.]]
        )/(dx**2)
        self.to(safe_cast(torch.device, device))
