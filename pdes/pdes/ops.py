import torch

import numpy as np

from torchdiffeq import odeint_adjoint as odeint

from torchfilter.util import safe_cast

# class LaplacianFilter(torch.nn.Conv2d):
#     def __init__(self, dx = 1, padding_mode='circular', device='cpu'):
#         super(LaplacianFilter, self).__init__(
#             in_channels=1, out_channels=1,
#             kernel_size=3, stride=1,
#             padding_mode = padding_mode,
#             bias=False, padding=2
#         )
#         #self.weight.detach()
#         #self.weight.requires_grad=False
#         self.weight[0,0,:,:] = torch.tensor(
#             [[0., 1., 0.,], [1., -4., 1.], [0., 1., 0.]]
#         )/(dx**2)
#         # self.weight.requires_grad = False
#         self.to(safe_cast(torch.device, device))

class LaplacianFilter(torch.nn.Module):
    def __init__(self, dx=1, padding_mode='circular', device='cpu'):
        super(LaplacianFilter, self).__init__()
        self.kernel = torch.tensor(
            [[0., 1., 0.,], [1., -4., 1.], [0., 1., 0.]],
            device=safe_cast(torch.device, device)
        ).unsqueeze(0).unsqueeze(0)/(dx**2)
    def forward(self, x):
        return torch.nn.functional.conv2d(x, self.kernel, bias=None, stride=1, padding=1)


class GradientFilter(torch.nn.Conv2d):
    def __init__(self, dx = 1, padding_mode='circular', device='cpu'):
        super(GradientFilter, self).__init__(
            in_channels=1, out_channels=2,
            kernel_size=3, stride=1,
            padding_mode = padding_mode,
            bias=False, padding=2
        )
        self.weight.detach()
        self.weight[0,0,:,:] = torch.tensor(
            [[0., -1., 0.,], [0., 0., 0.], [0., 1., 0.]]
        )/(dx)
        self.weight[1,0,:,:] = torch.tensor(
            [[0., 0., 0.,], [-1., 0., 1.], [0., 0., 0.]]
        )/(dx)
        self.to(safe_cast(torch.device, device))

def dot(a, b):
    ''' Returns the inner product of two filtered tensors '''
    return (a*b).sum(axis=1).view(b.shape[0], *b.shape[2:])

class DivergenceFilter(torch.nn.Module):
    def __init__(self, dx = 1, padding_mode='circular', device='cpu'):
        super(DivergenceFilter, self).__init__()
        self._grad = GradientFilter(dx, padding_mode, device)

    def forward(self, x):
        return self._grad(x).sum(axis=1).view(*x.shape)
