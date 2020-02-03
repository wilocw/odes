import torch

import numpy as np

from torchdiffeq import odeint_adjoint as odeint

from abc import abstractmethod

from ..abstract import _AbstractPDE
from ..ops import LaplacianFilter, GradientFilter, DivergenceFilter, dot

class _AbstractConvectionDiffusionReaction(_AbstractPDE):
    def __init__(self, params, dx=5e2, padding_mode='circular', device='cpu'):
        super(_AbstractConvectionDiffusionReaction, self).__init__()
        self._params = params
        self._lapl   = LaplacianFilter(dx, padding_mode, device)
        self._grad   = GradientFilter(dx, padding_mode, device)
        self._div    = DivergenceFilter(dx, padding_mode, device)

    @abstractmethod
    def reaction(self, t, u):
        NotImplemented

    @abstractmethod
    def convection(self, t, u):
        NotImplemented

    @abstractmethod
    def diffusion(self, t, u):
        NotImplemented

    def forward(self, t, u):
        return -self.convection(t,u)+self.diffusion(t,u)+self.reaction(t,u)

class _AbstractConvectionDiffusion(_AbstractConvectionDiffusionReaction):
    def reaction(self, t, u):
        return torch.zeros_like(u)

class ConvectionDiffusion(_AbstractConvectionDiffusion):
    def __init__(self, *args, flow_type='incompressible', **kwargs):
        super(ConvectionDiffusion, self).__init__(*args, **kwargs)
        self._flow_type = flow_type

    def convection(self, t, u):
        if self._flow_type is 'incompressible':
            return dot(self._params['v'], self._grad(u))
        else:
            return self._div(self._params['v']*u)

    def diffusion(self, t, u):
        return self._params['D']*self._lapl(u)
