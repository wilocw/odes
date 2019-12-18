import torch

from torch import nn as nn
from torch import optim as optim
from torch.nn import functional as F

from torchdiffeq import odeint_adjoint as odeint

import numpy as np

from .util import safe_cast, StateSpaceDynamics
from .filters import KalmanFilter

from abc import ABC as _Abstract
from abc import abstractmethod

class _Smoother(nn.Module, _Abstract):
    def __init__(self, device='cpu'):
        super(_Smoother, self).__init__()
        self.device = safe_cast(torch.device, device)
        self.to(self.device)

class RauchTungStriebelSmoother(_Smoother):

    def __init__(self, transition, process_noise = None, device='cpu'):
        ''' doctstring '''
        super(RauchTungStriebelSmoother, self).__init__(device)
        self._transition = safe_cast(torch.Tensor, transition).to(self.device)
        if process_noise is None:
            self._process_noise_cov = torch.eye(
                self._transition.shape[-1],
                device=self.device
            )
        else:
            self._process_noise_cov = safe_cast(torch.Tensor, process_noise).to(self.device)

    @classmethod
    def from_filter(cls, filter):
        return cls(filter.F,filter.Q,device=filter.device)

    @property
    def F(self):
        return self._transition
    @property
    def Q(self):
        return self._process_noise_cov

    def forward(self, xf, Pf, return_gain=False):
        xf, Pf = torch.flip(xf, [-3]), torch.flip(Pf, [-3])
        D = xf.shape[-2]

        Gs = []

        def joint_dynamics(t, state):
            i  = t.type(torch.int)
            xi, Pi = xf[...,i,:,:],  Pf[...,i,:,:]

            xn = state[...,0:D,:].view(-1,D,1)
            Pn = state[...,D:,:].view(-1,D,D)

            xp, Pp = self.F @ xi, self.F @ Pi @ self.F.transpose(-1,-2) + self.Q
            G      = Pi @ self.F.transpose(-1,-2) @ torch.inverse(Pp)

            if return_gain:
                Gs.insert(0, G)
            xs, Ps = xi + G@(xn-xp), Pi + G@(Pn-Pp)@G.transpose(-1,-2)
            return torch.cat([xs.view(-1,D,1), Ps.view(-1,D*D,1)],-2) - state

        joint_smoother = StateSpaceDynamics(joint_dynamics, device=self.device)

        t = torch.arange(1., xf.shape[-3], dtype=torch.get_default_dtype())

        xN, PN = xf[...,0,:,:],  Pf[...,0,:,:]
        state_N = torch.cat([xN.view(-1,D,1), PN.view(-1,D*D,1)],-2)

        states = odeint(joint_smoother, state_N, t, method='euler')

        xs = states[...,:,0:D,:].view(*xf.shape[:-4], xf.shape[-3]-1, D, 1)
        Ps = states[...,:,D: ,:].view(*xf.shape[:-4], xf.shape[-3]-1, D, D)

        xs = torch.cat([xN.view(*xN.shape[0:-2],1,*xN.shape[-2:]), xs], -3)
        Ps = torch.cat([PN.view(*xN.shape[0:-2],1,*PN.shape[-2:]), Ps], -3)

        xs, Ps  = torch.flip(xs, (-3,)), torch.flip(Ps, (-3,))

        return (xs, Ps, Gs) if return_gain else (xs, Ps)

    def smooth(self, x, P, return_gain=True):
        return self.forward(x,P, return_gain)
