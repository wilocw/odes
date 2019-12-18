import torch

from torch import nn as nn
from torch import optim as optim
from torch.nn import functional as F

from torchdiffeq import odeint_adjoint as odeint

import numpy as np

from .util import safe_cast, TimeInvariantDynamics

from abc import ABC as _Abstract
from abc import abstractmethod

class _Filter(nn.Module, _Abstract):
    def __init__(self, device='cpu'):
        super(_Filter, self).__init__()
        self.device = safe_cast(torch.device, device)
        self.to(self.device)

class KalmanFilter(_Filter):
    def __init__(self,
        transition, emission=None,
        process_noise=None, observation_noise=None,
        device='cpu'
    ):
        ''' docstring '''
        super(KalmanFilter, self).__init__(device)
        self._transition = safe_cast(torch.Tensor, transition).to(self.device)
        if emission is None:
            self._emission = torch.eye(
                self._transition.shape[-1],
                device=self.device
            )
        else:
            self._emission = safe_cast(torch.Tensor, emission).to(self.device)
        if process_noise is None:
            self._process_noise_cov = torch.eye(
                self._transition.shape[-1],
                device=self.device
            )
        else:
            self._process_noise_cov = safe_cast(torch.Tensor, process_noise).to(self.device)
        if observation_noise is None:
            self._observation_noise_cov = torch.eye(
                self.emission.shape[-1],
                device=self.device
            )
        else:
            self._observation_noise_cov = safe_cast(torch.Tensor, observation_noise).to(self.device)

    @property
    def F(self):
        return self._transition
    @property
    def H(self):
        return self._emission
    @property
    def Q(self):
        return self._process_noise_cov
    @property
    def R(self):
        return self._observation_noise_cov

    def _update(self, x, P, y):
        Py = self.H @ P @ self.H.transpose(-1,-2) + self.R
        K  = P @ self.H.transpose(-1,-2) @ torch.inverse(Py)
        return x + K@(y - self.H@x), P - K@Py@K.transpose(-1,-2)

    def forward(self, t, y, params):
        x0 = params['x0']
        P0 = params['P0']
        ix = params['obs_index']
        dt = t[1]-t[0]

        mean_predict = TimeInvariantDynamics(
            lambda x: (self.F@x - x)/dt, device=self.device
        )
        cov_predict  = TimeInvariantDynamics(
            lambda P: (self.F@P@self.F.transpose(-1,-2) + self.Q - P)/dt, device = self.device
        )
        xs, Ps = [], []
        for k, yi in enumerate(ix):
            if yi == 0:
                xi, Pi = x0+dt*mean_predict(t[0],x0), P0+dt*cov_predict(t[0],P0)
                xu, Pu = self._update(xi, Pi, y[...,k,:,:])
                xf = xu.view(*xi.shape[0:-2],1,*xi.shape[-2:])
                Pf = Pu.view(*Pi.shape[0:-2],1,*Pi.shape[-2:])
            else:
                t_ = t[ix[k-1]+1:yi+1]
                xf = odeint(mean_predict, x0, t_, method='euler')
                Pf = odeint(cov_predict,  P0, t_, method='euler')
                xi, Pi = xf[..., -1, :, :], Pf[..., -1, :, :]
                xu, Pu = self._update(xi, Pi, y[...,k,:,:])
                xf[...,-1,:,:], Pf[...,-1,:,:] = xu, Pu
            x0, P0 = xu, Pu
            xs.append(xf)
            Ps.append(Pf)
        if ix[-1] < len(t):
            t_ = t[ix[-1]+1:]
            xf = odeint(mean_predict, x0, t_, method='euler')
            Pf = odeint(cov_predict, P0, t_, method='euler')
            xs.append(xf)
            Ps.append(Pf)

        return torch.cat(xs,-3), torch.cat(Ps,-3)

    def filter(self, t, y, params):
        return self.forward(t, y, params)
