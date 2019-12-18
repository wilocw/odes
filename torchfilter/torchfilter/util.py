from warnings import warn

import torch
from torch import nn as nn

def safe_cast(new_type, x):
    ''' A utility for safely casting x to new_type.
            If x is None, returns None
            Else if type(x) is new_type, returns x
            Else tries to case x to new_type and returns cast x
            Except where it cannot, soft fails, printing warning and returns x
             unchanged.
    '''
    if x is None:
        return None
    if type(x) is new_type:
        return x
    try:
        if new_type is torch.Tensor:
            new_x = torch.tensor(x)
        else:
            new_x = new_type(x)
    except (TypeError, ValueError) as err:
        warn('Could not cast x to new_type: {}'.format(err), RuntimeWarning, 4)
        new_x = x
    except Exception as err:
        warn('Failed to cast x: {}'.format(err), RuntimeWarning, 6)
        new_x = x
    return new_x

class StateSpaceDynamics(nn.Module):
    def __init__(self, dynamics=None, device='cpu'):
        super(StateSpaceDynamics, self).__init__()
        self.device = safe_cast(torch.device, device)
        self.to(self.device)
        self.f = dynamics if dynamics is not None else (lambda x: x)
    def forward(self, t, x):
        return self.f(t,x)

class TimeInvariantDynamics(StateSpaceDynamics):
    def forward(self, t, x):
        return self.f(x)

class Sampler(nn.Module):
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
    def from_smoother(cls, smoother):
        return cls(smoother.F, smoother.Q, device=smoother.device)

    def forward(self, m, P, G=[], n=1, jitter=None):
        dims = [*m.shape[:-4], *m.shape[-2:]]
        xn = [m[...,0,:,:] + torch.cholesky(P[...,0,:,:])@torch.randn(n,*dims)]

        _jitter = 1e-4 if jitter is None else jitter

        for i in range(1, m.shape[-3]):
            xi = xn[-1]

            mu_, mu = m[...,i-1,:,:], m[...,i,:,:]
            Sigma_, Sigma = P[...,i-1,:,:], P[...,i,:,:]

            if i > len(G):
                mu_cond  = self.F@xi
                Sig_cond = self.Q
            else:
                Prec_ = torch.inverse(Sigma_)
                G_T = G[i-1].transpose(-1,-2)

                PG = Sigma @ G_T

                mu_cond = mu + PG @ Prec_ @ (xi - mu_)
                Sig_cond = Sigma - PG @ Prec_ @ PG.transpose(-1,-2)
                Sig_cond += _jitter*torch.eye(dims[-2])

            xi_ = mu_cond + torch.cholesky(Sig_cond)@torch.randn(n,*dims)
            xn.append()
        return torch.stack(xn,-3)
