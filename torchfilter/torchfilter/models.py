import torch
from torch import nn as nn
from torch.distributions import MultivariateNormal


from .utils import safe_cast, Sampler

from .filters import KalmanFilter
from .smoothers import RauchTungStriebelSmoother

from abc import ABC as _Abstract
from abc import abstractmethod

def _extract_from_dict(d, keys):
    if isinstance(keys, list):
        for k in keys:
            if k in d:
                return d[k]
        return None
    else:
        return d[keys] if keys in d else None

def _parse_dynamics(dynamics):
    if dynamics is None:
        return (None, None, None, None)
    _F = _extract_from_dict(dynamics, ['F','transition'])
    _Q = _extract_from_dict(dynamics, ['Q','process_noise','transition_noise'])
    _H = _extract_from_dict(dynamics, ['H','emission'])
    _R = _extract_from_dict(dynamics, ['R','sigma2','observation_noise','emission_noise'])
    return (_F,_Q,_H,_R)

class _Model(nn.Module, _Abstract):
    def __init__(self, device='cpu'):
        super(_Model, self).__init__()
        self.device = safe_cast(torch.device, device)
        self.to(self.device)

class LinearStateSpaceModel(_Model):
    def __init__(self,
        initial_condition=None, device='cpu', **kwargs):
        ''' docstring '''
        super(LinearStateSpaceModel, self).__init__(device)

        F, Q, H, R = _parse_dynamics(kwargs)

        if _x0 is

        self._kf = KalmanFilter(F, H, Q, R, device)
        self._rts = RauchTungStriebelSmoother.from_filter(self._kf)

    @property
    def initial_condition(self):
        return

    @initial_condition.setter
    def initial_condition(self, x):
        pass

    def train(self, ts, ys):
        ''' Optimise log likelihood '''
        NotImplemented

    def predict(self, t):
        ''' '''
        NotImplemented

class StateSpacegaussianProcess(LinearStateSpaceModel):
    def __init__(self, kernel=)
