import torch
from torch import nn as nn
from torch.distributions import MultivariateNormal

from .utils import safe_cast

from abc import ABC as _Abstract
from abc import abstractmethod

def _AbstractKernel(nn.Module, _Abstract):
    def __init__(self, device='cpu', dt=None):
        super(_AbstractKernel, self).__init__()
        self.to(safe_cast(torch.device, device))

    def _generate_dynamics(self, dt):
        self._flag_disc = False if dt is None else True

        if self._flag_disc:
            # self._F = torch.ex

    @abstractmethod
    def _transition(self):
        NotImplemented
    @abstractmethod
    def _emission(self):
        NotImplemented
    @abstractmethod
    def _process_noise(self):
        NotImplemented

    def forward(self, dt)
