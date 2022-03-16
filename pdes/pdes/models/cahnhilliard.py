import torch

import numpy as np

from torchdiffeq import odeint_adjoint as odeint

from abc import abstractmethod

from ..abstract import _AbstractPDE
from ..ops import LaplacianFilter

class _AbstractCahnHilliard(_AbstractPDE):
    def __init__(self,params,dx=5e2,M=1e8,padding_mode='circular',device='cpu'):
        super(_AbstractCahnHilliard, self).__init__()
        #self._params = params
        self._assign_params(params)
        self._M      = M
        self._lapl   = LaplacianFilter(dx, padding_mode, device)

    @abstractmethod
    def _assign_params(self, _params):
        NotImplemented

    @abstractmethod
    def _dF_dphi(self, phi):
        NotImplemented

    def forward(self, t, phi):
        mu = self._dF_dphi(phi)
        return self._M*self._lapl(mu)

class LandauCahnHilliard(_AbstractCahnHilliard):
    def _assign_params(self, _params):
        self._a = torch.nn.Parameter(_params['a'])
        self._b = torch.nn.Parameter(_params['b'])
        self._k = torch.nn.Parameter(_params['k'])

    def _dF_dphi(self, phi):
        #a,b,k = [self._params[k].view(-1,*[1]*3) for k in ('a','b','k')]
        return -self._a*phi + self._b*torch.pow(phi, 3) - self._k*self._lapl(phi)

class FloryHugginsCahnHilliard(_AbstractCahnHilliard):
    def _assign_params(self, _params):
        self._chi = torch.nn.Parameter(_params['chi'])
        self._k   = torch.nn.Parameter(_params['k'])

        self._nu = 100
        self._nua, self._nub = self._nu, self._nu
        # if 'nua' in _params and 'nub' in _params:
        #     self._nua = torch.nn.Parameter(_params['nua'])
        #     self._nub = torch.nn.Parameter(_params['nub'])
        #     self._nu  = torch.sqrt(self._nua*self._nub)
        # elif 'nu' in _params:
        #     self._nu = torch.nn.Parameter(_params['nu'])
        #     self._nua, self._nub = self._nu, self._nu

        if 'Na' in _params and 'Nb' in _params:
            self._Na = torch.nn.Parameter(_params['Na'])
            self._Nb = torch.nn.Parameter(_params['Nb'])
        elif 'N' in _params:
            self._N = torch.nn.Parameter(_params['N'])
            self._Na, self._Nb = self._N, self._N

    def _dF_dphi(self, phi):
        # Na,Nb,chi,k = [self._params[k].view(-1,*[1]*3) for k in ('Na','Nb','chi','k')]

        # return (1+torch.log(phi))/(self._Na*self._nua) - (1+torch.log(1-phi))/(self._Nb*self._nub) - self._chi*(2*phi-1)/self._nu - self._k*self._lapl(phi)
        return (1+torch.log(phi))/(self._Na) - (1+torch.log(1-phi))/(self._Nb) - self._chi*(2*phi-1) - self._k*self._lapl(phi)
