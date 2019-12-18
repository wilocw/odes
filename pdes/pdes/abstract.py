import torch

import numpy as np

from torchdiffeq import odeint_adjoint as odeint

from abc import ABC as _Abstract
from abc import abstractmethod

class _AbstractPDE(torch.nn.Module, _Abstract):
    def __init__(self):
        super(_AbstractPDE, self).__init__()
