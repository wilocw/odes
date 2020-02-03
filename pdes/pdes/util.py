from warnings import warn

import torch
from torch import nn as nn

import numpy as np

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

def complex_tensor(X):
    _X = X[...,None]
    return torch.cat([_X, torch.zeros_like(_X)], len(X.shape))

def conjugate(X):
    X[...,1] = -X[...,1]
    return X

def fftshift(X):
    dims  = X.shape
    nx, ny = dims[-3] // 2, dims[-2] // 2
    Y = torch.zeros_like(X)
    Y[..., 0:nx, 0:ny, :] = X[..., nx: , ny: , :]
    Y[..., 0:nx, ny: , :] = X[..., nx: , 0:ny, :]
    Y[..., nx: , ny: , :] = X[..., 0:nx, 0:ny, :]
    Y[..., nx: , 0:ny, :] = X[..., 0:nx, ny: , :]
    return Y

def scattering(phi, dx, calculate_q = False, device='cpu'):
    F = torch.fft(complex_tensor(phi), 2)
    F = fftshift(F)

    absF = torch.sqrt(F[..., 0]**2 + F[..., 1]**2)
    return radial_avg(absF, dx, calculate_q, device)

def radial_avg(X, dx, calculate_q = True, device='cpu'):
    device = safe_cast(torch.device, device)
    nx,ny = X.shape[-2:]
    xi, yi = np.indices((nx, ny))
    r = torch.tensor(
        np.sqrt((xi-0.5*nx)**2 + (yi-0.5*ny)**2).astype(np.int),
        device=device
    )
    if calculate_q:
        qsx = torch.linspace(-1/(2*dx), 1/(2*dx), nx)
        qsy = torch.linspace(-1/(2*dx), 1/(2*dx), ny)
        qr = torch.tensor(
            [[torch.sqrt((qx**2 + qy**2)) for qx in qsx] for qy in qsy],
            device=device
        )
        q = torch.bincount(r.view(-1), qr.view(-1)) / torch.bincount(r.view(-1))

    S = torch.stack(
        [torch.bincount(r.view(-1),X[i,...].view(-1))/torch.bincount(r.view(-1)) for i in range(X.shape[0])],
        0
    )
    return (q[1:],S[...,1:]) if calculate_q else S[...,1:]
