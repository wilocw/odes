#!/usr/bin.env python
from setuptools import setup

setup(
    name='torchfilter',
    version='0.0.1',
    description='Bayesian filtering and smoothing operators in PyTorch',
    author='Wil Ward',
    packages=[
        'torchfilter'
    ],
    install_requires=[
        'torch>=1.0.0',
        'torchdiffeq @ git+ssh://git@github.com/rtqichen/torchdiffeq'
    ]
)
