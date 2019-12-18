#!/usr/bin.env python
from setuptools import setup

setup(
    name='pdes',
    version='0.0.1',
    description='Experiments in parameter estimation with PDEs',
    author='Wil Ward',
    packages=[
        'pdes'
    ],
    install_requires=[
        'torch>=1.0.0',
        'torchdiffeq @ git+ssh://git@github.com/rtqichen/torchdiffeq'
    ]
)
