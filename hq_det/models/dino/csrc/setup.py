#!/usr/bin/env python
# Setup script for asymmetric focal loss CUDA extension

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ext_modules = [
    CUDAExtension(
        name='asymmetric_focal_loss_cuda',
        sources=[
            'asymmetric_focal_loss.cpp',
            'asymmetric_focal_loss_cuda.cu',
        ],
        extra_compile_args={
            'cxx': ['-O3', '-std=c++17'],
            'nvcc': ['-O3', '-std=c++17'],
        },
    )
]

setup(
    name='asymmetric_focal_loss_cuda',
    version='0.1.0',
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExtension},
    zip_safe=False,
)
