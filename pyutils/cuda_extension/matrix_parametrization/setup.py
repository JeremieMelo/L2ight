import torch.cuda
from setuptools import setup
from torch.utils.cpp_extension import CppExtension, CUDAExtension, BuildExtension
from torch.utils.cpp_extension import CUDA_HOME

ext_modules = []

if torch.cuda.is_available() and CUDA_HOME is not None:
    extension = CUDAExtension(
        'matrix_parametrization_cuda', [
            'matrix_parametrization_cuda.cpp',
            'matrix_parametrization_cuda_kernel.cu'
        ],
        extra_compile_args={'cxx': ['-g', "-fopenmp"],
                            'nvcc': ['-O3', "-arch=sm_60",
                                "-gencode=arch=compute_60,code=sm_60",
                                "-gencode=arch=compute_61,code=sm_61",
                                "-gencode=arch=compute_70,code=sm_70",
                                "-gencode=arch=compute_75,code=sm_75",
                                "-gencode=arch=compute_75,code=compute_75", "--use_fast_math"]})
    ext_modules.append(extension)

setup(
    name='matrix_parametrization_cuda',
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExtension})
