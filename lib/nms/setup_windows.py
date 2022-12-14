# ------------------------------------------------------------------
# Copyright (c) Nvidia
# Licensed under BSD 3-Clause "New" or "Revised" License
# Modified from Apex (https://github.com/NVIDIA/apex/)
# ------------------------------------------------------------------

import torch
from setuptools import setup, find_packages
import subprocess
from distutils.extension import Extension

import sys
import warnings
import os
import numpy as np

# ninja build does not work unless include_dirs are abs path
this_dir = os.path.dirname(os.path.abspath(__file__))

# Obtain the numpy include directory.  This logic works across numpy versions.
try:
    numpy_include = np.get_include()
except AttributeError:
    numpy_include = np.get_numpy_include()

if not torch.cuda.is_available():
    # https://github.com/NVIDIA/apex/issues/486
    # Extension builds after https://github.com/pytorch/pytorch/pull/23408 attempt to query torch.cuda.get_device_capability(),
    # which will fail if you are compiling in an environment without visible GPUs (e.g. during an nvidia-docker build command).
    print('\nWarning: Torch did not find available GPUs on this system.\n',
          'If your intention is to cross-compile, this is not an error.\n'
          'Volta (compute capability 7.0), and Turing (compute capability 7.5).\n'
          'If you wish to cross-compile for a single specific architecture,\n'
          'export TORCH_CUDA_ARCH_LIST="compute capability" before running setup.py.\n')
    if os.environ.get("TORCH_CUDA_ARCH_LIST", None) is None:
        os.environ["TORCH_CUDA_ARCH_LIST"] = "6.0;6.1;6.2;7.0;7.5"

print("torch.__version__  = ", torch.__version__)
TORCH_MAJOR = int(torch.__version__.split('.')[0])
TORCH_MINOR = int(torch.__version__.split('.')[1])

cmdclass = {}
ext_modules = []
extras = {}

from torch.utils.cpp_extension import BuildExtension
from torch.utils.cpp_extension import CUDAExtension
cmdclass['build_ext'] = BuildExtension

def check_cuda_torch_binary_vs_bare_metal(cuda_dir):
    raw_output = subprocess.check_output([cuda_dir + "/bin/nvcc", "-V"], universal_newlines=True)
    output = raw_output.split()
    release_idx = output.index("release") + 1
    release = output[release_idx].split(".")
    bare_metal_major = release[0]
    bare_metal_minor = release[1][0]
    torch_binary_major = torch.version.cuda.split(".")[0]
    torch_binary_minor = torch.version.cuda.split(".")[1]

    print("\nCompiling cuda extensions with")
    print(raw_output + "from " + cuda_dir + "/bin\n")

    if (bare_metal_major != torch_binary_major) or (bare_metal_minor != torch_binary_minor):
        raise RuntimeError("Cuda extensions are being compiled with a version of Cuda that does " +
                           "not match the version used to compile Pytorch binaries.  " +
                           "Pytorch binaries were compiled with Cuda {}.\n".format(torch.version.cuda) +
                           "In some cases, a minor-version mismatch will not cause later errors")

# Set up macros for forward/backward compatibility hack around
# https://github.com/pytorch/pytorch/commit/4404762d7dd955383acee92e6f06b48144a0742e
# and
# https://github.com/NVIDIA/apex/issues/456
# https://github.com/pytorch/pytorch/commit/eb7b39e02f7d75c26d8a795ea8c7fd911334da7e#diff-4632522f237f1e4e728cb824300403ac
version_ge_1_1 = []
if (TORCH_MAJOR > 1) or (TORCH_MAJOR == 1 and TORCH_MINOR > 0):
    version_ge_1_1 = ['-DVERSION_GE_1_1']
version_ge_1_3 = []
if (TORCH_MAJOR > 1) or (TORCH_MAJOR == 1 and TORCH_MINOR > 2):
    version_ge_1_3 = ['-DVERSION_GE_1_3']
version_ge_1_5 = []
if (TORCH_MAJOR > 1) or (TORCH_MAJOR == 1 and TORCH_MINOR > 4):
    version_ge_1_5 = ['-DVERSION_GE_1_5']
version_dependent_macros = version_ge_1_1 + version_ge_1_3 + version_ge_1_5

check_cuda_torch_binary_vs_bare_metal(torch.utils.cpp_extension.CUDA_HOME)

ext_modules.append(
    Extension(
        "cpu_nms",
        ["cpu_nms.pyx"],
        extra_compile_args={'cxx': ['/MD']},
        include_dirs = [numpy_include]
    ),
)

ext_modules.append(
    CUDAExtension(name='gpu_nms',
                    sources=['nms_kernel.cu', 'gpu_nms.pyx'],
                    include_dirs = [numpy_include],
                    extra_compile_args={'cxx': ['-O3',] + version_dependent_macros,
                                        'nvcc':['-O3',
                                                '-gencode', 'arch=compute_70,code=sm_70',
                                                '-U__CUDA_NO_HALF_OPERATORS__',
                                                '-U__CUDA_NO_HALF_CONVERSIONS__',
                                                '--expt-relaxed-constexpr',
                                                '--expt-extended-lambda',
                                                '--use_fast_math'] + version_dependent_macros}))

setup(
    name='nms',
    version='0.1',
    description='',
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    extras_require=extras,
)
