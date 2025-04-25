# ------------------------------------------------------------------------------------------------
# Modified from senseTime
# Modified from Deformable-DETR
# Heming Yang
# ------------------------------------------------------------------------------------------------

import os
import glob

import torch

from torch.utils.cpp_extension import CUDA_HOME
from torch.utils.cpp_extension import CppExtension
from torch.utils.cpp_extension import CUDAExtension
from torch.utils.cpp_extension import BuildExtension

from setuptools import find_packages
from setuptools import setup

requirements = ["torch", "torchvision"]
ext_name  = 'RoiAwarePoolHelper'


def get_extensions():
    if not torch.cuda.is_available():
        raise NotImplementedError('Cuda is not availabel')
    
    this_dir = os.path.dirname(os.path.abspath(__file__))
    extensions_dir = os.path.join(this_dir, "roi_aware_pool3d/src")

    source_cpu = glob.glob(os.path.join(extensions_dir, "*.cpp"))
    source_cuda = glob.glob(os.path.join(extensions_dir, "*.cu"))

    sources =  source_cpu + source_cuda
    sources = [os.path.join(extensions_dir, s) for s in sources]
    extra_compile_args = {"cxx": []}

    extension = CUDAExtension
    extra_compile_args["nvcc"] = [
        "-DCUDA_HAS_FP16=1",
        "-D__CUDA_NO_HALF_OPERATORS__",
        "-D__CUDA_NO_HALF_CONVERSIONS__",
        "-D__CUDA_NO_HALF2_OPERATORS__",
    ]




    # include_dirs = [extensions_dir]
    ext_modules = [
        extension(
            ext_name,
            sources,
            # include_dirs=include_dirs,
            # define_macros=define_macros,
            # extra_compile_args = extra_compile_args,
        )
    ]
    return ext_modules

setup(
    name = ext_name,
    version="1.0",
    # author="",
    # url="",
    # description=" roiaware pool3d  utils ",
    packages=find_packages(exclude=("configs", "tests",)),
    ext_modules=get_extensions(),
    cmdclass={"build_ext": BuildExtension},
)
