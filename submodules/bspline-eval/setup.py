from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name="bspline_eval",
    packages=["bspline_eval"],
    ext_modules=[
        CUDAExtension(
            name="bspline_eval._C",
            sources=["bspline_eval_kernel.cu"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
