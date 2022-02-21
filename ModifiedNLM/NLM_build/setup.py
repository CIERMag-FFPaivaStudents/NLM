from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy

ext_modules = [
    Extension(
        "_nl_means_denoising",
        ["_nl_means_denoising.pyx"],
        extra_compile_args=['-fopenmp','-O2'],
        extra_link_args=['-fopenmp','-O2'],
        include_dirs=[numpy.get_include()]
    )
]

setup(
    name='_nl_means_denoising',
    ext_modules=cythonize(ext_modules),
    include_dirs=[numpy.get_include()]

)
