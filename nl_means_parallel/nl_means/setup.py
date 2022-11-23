from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    ext_modules=cythonize("filters/_nl_means_denoising.pyx"),
    include_dirs=[numpy.get_include()]
)
