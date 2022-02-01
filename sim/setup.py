from setuptools import setup
from Cython.Build import cythonize
import numpy as np

setup(
    name='simcy',
    ext_modules=cythonize('simcy.pyx'),
    zip_safe=False,
    include_dirs=[np.get_include()]
)
