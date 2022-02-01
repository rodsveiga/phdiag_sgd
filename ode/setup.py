from setuptools import setup
from Cython.Build import cythonize
import numpy as np

setup(
    name='odecy',
    ext_modules=cythonize('odecy.pyx'),
    zip_safe=False,
    include_dirs=[np.get_include()]
)
