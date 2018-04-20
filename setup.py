# -*- coding=utf-8 -*-
"""这是surprise的cpy编译脚本"""
from setuptools import setup, Extension
# use:
# python setup.py build_ext --inplace
try:
    import numpy as np
except ImportError:
    exit('Please install numpy>=1.11.2 first.')


from Cython.Build import cythonize
__version__ = 'latest'
ext = '.pyx'

extensions = [
    Extension(
        'surprise.similarities',
        ['surprise/similarities' + ext],
        include_dirs=[np.get_include()]
    ),
    Extension(
        'surprise.prediction_algorithms.matrix_factorization',
        ['surprise/prediction_algorithms/matrix_factorization' + ext],
        include_dirs=[np.get_include()]),
    Extension('surprise.prediction_algorithms.optimize_baselines',
              ['surprise/prediction_algorithms/optimize_baselines' + ext],
              include_dirs=[np.get_include()]),
    Extension('surprise.prediction_algorithms.slope_one',
              ['surprise/prediction_algorithms/slope_one' + ext],
              include_dirs=[np.get_include()]),
    Extension('surprise.prediction_algorithms.co_clustering',
              ['surprise/prediction_algorithms/co_clustering' + ext],
              include_dirs=[np.get_include()]),
]


ext_modules = cythonize(extensions)

setup(
    ext_modules=ext_modules
)
