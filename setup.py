# -----------------------------------------------------------------------------
# Copyright (c) 2016, Nicolas P. Rougier
# Distributed under the (new) BSD License.
# -----------------------------------------------------------------------------
# from distutils.core import setup
# from distutils.extension import Extension
# from Cython.Distutils import build_ext
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np


extensions = [
    Extension('cdana/cdana', ['cdana/cdana.pyx'], include_dirs = [np.get_include()]),
]
setup(
    name="cdana",
    version="0.1",
    maintainer= "Nicolas P. Rougier",
    maintainer_email="Nicolas.Rougier@inria.fr",
    install_requires=['numpy', 'cython'],
    license = "BSD License",
    packages=['cdana'],
    ext_modules = cythonize(extensions)
)



#import numpy
#from distutils.core import setup, Extension
#from Cython.Build import cythonize
#setup(
#    include_dirs = [numpy.get_include()],
#    ext_modules  = cythonize("cdana.pyx"),
#)

# setup(
#     ext_modules=[
#         Extension("cdana", ["cdana.c"],
#                   include_dirs=[numpy.get_include()]),],
# )
