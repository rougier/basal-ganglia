# -----------------------------------------------------------------------------
# Copyright (c) 2016, Nicolas P. Rougier
# Distributed under the (new) BSD License.
# -----------------------------------------------------------------------------
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension('cdana.cdana', ['cdana/cdana.pyx'], include_dirs = [np.get_include()]),
]
setup(
    name="cdana",
    version="0.1",
    maintainer= "Nicolas P. Rougier",
    maintainer_email="Nicolas.Rougier@inria.fr",
    install_requires=['numpy', 'cython', 'tqdm'],
    license = "BSD License",
    packages=['cdana'],
    ext_modules = cythonize(extensions)
)
