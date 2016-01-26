from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension('cdana', ['cdana.pyx'], include_dirs = [np.get_include()]),
]
setup( ext_modules = cythonize(extensions) )



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
