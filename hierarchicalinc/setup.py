
import numpy as np

from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize, build_ext

setup(
    name='HierarchicalAngles',
    version='0.1.0',
    author='James S. Kuszlewicz',
    author_email='kuszlewicz@mps.mpg.de',
    license='LICENSE.txt',
    install_requires=["emcee >= 3.0rc2"],
    cmdclass = {'build_ext': build_ext},
    ext_modules= [Extension("integrands", ["integrands.pyx"], #cythonize('integrands.pyx'),
                 include_dirs=[np.get_include()])]
)