
import numpy as np

from setuptools import setup, Extension
from Cython.Build import cythonize, build_ext

setup(
    name='HierarchicalAngles',
    version='0.1.0',
    author='James S. Kuszlewicz',
    author_email='kuszlewicz@mps.mpg.de',
    license='LICENSE.txt',
    install_requires=["emcee >= 3.0rc2", "numpy", "scipy", "corner"],
    cmdclass = {'build_ext': build_ext},
    packages=[
        "hierarchicalinc",
    ],
    ext_modules= [Extension("hierarchicalinc.integrands", ["hierarchicalinc/integrands.pyx"], #cythonize('integrands.pyx'),
                 include_dirs=[np.get_include()])]
)
