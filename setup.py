#!/usr/bin/env python

from distutils.core import setup, Command

try:  # Python 3.x
    from distutils.command.build_py import build_py_2to3 as build_py
except ImportError:  # Python 2.x
    from distutils.command.build_py import build_py

from simqso import __version__

setup(name='simqso',
      version=__version__,
      description='Simulated Quasar Spectra',
      author='Ian McGreer',
      author_email='imcgreer@gmail.com,',
      license='BSD3',
      url='http://github.com/imcgreer/simqso',
      packages=['simqso'],
      provides=['simqso'],
      package_data={'simqso':['data/*.*']},
      requires=['numpy', 'matplotlib','scipy','astropy'],
      keywords=['Scientific/Engineering'],
     )

