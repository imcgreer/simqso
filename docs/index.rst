.. simqso documentation master file, created by
   sphinx-quickstart on Fri Apr  7 16:28:20 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

simqso
======

Welcome to the simqso documentation! simqso is a set of tools for generating 
mock quasar spectra and photometry. Simulated quasar spectra are built from
a series of components. Common quasar models are built-in, such as a broken
power-law continuum model and Gaussian emission line templates; however, the 
code allows user-defined features to be included. Mock spectra are generated
at arbitrary resolution and can be used to produce broadband photometry
representative of a number of surveys.

simqso consists of the following modules:

.. toctree::
   :maxdepth: 1
   :glob:

   source/*

Quasar Recipes
==============

A quasar spectra model can be constructed step-by-step either in an 
object-oriented fashion (using objects from :mod:`simqso.sqgrids`) or through 
a set of parameters defined by a dictionary. A simple reconstruction of the
BOSS DR9 quasar population using parameters can be found in bossqsos_ (see
also the notebook_), while a worked example of building a simulation from 
objects is given in GridExamples_.

.. _bossqsos: https://github.com/imcgreer/simqso/blob/master/examples/bossqsos.py
.. _notebook: https://github.com/imcgreer/simqso/blob/master/examples/bossqsos_example.ipynb
.. _GridExamples: https://github.com/imcgreer/simqso/blob/master/examples/GridExamples.ipynb

Emission lines can be added as a series of arbitary Gaussian profiles or 
using the built-in templates based on composite spectra of BOSS quasars.
See EmissionLineTemplates_ for a description of the built-in templates.

.. _EmissionLineTemplates: https://github.com/imcgreer/simqso/blob/master/examples/EmissionLineTemplates.ipynb

SimpleSpecExample_ puts this all together and shows how to build a complete
quasar spectrum, as well as how the individual components contribute to the
final spectrum.

.. _SimpleSpecExample: https://github.com/imcgreer/simqso/blob/master/examples/SimpleSpecExample.ipynb


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
