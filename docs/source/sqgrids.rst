***********************************
Simulation grids (`simqso.sqgrids`)
***********************************

Introduction
============
:mod:`simqso.sqgrids` provides classes for managing "points" (quasars) in a 
simulation. A collection of points is referred to as a grid, although they
are not required to be uniformly distributed.

:class:`simqso.sqgrids.QsoSimObjects` is the base container class for QSO grids. Objects can be 
distributed in a regular grid (:class:`simqso.sqgrids.QsoSimGrid`), or not, e.g. according to a 
luminosity function (:class:`simqso.sqgrids.QsoSimPoints`). 

Each point is defined by a set of variables (:class:`simqso.sqgrids.QsoSimVar`). The simplest
descriptive variables are absolute magnitude (:class:`simqso.sqgrids.AbsMagVar`), apparent
magnitude (:class:`simqso.sqgrids.AppMagVar`), and redshift (:class:`simqso.sqgrids.RedshiftVar`). Variables that inherit
from :class:`simqso.sqgrids.SpectralFeatureVar` define a function :func:`simqso.sqgrids.SpectralFeatureVar.add_to_spec` that knows how
to apply the variable to a spectrum. Variables can be multidimensional
(:class:`simqso.sqgrids.MultiDimVar`), e.g., :class:`simqso.sqgrids.BrokenPowerLawContinuumVar` has an extra dimension
to account for the set of power-law slopes defining the full continuum.

Each variable must know how to be randomly sampled to produce a given
realization of the simulation. Thus they are instantiated with a :class:`simqso.sqgrids.Sampler`
instance, which defines how to produce n samples of the variable.
Basic samplers include :class:`simqso.sqgrids.UniformSampler` which uniformly distributes values
between two bounds, and :class:`simqso.sqgrids.ConstSampler` which sets all values to be a single
constant. Many samplers inherit from :class:`simqso.sqgrids.CdfSampler`, which means values are 
drawn from a cumulative distribution function. For example, :class:`simqso.sqgrids.GaussianSampler`,
:class:`simqso.sqgrids.ExponentialSampler`, and :class:`simqso.sqgrids.LogNormalSampler` all generate points sampled from
the cdf of the eponymous function.

A simple tutorial for working with samplers, variables, and grids is provided
in https://github.com/imcgreer/simqso/blob/master/examples/GridExamples.ipynb.

Reference/API
=============

.. automodule:: simqso.sqgrids
    :members:
    :show-inheritance:

.. inheritance-diagram:: simqso.sqgrids
    :parts: 1

