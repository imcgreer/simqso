#!/usr/bin/env python

import numpy as np
from astropy.convolution import convolve,Gaussian1DKernel
from scipy.interpolate import interp1d

def fixed_R_dispersion(lam1,lam2,R):
	'''Generate a wavelength grid with fixed resolution (i.e., logarithmic)'''
	loglam1 = np.log(lam1)
	loglam2 = np.log(lam2)
	dloglam = R**-1
	loglam = np.arange(loglam1,loglam2+dloglam,dloglam)
	return np.exp(loglam)

def deres(f,Rin,Rout,fout=None):
	assert Rout < Rin
	if fout is None:
		fout = np.zeros_like(f)
	kern = Gaussian1DKernel(Rin/Rout)
	fout[:] = convolve(f,kern,boundary='extend')
	return fout

def resample(x1,y1,x2):
	resampfun = interp1d(x1,y1)
	return resampfun(x2)

