#!/usr/bin/env python

import os
import numpy as np
from astropy.convolution import convolve,Gaussian1DKernel
from scipy.interpolate import interp1d

datadir = os.path.split(__file__)[0]+'/data/'

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

def mag2lum(obsBand,restBand,z,cosmo,alpha_nu=-0.5):
	'''Convert observed mags to absolute mags using a simple power-law 
	   k-correction.
	'''
	z = np.asarray(z)
	DM = [cosmo.distmod(_z).value for _z in z.flat]
	DM = np.array(DM).reshape(z.shape)
	# CFHT: http://www.cfht.hawaii.edu/Science/mswg/filters.html
	effWave = {'SDSS-g':4670.,'SDSS-r':6165.,'SDSS-i':7471.,
	           'CFHT-g':4770.,'CFHT-r':6230.,'CFHT-i':7630.}
	obsWave = effWave[obsBand]
	try:
		restWave = float(restBand)
	except:
		restWave = effWave[restBand]
	# Following continuum K-corrections given in 
	#  Richards et al. 2006, AJ 131, 2766
	kcorr = -2.5*(1+alpha_nu)*np.log10(1+z) - \
	           2.5*alpha_nu*np.log10(restWave/obsWave)
	return kcorr + DM

