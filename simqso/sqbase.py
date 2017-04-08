#!/usr/bin/env python

import os
import numpy as np
from astropy.convolution import convolve,Gaussian1DKernel
from scipy.interpolate import interp1d
from astropy import units as u

datadir = os.path.split(__file__)[0]+'/data/'

def fixed_R_dispersion(lam1,lam2,R):
	'''Generate a wavelength grid with fixed resolution using logarithmic
	   wavelength bins.

	   Parameters
	   ----------
	   lam1,lam2 : float
	       Wavelengths endpoints in Angstroms.
	   R : float
	       Resolution [ d(lam)/lam ]

	   Returns
	   -------
	   wave : `~numpy.ndarray`
	       Wavelength array between lam1 and lam2 with resolution R.

	   Examples
	   --------
	   >>> from simqso.sqbase import fixed_R_dispersion
	   >>> wave = fixed_R_dispersion(3000,3010,1000)
	   >>> print wave
	   [ 3000.          3003.0015005   3006.006004    3009.01351351  3012.02403203]
	'''
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
	effWave = {'SDSS-g':4670.,'SDSS-r':6165.,'SDSS-i':7471.,'SDSS-z':8918,
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

class Spectrum(object):
	#
	def __init__(self,wave,f_lambda=None,z=0.0):
		'''Spectrum(wave,**kwargs)
		    wave: set the wavelength array (default flux will be zero)
		    keyword arguments:
		     f_lambda: flux density in f_lambda units
			 z: redshift of object
		'''
		self.wave = wave.astype(np.float) # XXX
		if f_lambda is None:
			self.f_lambda = np.zeros_like(self.wave)
		else:
			self.f_lambda = f_lambda
		self.z = z
	#
	def _getotherspec(self,other):
		if np.isscalar(other):
			a2 = other
		else:
			if ((self.wave.size == other.wave.size) and 
			    (np.max(np.abs(self.wave-other.wave)/self.wave) < 1e-3)):
				a2 = other.f_lambda
			else:
				warnings.warn('interpolated spectrum!')
				ofunc = interp1d(other.wave,other.f_lambda,
				                 bounds_error=False,fill_value=0.0)
				a2 = ofunc(self.wave)
		return a2
	def _op(self,other,op):
		'''generic math operations on spectrum, interpolating in wavelength
		   if necessary.'''
		a1 = self.f_lambda
		a2 = self._getotherspec(other)
		return Spectrum(self.wave,op(a1,a2),self.z)
	# implement +-*/
	def __add__(self,other):
		return self._op(other,np.add)
	def __sub__(self,other):
		return self._op(other,np.subtract)
	def __mul__(self,other):
		return self._op(other,np.multiply)
	def __div__(self,other):
		return self._op(other,np.divide)
	#
	def setRedshift(self,z):
		self.z = z
	def clear(self):
		self.z = -1.0
		self.f_lambda[:] = 0
	#
	def waveslice(self,w1,w2):
		'''return a cut of the spectrum in a wavelength range'''
		ii = np.where((self.wave > w1) & (self.wave < w2))[0]
		return Spectrum(self.wave[ii],self.f_lambda[ii],self.z)
	def resample(self,newWave):
		newFlux = interp1d(self.wave,self.f_lambda,
		                   bounds_error=False,fill_value=0.0)
		self.wave = newWave
		self.f_lambda = newFlux(newWave)
	def convolve_restframe(self,g,*args):
		self.f_lambda = g(self.wave/(1+self.z),self.f_lambda,*args)

