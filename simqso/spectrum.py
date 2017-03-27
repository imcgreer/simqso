#!/usr/bin/env python

import numpy as np
#from astropy.convolution import convolve,Gaussian1DKernel
from scipy.interpolate import interp1d
from astropy import units as u

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

def _Mtoflam(lam0,M,z,DM):
	nu0 = (lam0 * u.Angstrom).to(u.Hz,equivalencies=u.spectral()).value
	fnu0 = 10**(-0.4*(M+DM(z)+48.599934))
	flam0 = nu0*fnu0/lam0
	return flam0/(1+z)

class QSOSpectrum(Spectrum):
	def __init__(self,wave,**kwargs):
		super(QSOSpectrum,self).__init__(wave,**kwargs)
		self.templates = {}
	def convolve_restframe(self,g,*args):
		self.f_lambda = g(self.wave/(1+self.z),self.f_lambda,*args)
	def addTemplate(self,name,template):
		self.templates[name] = self.plcontinuum * template
		self.f_lambda += self.templates[name]

