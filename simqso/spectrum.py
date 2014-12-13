#!/usr/bin/env python

import numpy as np
#from astropy.convolution import convolve,Gaussian1DKernel
from scipy.interpolate import interp1d

class Spectrum(object):
	#
	def __init__(self,z,**kwargs):
		'''Spectrum(z,**kwargs)
			z: redshift of object
		    keyword arguments:
		     data: (wave,flux) spectrum is set to these values
		                       all other keywords are ignored
		     wave: set the wavelength array (flux will be zero)'''
		try:
			self.wave,self.f_lambda = kwargs['data']
		except KeyError:
			try:
				self.wave = kwargs['wave'].astype(np.float64)
			except KeyError:
				# using different dispersions seems to be broken
				raise ValueError
				self.wave = default_waveset.copy()
			self.f_lambda = np.zeros_like(self.wave)
		self.components = {}
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
		return Spectrum(self.z,data=(self.wave,op(a1,a2)))
	def add_op(self,other,op):
		sp = self._op(other,op)
		try:
			for c in self.components:
				sp.components[c] = self.components[c]
			for c in other.components:
				sp.components[c] = other.components[c]
		except AttributeError:
			# wasn't a Spectrum object
			pass
		return sp
	# implement +-*/
	def __add__(self,other):
		return self.add_op(other,np.add)
	def __sub__(self,other):
		return self._op(other,np.subtract)
	def __mul__(self,other):
		return self.add_op(other,np.multiply)
	def __div__(self,other):
		return self._op(other,np.divide)
	#
	def setRedshift(self,z):
		self.z = z
	def clear(self):
		del self.components
		self.z = -1.0
		self.components = {}
	#
	def waverange(self,w1,w2):
		'''return a cut of the spectrum in a wavelength range'''
		ii = np.where((self.wave > w1) & (self.wave < w2))[0]
		rv = Spectrum(self.z,data=(self.wave[ii],self.f_lambda[ii]))
		return rv
	#

