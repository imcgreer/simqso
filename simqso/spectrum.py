#!/usr/bin/env python

import numpy as np
#from astropy.convolution import convolve,Gaussian1DKernel
from scipy.interpolate import interp1d

class Spectrum(object):
	#
	def __init__(self,wave,**kwargs):
		'''Spectrum(wave,**kwargs)
		    wave: set the wavelength array (default flux will be zero)
		    keyword arguments:
			 z: redshift of object
		     flux: flux in f_lambda units
		'''
		self.wave = wave
		self.f_lambda = np.zeros_like(self.wave)
		self.components = {}
		self.z = kwargs.get('z',0.0)
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
		return Spectrum(self.wave,flux=self.f_lambda,z=self.z)
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
		return Spectrum(self.wave[ii],flux=self.f_lambda[ii],z=self.z)
		return rv
	#

class QSOSpectrum(Spectrum):
	def __init__(self,wave,**kwargs):
		Spectrum.__init__(self,wave,**kwargs)
		self.templates = {}
	#
	def setPowerLawContinuum(self,plaws,fluxNorm=None):
		self.components['PowerLawContinuum'] = plaws
		w1 = 1
		self.f_lambda[0] = 1.0
		z = self.z
		for plaw in plaws:
			if plaw[0] > 0:
				break_wave = plaw[0]*(1+z)
				w2 = np.searchsorted(self.wave,break_wave,side='right')
			else:
				w2 = self.wave.size
			if w1==w2:
				break
			alpha_lam = -(2+plaw[1]) # a_nu --> a_lam
			self.f_lambda[w1:w2] = \
			   self.f_lambda[w1-1] * \
			     (self.wave[w1:w2]/self.wave[w1-1])**alpha_lam
			w1 = w2
		if fluxNorm is not None:
			normwave = fluxNorm['wavelength']
			if plaws[0][0] > normwave or len(plaws)==1:
				norm_index = plaws[0][1]
			else:
				for pp,plaw in enumerate(plaws[1:],start=1):
					if plaw[0] > normwave:
						norm_index = plaws[pp-1][1]
						break
					elif plaw[0] < 0:
						norm_index = plaw[1]
			li = np.abs(self.wave-fluxNorm['wavelength']*(1+z)).argmin()
			if 'M_AB' in fluxNorm:
				mAB_off,M_AB_off = cmag.magnitude_AB1450(z,self.f_lambda[li],
				                            self.wave[li],nu_power=norm_index,
				                            **def_cosmo)
				dm = fluxNorm['M_AB'] - M_AB_off
				self.f_lambda *= 10**(-0.4*dm)
			else:
				# normalize to one at the specfied wavelength
				self.f_lambda /= self.f_lambda[li]
		self.plcontinuum = self.f_lambda.copy()
	#
	def addEmissionLines(self,emlines):
		self.components['EmissionLines'] = emlines
		wave,eqWidth,sigma = [p*(1+self.z) for p in emlines]
		self.templates['EmissionLines'] = np.zeros_like(self.plcontinuum)
		nsig = 3.5*np.array([-1.,1])
		A = eqWidth/(np.sqrt(2*np.pi)*sigma)
		twosig2 = 2*sigma**2
		for i in xrange(wave.shape[0]):
			i1,i2 = np.searchsorted(self.wave,wave[i]+nsig*sigma[i])
			if i2 != i1:
				lineprofile = A[i]*np.exp(-(self.wave[i1:i2]-wave[i])**2
				                           / twosig2[i])
				self.templates['EmissionLines'][i1:i2] += \
				                         self.plcontinuum[i1:i2]*lineprofile
		self.f_lambda += self.templates['EmissionLines']
	def addTemplate(self,name,template):
		self.templates[name] = self.plcontinuum * template
		self.f_lambda += self.templates[name]

