#!/usr/bin/env python

import numpy as np
#from astropy.convolution import convolve,Gaussian1DKernel
from scipy.interpolate import interp1d
from astropy import units as u

class Spectrum(object):
	#
	def __init__(self,wave,**kwargs):
		'''Spectrum(wave,**kwargs)
		    wave: set the wavelength array (default flux will be zero)
		    keyword arguments:
			 z: redshift of object
		     flux: flux in f_lambda units
		'''
		self.wave = wave.astype(np.float)
		self.f_lambda = kwargs.get('f_lambda',np.zeros_like(self.wave))
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
	def waveslice(self,w1,w2):
		'''return a cut of the spectrum in a wavelength range'''
		ii = np.where((self.wave > w1) & (self.wave < w2))[0]
		return Spectrum(self.wave[ii],flux=self.f_lambda[ii],z=self.z)
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
	#
	def setPowerLawContinuum(self,plaws,fluxNorm=None):
		self.components['PowerLawContinuum'] = plaws
		w1 = 1
		self.f_lambda[0] = 1.0
		z1 = 1 + self.z
		slopes,breakpts = plaws
		alpha_lams = -(2+slopes) # a_nu --> a_lam
		breakpts = breakpts.astype(np.float32)
		wb = np.searchsorted(self.wave,breakpts*z1)
		ii = np.where((wb>0)&(wb<=len(self.wave)))[0]
		wb = wb[ii]
		for alpha_lam,w2 in zip(alpha_lams[ii-1],wb):
			if w1==w2:
				break
			self.f_lambda[w1:w2] = \
			   self.f_lambda[w1-1] * \
			     (self.wave[w1:w2]/self.wave[w1-1])**alpha_lam
			w1 = w2
		if fluxNorm is not None:
			normwave = fluxNorm['wavelength']
			wave0 = self.wave/z1
			fnorm = _Mtoflam(normwave,fluxNorm['M_AB'],self.z,fluxNorm['DM'])
			if wave0[0] > normwave:
				raise NotImplementedError('outside of wave range: ',wave0[0],normwave)
				# XXX come back to this; for normalizing the flux when the norm
				#     wavelength is outside of the spectral range
				for alam,bkpt in zip(alpha_lams,breakpts):
					if bkpt > normwave:
						fnorm *= (normwave/bkpt)**alam
					if bkpt > wave0[0]:
						break
			elif wave0[-1] < normwave:
				raise NotImplementedError
			else:
				# ... to be strictly correct, would need to account for power law
				#     slope within the pixel
				fscale = fnorm/self.f_lambda[np.searchsorted(wave0,normwave)]
			self.f_lambda *= fscale
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
	def convolve_restframe(self,g,*args):
		self.f_lambda = g(self.wave/(1+self.z),self.f_lambda,*args)
	def addTemplate(self,name,template):
		self.templates[name] = self.plcontinuum * template
		self.f_lambda += self.templates[name]

