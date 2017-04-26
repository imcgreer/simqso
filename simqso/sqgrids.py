#!/usr/bin/env python

import os
import ast
from copy import copy
import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import simps
from scipy.signal import convolve
from scipy.stats import norm,lognorm,expon
from astropy.io import fits
from astropy.table import Table,hstack
from astropy import units as u
from astropy import cosmology

from .sqbase import datadir,Spectrum
from . import dustextinction


##############################################################################
# Samplers
##############################################################################

class Sampler(object):
	'''
	Base class for sampling one-dimensional values within a given bound.

	Subclasses must define the sample() function.

	Parameters
	----------
	    low,high : float
	        Lower and upper bounds for the sampler.
	'''
	def __init__(self,low,high):
		self.low = low
		self.high = high
	def sample(self,n,**kwargs):
		'''
		Return a set of n values obtained from the sampler.
		'''
		raise NotImplementedError
	def resample(self,*args,**kwargs):
		pass
	def __call__(self,n,**kwargs):
		return self.sample(n,**kwargs)
	def __str__(self):
		s = str((self.low,self.high))
		return s
	@staticmethod
	def _get_arrays(arr_list,ii):
		if ii is None:
			return arr_list
		else:
			return [ a[ii] for a in arr_list ]

class FixedSampler(Sampler):
	'''
	Use a fixed set of values as the sample.

	>>> from simqso.sqgrids import FixedSampler
	>>> s = FixedSampler([1.,2.,3.])
	>>> s(3)
	[1.0, 2.0, 3.0]
	'''
	def __init__(self,vals):
		self.low = None
		self.high = None
		self.vals = vals
	def sample(self,n,**kwargs):
		if n is not None and n != len(self.vals):
			raise ValueError
		return self.vals

class NullSampler(Sampler):
	'''
	Special container for variables which are not sampled.
	'''
	def __init__(self):
		pass
	def sample(self,n,**kwargs):
		return None

class IndexSampler(Sampler):
	'''
	Special container for variables which need an index into the grid.
	'''
	def __init__(self):
		pass
	def sample(self,n,**kwargs):
		return None

class RandomSubSampler(Sampler):
	def __init__(self,n):
		super(RandomSubSampler,self).__init__(0,n)
	def sample(self,n,**kwargs):
		return np.random.randint(self.low,self.high,n)

class ConstSampler(Sampler):
	'''
	Returns a constant for all samples.

	>>> from simqso.sqgrids import ConstSampler
	>>> s = ConstSampler(17)
	>>> s(3)
	array([17, 17, 17])
	'''
	def __init__(self,*val):
		self.low = None
		self.high = None
		self.val = val
	def sample(self,n,**kwargs):
		return np.repeat(self.val,n)

class UniformSampler(Sampler):
	'''
	Returns values uniformly sampled between ``low`` and ``high``, inclusive.

	>>> from simqso.sqgrids import UniformSampler
	>>> s = UniformSampler(0,1)
	>>> s(3)
	array([ 0. ,  0.5,  1. ])
	'''
	def sample(self,n,**kwargs):
		return np.linspace(self.low,self.high,n)

class CdfSampler(Sampler):
	'''
	Returns values sampled from a cumulative distribution function, within
	the bounds passed during instantiation.

	Subclasses must implement the cdf(x) and ppf(x) functions.

	Parameters
	----------
	    low,high : float
	        Lower and upper bounds for the sampler.
	'''
	def _init_cdf(self):
		self.cdf_low = self.rv.cdf(self.low)
		self.cdf_high = self.rv.cdf(self.high)
	def _getpoints(self,x,ii=None):
		cdf_low,cdf_high = self._get_arrays((self.cdf_low,self.cdf_high),ii)
		return cdf_low + (cdf_high-cdf_low)*x
		return self.rv.ppf(x) # XXX get rid of self.rv
	def sample(self,n,**kwargs):
		x = np.random.random(n)
		return self._sample(self._getpoints(x))

class PowerLawSampler(CdfSampler):
	'''
	Returns values sampled from a power law distribution with index a.

	Unlike scipy.stats.powerlaw, allows a<0, but then requires low>0 in
	that case.

	Examples
	--------
	>>> from simqso.sqgrids import PowerLawSampler
	>>> s = PowerLawSampler(1,2,-2)
	>>> s(3)
	array([ 1.4537,  1.1208,  1.1691])
	'''
	def __init__(self,low,high,a):
		if a<0 and low<=0:
			raise ValueError
		# defining cdf and ppf function within this class
		self.rv = self # XXX get rid of self.rv
		super(PowerLawSampler,self).__init__(low,high)
		self.a = a
		self._init_cdf()
	def cdf(self,x):
		x1,x2,a = self.low,self.high,self.a
		if np.any(x<x1) or np.any(x>x2):
			raise ValueError
		return (x**(a+1) - x1**(a+1)) / (x2**(a+1) - x1**(a+1))
	def ppf(self,y):
		if np.any(y<0) or np.any(y>1):
			raise ValueError
		x1,x2,a = self.low,self.high,self.a
		return np.power( (x2**(a+1)-x1**(a+1))*y + x1**(a+1), (a+1)**-1 )

class GaussianSampler(CdfSampler):
	'''
	Returns values sampled from a Gaussian distibution N(mean,sigma).

	Examples
	--------
	>>> from simqso.sqgrids import GaussianSampler
	>>> s = GaussianSampler(50.,10.)
	>>> s(3)
	array([ 50.07  ,  42.0223,  58.9512])
	'''
	def __init__(self,mean,sigma,low=-np.inf,high=np.inf):
		super(GaussianSampler,self).__init__(low,high)
		self.mean = mean
		self.sigma = sigma
		self.rv = self
#		self._reset()
		self._init_cdf()
#	def _reset(self):
#		self.rv = norm(loc=self.mean,scale=self.sigma)
	def ppf(self,x,ii=None):
		mean,sigma = self._get_arrays((self.mean,self.sigma),ii)
		return norm.ppf(x,loc=mean,scale=sigma)
	def cdf(self,x,ii=None):
		mean,sigma = self._get_arrays((self.mean,self.sigma),ii)
		return norm.cdf(x,loc=mean,scale=sigma)
	def _sample(self,x,ii=None):
		return self.ppf(x,ii)
	def update(self,mean,sigma,ii=None):
		if ii is None: ii = np.s_[:]
		self.mean[ii] = mean
		self.sigma[ii] = sigma

#class LogNormalSampler(CdfSampler):
#	'''
#	Returns values sampled from a lognormal distibution lognorm(mean,sigma).
#
#	Examples
#	--------
#	'''
#	def __init__(self,mean,sigma,low,high):
#		if low <= 0:
#			raise ValueError
#		super(LogNormalSampler,self).__init__(low,high)
#		self.mean = mean
#		self.sigma = sigma
#		self.rv = lognorm(loc=self.mean,scale=self.sigma)
#		self._init_cdf()

class ExponentialSampler(CdfSampler):
	'''
	Returns values sampled from an exponential distibution with a given
	scale parameter.

	Examples
	--------
	>>> from simqso.sqgrids import ExponentialSampler
	>>> s = ExponentialSampler(0.1)
	>>> s(3)
	array([ 0.08072409,  0.45771082,  0.03769428])
	'''
	def __init__(self,scale,low=0,high=np.inf):
		super(ExponentialSampler,self).__init__(low,high)
		self.scale = scale
		self._reset()
		self._init_cdf()
	def _reset(self):
		self.rv = expon(scale=self.scale)

#class DoublePowerLawSampler(Sampler):
#	def __init__(self,a,b,x0,low=-np.inf,high=np.inf):
#		super(DoublePowerLawSampler,self).__init__(low,high)
#		self.a = a
#		self.b = b
#		self.x0 = x0
#	def sample(self,n):
#		raise NotImplementedError

class LinearTrendWithAsymScatterSampler(Sampler):
	'''
	Returns values sampled from a set of linear trends that define the
	Gaussian mean and sigma at each point x.

	Must be calibrated with a set of input points that define where to
	sample the linear trends.
	'''
	def __init__(self,coeffs,pts,low=-np.inf,high=np.inf):
		super(LinearTrendWithAsymScatterSampler,self).__init__(low,high)
		self.coeffs = coeffs
		self.npts = len(pts)
		self.loSampler = None
		self.hiSampler = None
		self._reset(pts)
	def _reset(self,pts,ii=None):
		xmn,xlo,xhi = [ np.polyval(c,pts) for c in self.coeffs ]
		siglo = np.clip(xmn-xlo,1e-10,np.inf)
		sighi = np.clip(xhi-xmn,1e-10,np.inf)
		if self.loSampler is None:
			self.loSampler = GaussianSampler(xmn,siglo,
			                                 low=self.low,high=self.high)
		else:
			self.loSampler.update(xmn,siglo,ii)
		if self.hiSampler is None:
			self.hiSampler = GaussianSampler(xmn,sighi,
			                                 low=self.low,high=self.high)
		else:
			self.hiSampler.update(xmn,sighi,ii)
	def _sample(self,x,ii=None):
		xlo = self.loSampler._sample(self.loSampler._getpoints(x,ii),ii)
		xhi = self.hiSampler._sample(self.hiSampler._getpoints(x,ii),ii)
		return np.clip(np.choose(x>0.5,[xlo,xhi]),0,np.inf)

class BaldwinEffectSampler(LinearTrendWithAsymScatterSampler):
	'''
	Uses LinearTrendWithAsymScatterSampler to implement the Baldwin Effect,
	by sampling from mean, upper, and lower log-linear trends as a function
	of absolute magnitude.
	'''
	def __init__(self,coeffs,absMag,x=None,low=-np.inf,high=np.inf):
		super(BaldwinEffectSampler,self).__init__(coeffs,absMag,
		                                          low=low,high=high)
		self.x = x
	def sample(self,n=None,ii=None):
		if n is None:
			n = len(self.x)
		elif n != self.npts:
			raise ValueError("BaldwinEffectSampler input does not match "
			                 "preset (%d != %d)" % (n,self.npts))
		if self.x is None:
			# save the x values for reuse
			self.x = np.random.random(n)
		x = self.x if ii is None else self.x[ii]
		return self._sample(x,ii)
	def resample(self,absMag,ii=None,**kwargs):
		self._reset(absMag,ii=ii)



##############################################################################
# Simulation variables
##############################################################################

class QsoSimVar(object):
	'''
	Base class for variables used to define points within simulation grid.
	Each variable must have a name and a Sampler instance for generating
	values of the variable.

	Parameters
	----------
	sampler : :class:`simqso.sqgrids.Sampler` instance
	name : str
	    Unique name for variable.
	'''
	def __init__(self,sampler,name=None):
		self.sampler = sampler
		if name is not None:
			self.name = name
		self.update = False
		self.meta = {}
	def __call__(self,n,**kwargs):
		return self.sampler(n,**kwargs)
	def resample(self,*args,**kwargs):
		'''
		Update the samplers of any dependent variables and then resample.
		'''
		self.sampler.resample(*args,**kwargs)
	def __str__(self):
		return str(self.sampler)
	def updateMeta(self,meta):
		'''
		Update the meta-data dictionary associated with the variable.
		'''
		for k,v in self.meta.items():
			meta[k] = v

class MultiDimVar(QsoSimVar):
	'''
	Special case of QsoSimVar that handles multi-dimensional variables.
	The last dimension must be a sequence of Sampler instances, which can
	be nested in as many outer dimensions as necessary.
	'''
	def _recurse_call(self,samplers,n,**kwargs):
		if isinstance(samplers,Sampler):
			return samplers(n,**kwargs)
		else:
			return [ self._recurse_call(sampler,n,**kwargs) 
			           for sampler in samplers ]
	def _recurse_resample(self,samplers,*args,**kwargs):
		if isinstance(samplers,Sampler):
			samplers.resample(*args,**kwargs)
		else:
			for sampler in samplers:
				self._recurse_resample(sampler,*args,**kwargs)
	def __call__(self,n,**kwargs):
		arr = self._recurse_call(self.sampler,n,**kwargs)
		return np.rollaxis(np.array(arr),-1)
	def resample(self,*args,**kwargs):
		self._recurse_resample(self.sampler,*args,**kwargs)

class SpectralFeatureVar(object):
	'''
	Mix-in class to define variables that act on spectra.

	Subclasses must define the render() function.
	'''
	def render(self,wave,z,par,**kwargs):
		raise NotImplementedError
	def add_to_spec(self,spec,par,**kwargs):
		'''
		Applies the variable to an input spectrum.

		Parameters
		----------
		spec : :class:`simqso.sqbase.Spectrum` instance
		par : sampled values of the variable that are passed to render()
		'''
		spec.f_lambda[:] += self.render(spec.wave,spec.z,par,**kwargs)
		return spec

class AppMagVar(QsoSimVar):
	'''
	An apparent magnitude variable, defined in an observed bandpass ``band``.
	'''
	name = 'appMag'
	def __init__(self,sampler,band):
		super(AppMagVar,self).__init__(sampler)
		self.obsBand = band

class AbsMagVar(QsoSimVar):
	'''
	An absolute magnitude variable, defined at rest-frame wavelength
	``restWave`` in Angstroms.
	'''
	name = 'absMag'
	def __init__(self,sampler,restWave=None):
		'''if restWave is none then bolometric'''
		super(AbsMagVar,self).__init__(sampler)
		self.restWave = restWave

class RedshiftVar(QsoSimVar):
	'''
	A redshift variable.
	'''
	name = 'z'

class ContinuumVar(QsoSimVar,SpectralFeatureVar):
	'''
	Base class for variables that define the quasar spectral continuum.
	'''
	pass

def _Mtoflam(lam0,M,z,DM):
	nu0 = (lam0 * u.Angstrom).to(u.Hz,equivalencies=u.spectral()).value
	fnu0 = 10**(-0.4*(M+DM(z)+48.599934))
	flam0 = nu0*fnu0/lam0
	return flam0/(1+z)

class BrokenPowerLawContinuumVar(ContinuumVar,MultiDimVar):
	'''
	Representation of a quasar continuum as a series of broken power laws.

	Parameters
	----------
	samplers : sequence of :class:`simqso.sqgrids.Sampler` instances
	    Each sampler instance defines the power law spectral index at a given
	    section of the continuum, as alpha_nu where f_nu = nu^alpha_nu.
	breakPts : sequence of floats
	    Break wavelengths in Angstroms.

	Examples
	--------
	>>> from simqso.sqgrids import BrokenPowerLawContinuumVar,GaussianSampler
	>>> v = BrokenPowerLawContinuumVar([GaussianSampler(-1.5,0.3),GaussianSampler(-0.5,0.3)],[1215.7])
	>>> v(3)
	array([[-1.801, -1.217],
	       [-1.56 , -0.594],
	       [-1.605, -0.248]])
	'''
	name = 'slopes'
	def __init__(self,samplers,breakPts):
		super(BrokenPowerLawContinuumVar,self).__init__(samplers)
		self.breakPts = np.asarray(breakPts).astype(np.float32)
		self.meta['CNTBKPTS'] = ','.join(['%.1f' % b for b in self.breakPts])
	def render(self,wave,z,slopes,fluxNorm=None):
		'''
		Renders the broken power law continuum at redshift ``z`` given the
		set of sampled ``slopes``. Aribrarily normalized unless the
		``fluxNorm`` parameter is supplied.

		Parameters
		----------
		fluxNorm : dict
		    wavelength : float
		        rest-frame wavelength in Angstroms at which to normalize 
		        spectrum.
		    M_AB : float
		        absolute AB magnitude at ``wavelength``
		    DM : function to return distance modulus, as in ``DM(z)``
		'''
		spec = np.zeros_like(wave)
		w1 = 1
		spec[0] = 1.0
		z1 = 1 + z
		alpha_lams = -(2+np.asarray(slopes)) # a_nu --> a_lam
		# add a breakpoint beyond the red edge of the spectrum in order
		# to fill using the last power law slope if necessary
		breakpts = np.concatenate([[0,],self.breakPts,[wave[-1]+1]])
		wb = np.searchsorted(wave,breakpts*z1)
		ii = np.where((wb>0)&(wb<=len(wave)))[0]
		wb = wb[ii]
		for alpha_lam,w2 in zip(alpha_lams[ii-1],wb):
			if w1==w2:
				break
			spec[w1:w2] = spec[w1-1] * (wave[w1:w2]/wave[w1-1])**alpha_lam
			w1 = w2
		if fluxNorm is not None:
			normwave = fluxNorm['wavelength']
			wave0 = wave/z1
			fnorm = _Mtoflam(normwave,fluxNorm['M_AB'],z,fluxNorm['DM'])
			if wave0[0] > normwave:
				raise NotImplementedError("outside of wave range: ",
				                          wave0[0],normwave)
				# XXX come back to this; for normalizing the flux when the norm
				#     wavelength is outside of the spectral range
				for alam,bkpt in zip(alpha_lams,breakpts):
					if bkpt > normwave:
						fnorm *= (normwave/bkpt)**alam
					if bkpt > wave0[0]:
						break
			elif wave0[-1] < normwave:
				raise NotImplementedError("%.1f (%.1f) outside lower "
				 "wavelength bound %.1f" % (wave0[-1],wave[-1],normwave))
			else:
				# ... to be strictly correct, would need to account for power law
				#     slope within the pixel
				fscale = fnorm/spec[np.searchsorted(wave0,normwave)]
			spec[:] *= fscale
		return spec

class EmissionFeatureVar(QsoSimVar,SpectralFeatureVar):
	'''
	Base class for variables that define quasar spectral emission features.
	'''
	pass

def render_gaussians(wave,z,lines):
	emspec = np.zeros_like(wave)
	lines = lines[lines[:,1]>0]
	lineWave,eqWidth,sigma = lines.T * (1+z)
	A = eqWidth/(np.sqrt(2*np.pi)*sigma)
	twosig2 = 2*sigma**2
	nsig = (np.sqrt(-2*np.log(1e-3/A))*np.array([[-1.],[1]])).T
	ii = np.where(np.logical_and(lineWave>wave[0],lineWave<wave[-1]))[0]
	for i in ii:
		i1,i2 = np.searchsorted(wave,lineWave[i]+nsig[i]*sigma[i])
		emspec[i1:i2] += A[i]*np.exp(-(wave[i1:i2]-lineWave[i])**2
		                                   / twosig2[i])
	return emspec

class GaussianEmissionLineVar(EmissionFeatureVar,MultiDimVar):
	'''
	A single Gaussian emission line. Must be instantiated with three samplers
	for the profile, namely (wavelength, equivalent width, sigma). All
	parameters are given in the rest-frame and in Angstroms.

	Examples
	--------
	>>> from simqso.sqgrids import GaussianEmissionLineVar
	>>> v = GaussianEmissionLineVar([GaussianSampler(1215.7,0.1),GaussianSampler(100.,10.),GaussianSampler(10.,1.)])
	>>> v(3)
	array([[ 1215.645,   113.125,     9.099],
	       [ 1215.987,   109.654,     9.312],
	       [ 1215.74 ,   101.765,    10.822]])
	'''
	def render(self,wave,z,par):
		return render_gaussians(wave,z,np.array([par]))

class GaussianLineEqWidthVar(EmissionFeatureVar):
	'''
	this is an arguably kludgy way of making it possible to include
	line EW as a variable in grids, by reducing the line to a single
	parameter

	Parameters
	----------
	sampler : :class:`simqso.sqgrids.Sampler` instance
	    Sampler for generating equivalent width values.
	name : str 
	    Name of emission line.
	wave0,width0 : float
	    Fixed Gaussian parameters for the rest-frame wavelength and sigma
	    in Angstroms. Only the equivalent width is sampled.
	'''
	def __init__(self,sampler,name,wave0,width0,log=False):
		super(GaussianLineEqWidthVar,self).__init__(sampler,name)
		self.wave0 = wave0
		self.width0 = width0
		self.log = log
	def render(self,wave,z,ew0):
		if self.log:
			ew0 = np.power(10,ew0)
		return render_gaussians(wave,z,
		                        np.array([[self.wave0,ew0,self.width0]]))

class GaussianEmissionLinesTemplateVar(EmissionFeatureVar,MultiDimVar):
	'''
	A multidimensional variable representing a template of Gaussian-profile
	emission lines.
	'''
	name = 'emLines'
	def render(self,wave,z,lines):
		return render_gaussians(wave,z,lines)

class BossDr9EmissionLineTemplateVar(GaussianEmissionLinesTemplateVar):
	'''
	Subclass of GaussianEmissionLinesTemplateVar that obtains log-linear
	trends for the emission lines from the BOSS DR9 model (Ross et al. 2013).
	TODO: this should really start with the file
	'''
	def __init__(self,samplers,lineNames):
		super(BossDr9EmissionLineTemplateVar,self).__init__(samplers)
		self.lineNames = lineNames
		self.meta['LINEMODL'] = 'BOSS DR9 Log-linear trends with luminosity'
		self.meta['LINENAME'] = ','.join(lineNames)
	def __call__(self,n=None,ii=None):
		lpar = super(BossDr9EmissionLineTemplateVar,self).__call__(n,ii=ii)
		lpar[...,1:] = np.power(10,lpar[...,1:])
		return lpar

class FeTemplateVar(EmissionFeatureVar):
	'''
	Variable used to store an iron emission template, and then render it
	at an input redshift.

	Since the template is fixed it uses a :class:`simqso.sqgrids.NullSampler`
	instance internally.
	'''
	def __init__(self,feGrid):
		super(FeTemplateVar,self).__init__(NullSampler())
		self.feGrid = feGrid
	def render(self,wave,z,par):
		return self.feGrid.get(z)

class HIAbsorptionVar(QsoSimVar,SpectralFeatureVar):
	'''
	Variable used to store IGM HI absorption spectra.

	Since the spectra are precomputed a :class:`simqso.sqgrids.IndexSampler`
	instance is used internally to map the forest sightlines to individual 
	spectra.
	'''
	name = 'igmlos'
	def __init__(self,forest,losMap=None):
		N = forest.numSightLines
		if losMap is None:
			s = RandomSubSampler(N)
		else:
			s = FixedSampler(losMap)
		super(HIAbsorptionVar,self).__init__(s)
		self.forest = forest
	def add_to_spec(self,spec,sightLine,advance=True,**kwargs):
		if advance:
			T = self.forest.next_spec(sightLine,spec.z)
		else:
			# this is needed when iterating the spectrum -- don't want to
			# advance to the next redshift, just keep reusing current forest
			T = self.forest.current_spec(sightLine,spec.z)
		spec.f_lambda[:len(T)] *= T
		return spec

class DustExtinctionVar(QsoSimVar,SpectralFeatureVar):
	'''
	Base class for dust extinction features. Dust curves are provided in the
	rest frame and convolved with input spectra.
	'''
	@staticmethod
	def dustCurve(name):
		return dustextinction.dust_fn[name]
	def add_to_spec(self,spec,ebv,**kwargs):
		spec.convolve_restframe(self.dustCurve(self.dustCurveName),ebv)
		return spec

class SMCDustVar(DustExtinctionVar):
	'''
	SMC dust extinction curve from XXX.
	'''
	name = 'smcDustEBV'
	dustCurveName = 'SMC'
	meta = {'DUSTMODL':'SMC'}

class CalzettiDustVar(DustExtinctionVar):
	'''
	Calzetti XXX dust extinction curve for starburst galaxies.
	'''
	name = 'calzettiDustEBV'
	dustCurveName = 'CalzettiSB'
	meta = {'DUSTMODL':'Calzetti Starburst'}

class BlackHoleMassVar(QsoSimVar):
	'''
	A black hole mass variable, in units of log(Msun).
	'''
	name = 'logBhMass'

class EddingtonRatioVar(QsoSimVar):
	'''
	A dimensionless Eddington ratio variable, as lambda_edd = L/L_edd.
	'''
	name = 'logEddRatio'

class AbsMagFromAppMagVar(AbsMagVar):
	'''
	A variable that provides a conversion from apparent magnitude to
	absolute magnitude.

	Internally uses a :class:`simqso.sqgrids.FixedSampler` instance after
	converting to absMag.

	Parameters
	----------
	appMag : ndarray
	    Apparent magnitudes (usually from an AppMagVar).
	m2M : function
	    Conversion from apparent to absolute mag, as m2M(z) = K(z) + DM(z)
	restWave : float
		Rest wavelength in Angstroms for the absolute magnitudes.
	'''
	def __init__(self,appMag,m2M,restWave=None):
		absMag = m2M(appMag)
		sampler = FixedSampler(absMag)
		super(AbsMagFromAppMagVar,self).__init__(sampler,restWave)

class AbsMagFromBHMassEddRatioVar(AbsMagVar):
	'''
	A variable that provides a conversion from black hole mass and Eddington
	ratio to absolute magnitude.

	Internally uses a :class:`simqso.sqgrids.FixedSampler` instance after
	converting to absMag.

	TODO: uses a fixed BC estimate, should be an input.

	Parameters
	----------
	logBhMass : ndarray
	    Log of black hole mass in Msun. (e.g., from an BlackHoleMassVar).
	logEddRatio : ndarray
	    Log of dimensionless Eddington ratio (e.g., from an EddingtonRatioVar).
	restWave : float
		Rest wavelength in Angstroms for the absolute magnitudes.
	'''
	def __init__(self,logBhMass,logEddRatio,restWave=None):
		eddLum = 1.26e38 * 10**logBhMass
		lum = 10**logEddRatio * eddLum
		BC1450 = 5.0 # rough value from Richards+06
		lnu1450 = lum / BC1450
		M1450 = magnitude_AB_from_L_nu(lnu1450/2e15)
		sampler = FixedSampler(M1450)
		super(AbsMagFromBHMassEddRatioVar,self).__init__(sampler,restWave)

class SynMagVar(QsoSimVar):
	'''
	Container for synthetic magnitudes.
	'''
	name = 'synMag'

class SynFluxVar(QsoSimVar):
	'''
	Container for synthetic fluxes.
	'''
	name = 'synFlux'


##############################################################################
# Simulation grids
##############################################################################

class QsoSimObjects(object):
	'''
	A collection of simulated quasar objects. Objects are defined by a set
	of variables (`QsoSimVar`). The values for the variables are maintained 
	internally as an `astropy.table.Table`, which can be saved and restored.

	Parameters
	----------
	qsoVars : list of `QsoSimVar` instances
	    Set of variables used to initialize the simulation grid.
	cosmo : `astropy.cosmology.FLRW` instance
	    Cosmology used for the simulation.
	units : str
		One of "flux" or "luminosity", XXX should be handled internally...
	'''
	def __init__(self,qsoVars=[],cosmo=None,units=None):
		self.qsoVars = qsoVars
		self.cosmo = cosmo
		self.units = units
	def setCosmology(self,cosmodef):
		if type(cosmodef) is dict:
			self.cosmo = cosmology.FlatLambdaCDM(**cosmodef)
		elif isinstance(cosmodef,basestring):
			self.cosmo = cosmology.FlatLambdaCDM(**eval(cosmodef))
		elif isinstance(cosmodef,cosmology.FLRW):
			self.cosmo = cosmodef
		elif cosmodef is None:
			self.cosmo = cosmology.get_current()
		else:
			raise ValueError
	def __iter__(self):
		for obj in self.data:
			yield obj
	def group_by(self,varName,with_index=False):
		if with_index:
			self.data['_ii'] = np.arange(self.nObj)
		data_grouped = self.data.group_by(varName)
		if with_index:
			ii = self.data['_ii'].copy()
			del self.data['_ii']
			return data_grouped.groups,ii
		else:
			return data_grouped.groups
	def __getattr__(self,name):
		try:
			return self.data[name]
		except KeyError:
			raise AttributeError("no attribute "+name)
	def addVar(self,var):
		'''
		Add a variable to the simulation.
		'''
		self.qsoVars.append(var)
		vals = var(self.nObj)
		if vals is not None:
			self.data[var.name] = vals
	def addVars(self,newVars):
		'''
		Add a list of variables to the simulation.
		'''
		for var in newVars:
			self.addVar(var)
	def addData(self,data):
		self.data = hstack([self.data,data])
	def getVars(self,varType=QsoSimVar):
		'''
		Return all variables that are instances of varType.
		'''
		return filter(lambda v: isinstance(v,varType),self.qsoVars)
	def resample(self):
		for var in self.qsoVars:
			if var.update:
				var.resample(self.data[var.dependentVars])
				self.data[var.name] = var(self.nObj)
	def distMod(self,z):
		return self.cosmo.distmod(z).value
	def read(self,gridFile,clean=True):
		'''
		Read a simulation grid from a file.
		'''
		self.data = Table.read(gridFile)
		if clean:
			# XXX it's hacky to be aware of these colnames here, but need to
			# know how to delete derived quantities that will be recomputed
			for k in ['obsFlux','obsMag','obsFluxErr','obsMagErr',
			          'synMag','synFlux']:
				del self.data[k]
		self.nObj = len(self.data)
		hdr = fits.getheader(gridFile,1)
		self.units = hdr['GRIDUNIT']
		self.gridShape = eval(hdr['GRIDDIM'])
		hdr = fits.getheader(gridFile,1)
		self.simPars = ast.literal_eval(hdr['SQPARAMS'])
		self.setCosmology(self.simPars['Cosmology'])
	@staticmethod
	def cosmo_str(cosmodef):
		if isinstance(cosmodef,cosmology.FLRW):
			d = dict(name=cosmodef.name,H0=cosmodef.H0.value,
			         Om0=cosmodef.Om0)
			if cosmodef.Ob0:
				d['Ob0'] = cosmodef.Ob0
			cosmodef = d
		return str(cosmodef)
	def write(self,simPars,outputDir='.',outFn=None):
		'''
		Write a simulation grid to a FITS file as a binary table, storing 
		meta-data in the header.
		'''
		tab = self.data
		simPars = copy(simPars)
		simPars['Cosmology'] = self.cosmo_str(simPars['Cosmology'])
		if 'QLFmodel' in simPars['GridParams']:
			s = str(simPars['GridParams']['QLFmodel']).replace('\n',';')
			simPars['GridParams']['QLFmodel'] = s
		tab.meta['SQPARAMS'] = str(simPars)
		tab.meta['GRIDUNIT'] = self.units
		tab.meta['GRIDDIM'] = str(self.gridShape)
		for var in self.qsoVars:
			var.updateMeta(tab.meta)
		if outFn is None:
			outFn = simPars['FileName']+'.fits'
		tab.write(os.path.join(outputDir,outFn),overwrite=True)

class QsoSimPoints(QsoSimObjects):
	'''
	Simulation grid represented as a list of points.

	Parameters
	----------
	qsoVars : list of `QsoSimVar` instances
	    Set of variables used to initialize the simulation grid.
	n : int
	    Number of points in the grid. Not required (None) if the input
	    variables already know how to sample the correct number of points
	    (e.g., if they all use a `FixedSampler`).
	'''
	def __init__(self,qsoVars,n=None,**kwargs):
		super(QsoSimPoints,self).__init__(qsoVars,**kwargs)
		data = { var.name:var(n) for var in qsoVars }
		self.data = Table(data)
		self.nObj = len(self.data)
		self.gridShape = (self.nObj,)
	def __str__(self):
		return str(self.data)

class QsoSimGrid(QsoSimObjects):
	'''
	Simulation grid represented as a uniform grid. Within each grid cell
	``nPerBin`` objects are randomly sampled to fill the cell.

	Parameters
	----------
	qsoVars : list of `QsoSimVar` instances
	    Set of variables used to initialize the simulation grid.
	nBins : tuple
		Number of bins along each grid axis (i.e., each variable).
	nPerBin : int
		Number of objects within each grid cell.
	'''
	def __init__(self,qsoVars,nBins,nPerBin,**kwargs):
		super(QsoSimGrid,self).__init__(qsoVars,**kwargs)
		self.gridShape = nBins + (nPerBin,)
		axes = [ var(n+1) for n,var in zip(nBins,qsoVars) ]
		self.gridEdges = np.meshgrid(*axes,indexing='ij')
		self.gridCenters = [ a[:-1]+np.diff(a)/2 for a in axes ]
		data = {}
		for i,(v,g) in enumerate(zip(qsoVars,self.gridEdges)):
			x = np.random.random(self.gridShape)
			s = [ slice(0,-1,1) for j in range(len(qsoVars)) ]
			pts0 = g[s][...,np.newaxis] 
			binsz = np.diff(g,axis=i)
			s[i] = slice(None)
			pts = pts0 + x*binsz[s][...,np.newaxis]
			data[v.name] = pts.flatten()
		self.data = Table(data)
		self.nObj = len(self.data)
	def asGrid(self,name):
		# in case the column has extra axes (i.e., for flux vectors)
		outShape = self.gridShape + self.data[name].shape[1:]
		return np.asarray(self.data[name]).reshape(outShape)
	def __str__(self):
		s = "grid dimensions: "+str(self.gridShape)+"\n"
		s += str(self.gridEdges)+"\n"
		s += str(self.data)
		return s



def generateQlfPoints(qlf,mRange,zRange,m2M,cosmo,band,**kwargs):
	'''
	Generate a `QsoSimPoints` grid fed by `AppMagVar` and `RedshiftVar`
	instances which are sampled from an input luminosity function.
	'''
	m,z = qlf.sample_from_fluxrange(mRange,zRange,m2M,cosmo,**kwargs)
	m = AppMagVar(FixedSampler(m),band=band)
	z = RedshiftVar(FixedSampler(z))
	return QsoSimPoints([m,z],units='flux',cosmo=cosmo)

def generateBEffEmissionLines(M1450,**kwargs):
	trendFn = kwargs.get('EmissionLineTrendFilename','emlinetrends_v6')
	indy = kwargs.get('EmLineIndependentScatter',False)
	noScatter = kwargs.get('NoScatter',False)
	excludeLines = kwargs.get('ExcludeLines',[])
	onlyLines = kwargs.get('OnlyLines')
	M_i = M1450 - 1.486 + 0.596
	lineCatalog = Table.read(datadir+trendFn+'.fits')
	for line,scl in kwargs.get('scaleEWs',{}).items():
		i = np.where(lineCatalog['name']==line)[0][0]
		lineCatalog['logEW'][i,:,1] += np.log10(scl)
	if noScatter:
		for k in ['wavelength','logEW','logWidth']:
			lineCatalog[k][:,1:] = lineCatalog[k][:,[0]]
	if indy:
		x1 = x2 = x3 = None
	else:
		x1 = np.random.random(len(M_i))
		x2 = np.random.random(len(M_i))
		x3 = np.random.random(len(M_i))
	#
	useLines = ~np.in1d(lineCatalog['name'],excludeLines)
	if onlyLines is not None:
		useLines &= np.in1d(lineCatalog['name'],onlyLines)
	#
	lineList = [ (BaldwinEffectSampler(l['wavelength'],M_i,x1),
	              BaldwinEffectSampler(l['logEW'],M_i,x2),
	              BaldwinEffectSampler(l['logWidth'],M_i,x3))
	             for l in lineCatalog[useLines] ]
	lines = BossDr9EmissionLineTemplateVar(lineList,
	                                       lineCatalog['name'][useLines])
	lines.update = True # XXX a better way?
	lines.dependentVars = 'absMag'
	return lines

def generateVdBCompositeEmLines(minEW=1.0,noFe=False):
	all_lines = Table.read(datadir+'VandenBerk2001_AJ122_549_table2.txt',
	                       format='ascii')
	# blended lines are repeated in the table
	l,li = np.unique(all_lines['OWave'],return_index=True)
	lines = all_lines[li]
	li = np.where(lines['EqWid'] > minEW)[0]
	lines = lines[li]
	#
	if noFe:
		isFe = lines['ID'].find('Fe') == 0
		lines = lines[~isFe]
	print 'using the following lines from VdB template: ',
	print ','.join(list(lines['ID']))
	c = ConstSampler
	lineList = [ [c(l['OWave']),c(l['EqWid']),c(l['Width'])] for l in lines ]
	lines = GaussianEmissionLinesTemplateVar(lineList)
	lines.meta['LINEMODL'] = 'Fixed Vanden Berk et al. 2001 emission lines'
	return lines


class VW01FeTemplateGrid(object):
	def __init__(self,z,wave,fwhm=5000.,scales=None,useopt=True):
		z1 = max(0,z.min()-0.1)
		z2 = z.max() + 0.1
		nz = int((z2-z1)/0.005) + 1
		self.zbins = np.linspace(z1,z2,nz)
		self.feGrid = np.empty((self.zbins.shape[0],wave.shape[0]))
		self.useopt = useopt
		# the Fe template is an equivalent width spectrum
		wave0,ew0 = self._restFrameFeTemplate(fwhm,scales)
		for i,zbin in enumerate(self.zbins):
			# in units of EW - no (1+z) when redshifting
			rfEW = interp1d(wave0*(1+zbin),ew0,kind='slinear',
			                bounds_error=False,fill_value=0.0)
			self.feGrid[i] = rfEW(wave)
	def _loadVW01Fe(self,wave):
		fepath = datadir+'VW01_Fe/'
		feTemplate = np.zeros_like(wave)
		if self.useopt:
			templnames = ['Fe_UVOPT_V01_T06_BR92','Fe2_UV191','Fe3_UV47']
		else:
			templnames = ['Fe_UVtemplt_B','Fe2_UV191','Fe3_UV47']
		tmplfits = fits.open(os.path.join(datadir,'simqso_templates.fits'))
		for t in templnames:
			extnm = t if 'UVOPT' in t else 'VW01_'+t
			tspec = tmplfits[extnm].data
			spec = interp1d(tspec['wave'],tspec['f_lambda'],kind='slinear')
			w1,w2 = np.searchsorted(wave,[tspec['wave'][0],tspec['wave'][-1]])
			feTemplate[w1:w2] += spec(wave[w1:w2])
		return feTemplate
	def _restFrameFeTemplate(self,FWHM_kms,feScalings):
		if self.useopt:
			wave = np.logspace(np.log(1075.),np.log(7500.),9202,base=np.e)
		else:
			wave = np.logspace(np.log(1075.),np.log(3089.),5000,base=np.e)
		feTemplate = self._loadVW01Fe(wave)
		# rescale segments of the Fe template
		if feScalings is None:
			feScalings = [(0,1e4,1.0),]
		print 'using Fe scales: ',feScalings
		for w1,w2,fscl in feScalings:
			wi1,wi2 = np.searchsorted(wave,(w1,w2))
			feTemplate[wi1:wi2] *= fscl
		# calculate the total flux (actually, EW since continuum is divided out)
		flux0 = simps(feTemplate,wave)
		FWHM_1Zw1 = 900.
		c_kms = 3e5
		sigma_conv = np.sqrt(FWHM_kms**2 - FWHM_1Zw1**2) / \
		                     (2*np.sqrt(2*np.log(2))) / c_kms
		dloglam = np.log(wave[1]) - np.log(wave[0])
		x = np.arange(-5*sigma_conv,5*sigma_conv,dloglam)
		gkern = np.exp(-x**2/(2*sigma_conv**2)) / (np.sqrt(2*np.pi)*sigma_conv)
		broadenedTemp = convolve(feTemplate,gkern,mode='same')
		feFlux = broadenedTemp
		feFlux *= flux0/simps(feFlux,wave)
		return wave,feFlux
	def get(self,z):
		zi = np.searchsorted(self.zbins,z)
		return self.feGrid[zi]

