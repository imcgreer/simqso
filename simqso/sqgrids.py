#!/usr/bin/env python

from __future__ import print_function

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
from astropy.constants import sigma_sb,b_wien
try:
    from astropy.modeling.blackbody import blackbody_lambda
except ImportError:
    #from astropy.analytic_functions import blackbody_lambda
    def blackbody_lambda(in_x, temperature):
        import astropy.constants as const
        import astropy.units as u
        from astropy.utils.exceptions import AstropyUserWarning
        # Units
        FNU = u.erg / (u.cm**2 * u.s * u.Hz)
        FLAM = u.erg / (u.cm**2 * u.s * u.AA)
        def blackbody_nu(in_x, temperature):
            # Convert to units for calculations, also force double precision
            with u.add_enabled_equivalencies(u.spectral() + u.temperature()):
                freq = u.Quantity(in_x, u.Hz, dtype=np.float64)
                temp = u.Quantity(temperature, u.K, dtype=np.float64)
            # Check if input values are physically possible
            if temp < 0:
                raise ValueError('Invalid temperature {0}'.format(temp))
            if np.any(freq <= 0):  # pragma: no cover
                warnings.warn('Input contains invalid wavelength/frequency value(s)',
                              AstropyUserWarning)
            # Calculate blackbody flux
            bb_nu = (2.0 * const.h * freq ** 3 /
                     (const.c ** 2 * np.expm1(const.h * freq / (const.k_B * temp))))
            flux = bb_nu.to(FNU, u.spectral_density(freq))
            return flux / u.sr  # Add per steradian to output flux unit
        bb_nu = blackbody_nu(in_x, temperature) * u.sr  # Remove sr for conversion
        flux = bb_nu.to(FLAM, u.spectral_density(in_x))
        return flux / u.sr  # Add per steradian to output flux unit

from .sqbase import datadir,Spectrum
from . import dustextinction
from . import sqphoto


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
    def __str__(self):
        return 'FixedSampler'

class NullSampler(Sampler):
    '''
    Special container for variables which are not sampled.
    '''
    def __init__(self):
        pass
    def sample(self,n,**kwargs):
        return None
    def __str__(self):
        return 'NullSampler'

class IndexSampler(Sampler):
    '''
    Special container for variables which need an index into the grid.
    '''
    def __init__(self):
        pass
    def sample(self,n,**kwargs):
        return None
    def __str__(self):
        return 'IndexSampler'

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
    def __str__(self):
        return 'UniformSampler(%s,%s)' % (self.low,self.high)

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
        self.cdf_low = self.cdf(self.low)
        self.cdf_high = self.cdf(self.high)
    def _getpoints(self,x,ii=None):
        cdf_low,cdf_high = self._get_arrays((self.cdf_low,self.cdf_high),ii)
        return cdf_low + (cdf_high-cdf_low)*x
    def _sample(self,x,ii=None):
        return self.ppf(x,ii)
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
        self._init_cdf()
    def ppf(self,x,ii=None):
        mean,sigma = self._get_arrays((self.mean,self.sigma),ii)
        return norm.ppf(x,loc=mean,scale=sigma)
    def cdf(self,x,ii=None):
        mean,sigma = self._get_arrays((self.mean,self.sigma),ii)
        return norm.cdf(x,loc=mean,scale=sigma)
    def update(self,mean,sigma,ii=None):
        if ii is None: ii = np.s_[:]
        self.mean[ii] = mean
        self.sigma[ii] = sigma
    def __str__(self):
        return 'GaussianSampler(%s,%s,%s,%s)' % \
            (self.mean,self.sigma,self.low,self.high)

#class LogNormalSampler(CdfSampler):
#    '''
#    Returns values sampled from a lognormal distibution lognorm(mean,sigma).
#
#    Examples
#    --------
#    '''
#    def __init__(self,mean,sigma,low,high):
#        if low <= 0:
#            raise ValueError
#        super(LogNormalSampler,self).__init__(low,high)
#        self.mean = mean
#        self.sigma = sigma
#        self.rv = lognorm(loc=self.mean,scale=self.sigma)
#        self._init_cdf()

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
        self._init_cdf()
    def ppf(self,x,ii=None):
        scale = self._get_arrays((self.scale,),ii)[0]
        return expon.ppf(x,scale=scale)
    def cdf(self,x,ii=None):
        scale = self._get_arrays((self.scale,),ii)[0]
        return expon.cdf(x,scale=scale)

#class DoublePowerLawSampler(Sampler):
#    def __init__(self,a,b,x0,low=-np.inf,high=np.inf):
#        super(DoublePowerLawSampler,self).__init__(low,high)
#        self.a = a
#        self.b = b
#        self.x0 = x0
#    def sample(self,n):
#        raise NotImplementedError

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
    seed : int
        Seed to apply to RNG before sampling the variable. If None there
        is no call to random.seed().
    '''
    def __init__(self,sampler,name=None,seed=None,meta=None):
        self.sampler = sampler
        if name is not None:
            self.name = name
        self.meta = {}
        self.dependentVars = None
        self.assocVar = None
        self.dtype = np.float32
        self.seed = seed
        self.varmeta = meta
    def __call__(self,n,**kwargs):
        if self.seed:
            np.random.seed(self.seed)
        vals = self.sampler(n,**kwargs)
        if vals is not None:
            vals = np.array(vals).astype(self.dtype)
        return vals
    def resample(self,*args,**kwargs):
        '''
        Update the samplers of any dependent variables and then resample.
        '''
        self.sampler.resample(*args,**kwargs)
    def _sampler_to_string(self):
        return str(self.sampler)
    def set_associated_var(self,assocVar):
        self.assocVar = assocVar
    def get_associated_var(self):
        return self.assocVar
    def set_seed(self,seed,overwrite=False):
        '''
        Update the random seed used when sampling the variable. If
        overwrite is False an exisiting seed will be preserved.
        '''
        if overwrite or self.seed is None:
            self.seed = seed
    def updateMeta(self,meta,axPfx):
        '''
        Update the meta-data dictionary associated with the variable.
        '''
        for k,v in self.meta.items():
            meta[k] = v
        meta[axPfx+'TYPE'] = self.__class__.__name__
        meta[axPfx+'NAME'] = str(self.name)
        meta[axPfx+'SMPL'] = self._sampler_to_string()
        if self.seed is not None:
            meta[axPfx+'SEED'] = str(self.seed)
        if self.varmeta is not None:
            meta[axPfx+'META'] = str(self.varmeta)

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
        return np.rollaxis(np.array(arr),-1).astype(self.dtype)
    def resample(self,*args,**kwargs):
        self._recurse_resample(self.sampler,*args,**kwargs)
    def _sampler_to_string(self):
        return 'MultiDimVar(%d)' % len(self.sampler)

class SpectralFeatureVar(object):
    '''
    Mix-in class to define variables that act on spectra.

    Subclasses must define the render() function.
    '''
    def render(self,wave,z,par,assocvals=None):
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
    def __init__(self,sampler,obsBand,**kwargs):
        super(AppMagVar,self).__init__(sampler,**kwargs)
        self.obsBand = obsBand
    def updateMeta(self,meta,axPfx):
        super(AppMagVar,self).updateMeta(meta,axPfx)
        meta[axPfx+'VARG'] = '"%s"' % self.obsBand

class AbsMagVar(QsoSimVar):
    '''
    An absolute magnitude variable, defined at rest-frame wavelength
    ``restWave`` in Angstroms.
    '''
    name = 'absMag'
    def __init__(self,sampler,restWave=None,**kwargs):
        '''if restWave is none then bolometric'''
        super(AbsMagVar,self).__init__(sampler,**kwargs)
        self.restWave = restWave
    def updateMeta(self,meta,axPfx):
        super(AbsMagVar,self).updateMeta(meta,axPfx)
        meta[axPfx+'VARG'] = 'restWave=%s' % self.restWave

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
    >>> v = BrokenPowerLawContinuumVar([GaussianSampler(-1.5,0.3),
                                        GaussianSampler(-0.5,0.3)],
                                       [1215.7])
    >>> v(3)
    array([[-1.801, -1.217],
           [-1.56 , -0.594],
           [-1.605, -0.248]])
    '''
    name = 'slopes'
    def __init__(self,samplers,breakPts,**kwargs):
        super(BrokenPowerLawContinuumVar,self).__init__(samplers,**kwargs)
        self.breakPts = np.asarray(breakPts).astype(np.float32)
    def _normalize(self,wave,z,slopes,fluxNorm=None,
                   minwave=12.4,getspec=True):
        z1 = 1 + z
        alpha_lams = -(2+np.asarray(slopes)) # a_nu --> a_lam
        alpha_lams = alpha_lams.squeeze() # comes in as [1,N] from grid
        breakpts = self.breakPts
        if breakpts[0] > minwave:
            breakpts = np.concatenate([[minwave],breakpts]) * z1
        f0 = np.ones_like(breakpts)
        for i,breakwv in enumerate(breakpts[1:],start=1):
            wv1 = breakpts[i-1]
            f0[i] = f0[i-1]*(breakwv/wv1)**alpha_lams[i-1]
        if fluxNorm is not None:
            normwv = fluxNorm['wavelength']
            fnorm = _Mtoflam(normwv,fluxNorm['M_AB'],z,fluxNorm['DM'])
            i = np.searchsorted(self.breakPts,normwv)
            fatnorm = f0[i]*(normwv/self.breakPts[i-1])**alpha_lams[i]
            f0 *= fnorm/fatnorm
        if getspec:
            ii = np.digitize(wave,self.breakPts*z1)
            spec = f0[ii]*(wave/breakpts[ii])**alpha_lams[ii]
            return spec
        else:
            return f0,alpha_lams,breakpts
    def render(self,wave,z,slopes,fluxNorm=None,assocvals=None):
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
        return self._normalize(wave,z,slopes,fluxNorm)
    def total_flux(self,slopes,fluxNorm,z,minwave=12.4,maxwave=1.3e4):
        f0,alpha_lams,breakwvs = self._normalize(None,z,slopes,fluxNorm,
                                                 minwave=minwave,
                                                 getspec=False)
        breakwvs /= (1+z)
        i = np.searchsorted(breakwvs,maxwave) - 1
        breakwvs = breakwvs[:i]
        breakwvs = np.concatenate([breakwvs,[maxwave]])
        f0,alpha_lams = f0[:i],alpha_lams[:i]
        waveratio = breakwvs[1:]/breakwvs[:-1]
        # integrate the power law spectra across the break
        ftot = np.sum( breakwvs[:-1]*f0*(np.power(waveratio,alpha_lams+1)-1)
                         / (alpha_lams+1))
        return ftot
    def updateMeta(self,meta,axPfx):
        super(BrokenPowerLawContinuumVar,self).updateMeta(meta,axPfx)
        meta[axPfx+'VARG'] = str(list(self.breakPts))

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
    def render(self,wave,z,par,assocvals=None):
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
    def __init__(self,sampler,name,wave0,width0,log=False,**kwargs):
        super(GaussianLineEqWidthVar,self).__init__(sampler,name,**kwargs)
        self.wave0 = wave0
        self.width0 = width0
        self.log = log
    def render(self,wave,z,ew0,assocvals=None):
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
    def render(self,wave,z,lines,assocvals=None):
        return render_gaussians(wave,z,lines)

class BossDr9EmissionLineTemplateVar(GaussianEmissionLinesTemplateVar):
    '''
    Subclass of GaussianEmissionLinesTemplateVar that obtains log-linear
    trends for the emission lines from the BOSS DR9 model (Ross et al. 2013).
    TODO: this should really start with the file
    '''
    def __init__(self,samplers,lineNames=None,**kwargs):
        super(BossDr9EmissionLineTemplateVar,self).__init__(samplers,**kwargs)
        if lineNames is not None:
            self.lineNames = lineNames
            self.meta['LINEMODL'] = 'BOSS DR9 Log-linear trends with luminosity'
            self.meta['LINENAME'] = ','.join(lineNames)
        self.dependentVars = 'absMag'
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
    name = 'fetempl'
    def __init__(self,feGrid=None,**kwargs):
        super(FeTemplateVar,self).__init__(NullSampler(),**kwargs)
        if feGrid is not None:
            self.set_template_grid(feGrid)
    def set_template_grid(self,feGrid):
        self.feGrid = feGrid
        self.varmeta = dict(fwhm=self.feGrid.fwhm,scales=self.feGrid.scales,
                            useopt=self.feGrid.useopt)
    def render(self,wave,z,par,assocvals=None):
        return self.feGrid.get(z)

class DustBlackbodyVar(ContinuumVar,MultiDimVar):
    '''
    Variable used to represent a warm dust component as a single blackbody,
    treated as a continuum.
    '''
    def __init__(self,*args,**kwargs):
        self.approxMode = kwargs.pop('approx_mode','table')
        super(DustBlackbodyVar,self).__init__(*args,**kwargs)
        self._init_bb()
    def _init_bb(self):
        self.rfwave = {}
        self.Blam = {}
    def _calc_bb(self,Tdust,wave,z):
        lampeak = (b_wien/(Tdust*u.K)).to('Angstrom').value
        lam1,lam2 = np.array([0.25,15])*lampeak
        if Tdust not in self.Blam:
            npts = 100
            # this hack keeps the points more closely spaced near lampeak
            dwv1 = np.logspace(np.log10(lam1),np.log10(lampeak),npts)
            dwv1 = np.cumsum(np.diff(dwv1))
            rfwv2 = np.logspace(np.log10(lampeak),np.log10(lam2),npts)
            rfwave = np.concatenate([ (lampeak-dwv1)[::-1], rfwv2 ])
            self.rfwave[Tdust] = rfwave
            bvals = blackbody_lambda(self.rfwave[Tdust],Tdust).value
            self.Blam[Tdust] = interp1d(self.rfwave[Tdust],bvals,
                                        kind='cubic')
        Blam = self.Blam[Tdust]
        i1,i2 = np.searchsorted(wave/(1+z),[lam1,lam2])
        flam = np.zeros_like(wave)
        flam[i1:i2] = Blam(wave[i1:i2]/(1+z))
        return flam
    def render(self,wave,z,par,fluxNorm=None,assocvals=None):
        assert isinstance(self.assocVar,ContinuumVar)
        assert assocvals is not None
        fracdust,Tdust = par
        L_bb = (sigma_sb.cgs * Tdust**4).value
        fdisk = self.assocVar.total_flux(assocvals,fluxNorm,z)
        flux_bb = fracdust * fdisk
        Blam = self._calc_bb(Tdust,wave,z)
        return flux_bb * np.pi * Blam / L_bb

class SightlineVar(QsoSimVar):
    '''
    Variable used to associate quasars with lines-of-sight.

    Since the spectra are precomputed a :class:`simqso.sqgrids.IndexSampler`
    instance is used internally to map the sightlines to individual 
    spectra.
    '''
    name = 'igmlos'
    def __init__(self,forest,losMap=None,**kwargs):
        self.subsample = kwargs.pop('subsample',True)
        if isinstance(forest,Sampler):
            s = forest
            forest = None
        else:
            N = forest.numSightLines
            if losMap is None:
                if self.subsample:
                    s = RandomSubSampler(N)
                else:
                    s = FixedSampler(np.arange(N,dtype=np.int32))
            else:
                s = FixedSampler(losMap)
        super(SightlineVar,self).__init__(s,**kwargs)
        if forest is not None:
            self.set_forest_grid(forest)
        self.dtype = np.int32
    def set_forest_grid(self,forest):
        self.forest = forest
        self.varmeta = (self.forest.forestModel,self.forest.numSightLines,
                        dict(zmax=self.forest.zmax,seed=self.forest.seed,
                             subsample=self.forest.subsample))

class HIAbsorptionVar(SightlineVar,SpectralFeatureVar):
    '''
    Variable used to store IGM HI absorption spectra.

    Since the spectra are precomputed a :class:`simqso.sqgrids.IndexSampler`
    instance is used internally to map the forest sightlines to individual 
    spectra.
    '''
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
    def __init__(self,appMag,z,kcorr,cosmo,restWave=None,**kwargs):
        absMag = appMag - self.cosmo.distmod(z).value - kcorr(appMag,z)
        sampler = FixedSampler(absMag)
        super(AbsMagFromAppMagVar,self).__init__(sampler,restWave,**kwargs)

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
    def __init__(self,logBhMass,logEddRatio,restWave=None,**kwargs):
        eddLum = 1.26e38 * 10**logBhMass
        lum = 10**logEddRatio * eddLum
        BC1450 = 5.0 # rough value from Richards+06
        lnu1450 = lum / BC1450
        M1450 = magnitude_AB_from_L_nu(lnu1450/2e15)
        sampler = FixedSampler(M1450)
        super(AbsMagFromBHMassEddRatioVar,self).__init__(sampler,restWave,
                                                         **kwargs)

class TimeVar(QsoSimVar):
    '''
    Variable to associate instantaneous quasar spectrum with a epoch.
    '''
    name = 't'

class DrwTimeSeriesVar(MultiDimVar):
    r'''
    Variable that associates damped random walk (DRW) parameters
    :math:`\tau` and :math:`\sigma` with a quasar.
    '''
    name = 'drwTauSigma'

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
    seed : int
        Seed for RNG. Note the seed is only applied at initialization,
        individual variables can be seeded with QsoSimVar.__init___().
    '''
    def __init__(self,qsoVars=[],cosmo=None,units=None,seed=None):
        self.cosmo = cosmo
        self.units = units
        self.seed = seed
        self.qsoVars = qsoVars
        if len(qsoVars) > 0:
            self.varNames = [ v.name for v in qsoVars ]
        else:
            self.varNames = []
        self.photoMap = None
        if self.seed:
            np.random.seed(self.seed)
    def setCosmology(self,cosmodef):
        if type(cosmodef) is dict:
            self.cosmo = cosmology.FlatLambdaCDM(**cosmodef)
        elif isinstance(cosmodef,str):
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
            # only keep this in the grouped table
            del self.data['_ii']
        return data_grouped.groups
    def __getattr__(self,name):
        try:
            return self.data[name]
        except KeyError:
            raise AttributeError("no attribute "+name)
    def addVar(self,var,noVals=False):
        '''
        Add a variable to the simulation.
        '''
        self.qsoVars.append(var)
        self.varNames.append(var.name)
        if not noVals:
            vals = var(self.nObj)
            if vals is not None:
                self.data[var.name] = vals
    def addVars(self,newVars,noVals=False):
        '''
        Add a list of variables to the simulation.
        '''
        for var in newVars:
            self.addVar(var,noVals=noVals)
    def addData(self,data):
        self.data = hstack([self.data,data])
    def getVars(self,varType=QsoSimVar):
        '''
        Return all variables that are instances of varType.
        If varType is a string, return the variable with name varType.
        '''
        if isinstance(varType,str):
            return self.qsoVars[self.varNames.index(varType)]
        else:
            return [v for v in self.qsoVars if isinstance(v,varType)]
    def varIndex(self,varName):
        return self.varNames.index(varName)
    def resample(self):
        for var in self.qsoVars:
            if var.dependentVars is not None:
                var.resample(self.data[var.dependentVars])
                self.data[var.name] = var(self.nObj)
    def distMod(self,z):
        return self.cosmo.distmod(z).value
    def loadPhotoMap(self,photoSys):
        self.photoMap = sqphoto.load_photo_map(photoSys)
        self.photoBands = list(self.photoMap['bandpasses'].keys())
    def getPhotoCache(self,wave):
        if self.photoMap:
            return sqphoto.getPhotoCache(wave,self.photoMap)
        else:
            return None
    def getBandIndex(self,band):
        return next(j for j in range(len(self.photoBands))
                      if self.photoMap['filtName'][self.photoBands[j]]==band)
    def getObsBandIndex(self):
        return self.getBandIndex(self.getVars(AppMagVar)[0].obsBand)
    def read(self,gridFile,clean=False,extname=None):
        '''
        Read a simulation grid from a file.
        '''
        self.data = Table.read(gridFile,hdu=extname)
        if clean:
            # XXX it's hacky to be aware of these colnames here, but need to
            # know how to delete derived quantities that will be recomputed
            for k in ['obsFlux','obsMag','obsFluxErr','obsMagErr',
                      'synMag','synFlux']:
                try:
                    del self.data[k]
                except KeyError:
                    pass
        self.nObj = len(self.data)
        hdr = fits.getheader(gridFile,extname=extname)
        self.units = hdr['GRIDUNIT']
        self.gridShape = eval(hdr['GRIDDIM'])
        self.setCosmology(hdr['COSMO'])
        try:
            self.simPars = ast.literal_eval(hdr['SQPARAMS'])
        except:
            #print('WARNING: no params in header')
            pass
        for i,v in enumerate(range(hdr['NSIMVAR'])):
            cls = hdr['AX%dTYPE'%i]
            name = hdr['AX%dNAME'%i]
            smplr = hdr['AX%dSMPL'%i] 
            seed = hdr.get('AX%dSEED'%i)
            vargs = hdr.get('AX%dVARG'%i)
            varmeta = hdr.get('AX%dMETA'%i)
            try:
                initargs = []
                if name in self.data.colnames:
                    # treat all input variables as fixed, using the values
                    # stored in gridFile
                    initargs.append( 'FixedSampler(self.data[name])' )
                if vargs is not None:
                    initargs.append('%s' % vargs)
                if seed is not None:
                    initargs.append('seed=%s' % seed)
                if varmeta is not None:
                    initargs.append('meta=%s' % varmeta)
                initargs = cls+'('+','.join(initargs)+')'
                #print('[{}]: '.format(name),initargs)
                c = eval(initargs)
                if c.name != name:
                    c.name = name
                self.addVar(c,noVals=True)
            except AttributeError:
                print('WARNING: failed to restore %s' % cls)
        self._restore(hdr)
    def _restore(self,hdr):
        pass
    @staticmethod
    def cosmo_str(cosmodef):
        if isinstance(cosmodef,cosmology.FLRW):
            d = dict(name=cosmodef.name,H0=cosmodef.H0.value,
                     Om0=cosmodef.Om0)
            if cosmodef.Ob0:
                d['Ob0'] = cosmodef.Ob0
            cosmodef = d
        return str(cosmodef)
    def write(self,outFn=None,simPars=None,outputDir='.',
              extname=None,overwrite=False):
        '''
        Write a simulation grid to a FITS file as a binary table, storing 
        meta-data in the header.
        '''
        tab = self.data
        if simPars is not None:
            simPars = copy(simPars)
            simPars['Cosmology'] = self.cosmo_str(simPars['Cosmology'])
            if 'QLFmodel' in simPars['GridParams']:
                s = str(simPars['GridParams']['QLFmodel']).replace('\n',';')
                simPars['GridParams']['QLFmodel'] = s
            tab.meta['SQPARAMS'] = str(simPars)
            if outFn is None:
                outFn = simPars['FileName']
        tab.meta['COSMO'] = self.cosmo_str(self.cosmo)
        tab.meta['GRIDUNIT'] = self.units
        tab.meta['GRIDDIM'] = str(self.gridShape)
        tab.meta['RANDSEED'] = str(self.seed)
        if self.photoMap is not None:
            tab.meta['OBSBANDS'] = ','.join(self.photoMap['bandpasses'])
        for i,var in enumerate(self.qsoVars):
            var.updateMeta(tab.meta,'AX%d'%i)
        tab.meta['NSIMVAR'] = len(self.qsoVars)
        if outFn is None:
            outFn = 'qsosim.fits'
        if not outFn.endswith('.fits'):
            outFn += '.fits'
        outFn = os.path.join(outputDir,outFn)
        tabhdu = fits.table_to_hdu(tab)
        if extname is not None:
            tabhdu.name = extname
        if not os.path.isfile(outFn):
            tabhdu.writeto(outFn)
        else:
            hdus = fits.open(outFn,mode='update')
            if extname in hdus:
                if overwrite:
                    hdus[extname] = tabhdu
                else:
                    pass
            else:
                hdus.append(tabhdu)
            hdus.close()

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
    def __init__(self,*args,**kwargs):
        self.fixedVars = kwargs.pop('fixed_vars',[])
        super(QsoSimGrid,self).__init__(**kwargs)
        if len(args) > 0:
            gridVars,nBins,nPerBin = args
            s = []
            for n,v in zip(nBins,gridVars):
                if n is None:
                    n = len(v(None))
                s.append(n)
            self.gridShape = tuple(s) + (nPerBin,)
            self._init_grid(gridVars)
            self._init_grid_data(gridVars)
            self.addVars(gridVars,noVals=True)
    def _init_grid(self,gridVars):
        axes = []
        self.gridCenters = []
        for n,var in zip(self.gridShape[:-1],gridVars):
            if isinstance(var.sampler,UniformSampler):
                if var.name in self.fixedVars:
                    nax = n
                else:
                    nax = n+1
            elif isinstance(var.sampler,FixedSampler):
                if not var.name in self.fixedVars:
                    raise ValueError
                nax = n
            else:
                raise ValueError
            axis = var(nax)
            axes.append(axis)
            if var.name in self.fixedVars:
                self.gridCenters.append(axis[:n])
            else:
                self.gridCenters.append(axis[:n]+np.diff(axis)/2)
        self.gridEdges = np.meshgrid(*axes,indexing='ij')
        self.nGridDim = len(self.gridShape)-1
    def _init_grid_data(self,gridVars):
        data = {}
        for i,(v,g) in enumerate(zip(gridVars,self.gridEdges)):
            s = [ slice(0,n,1) for n in self.gridShape[:self.nGridDim] ]
            pts0 = g[s][...,np.newaxis] 
            if v.name in self.fixedVars:
                pts = np.tile(pts0,self.gridShape[-1])
            else:
                x = np.random.random(self.gridShape)
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
    def _restore(self,hdr):
        self._init_grid(self.qsoVars[:len(self.gridShape)-1])
    def __str__(self):
        s = "grid dimensions: "+str(self.gridShape)+"\n"
        s += str(self.gridEdges)+"\n"
        s += str(self.data)
        return s



def generateQlfPoints(qlf,mRange,zRange,kcorr,**kwargs):
    '''
    Generate a `QsoSimPoints` grid fed by `AppMagVar` and `RedshiftVar`
    instances which are sampled from an input luminosity function.

    Parameters
    ----------
    qlf : `lumfun.LuminosityFunction` instance
        Representation of QLF to sample from.
    mRange : sequence (m_min,m_max)
        Range of apparent magnitudes to sample within [e.g., (17,22)].
    zRange : sequence (z_min,z_max)
        Range of redshifts to sample within [e.g., (2.2,3.5)].
    kcorr : callable
        K-correcton defined by either an `sqbase.SimKCorr` object or an 
        equivalent callable that accepts as arguments (m,z,inverse=False) 
        where m is apparent (absolute) magnitude if inverse is False (True),
        and z is redshift.
    qlfseed : int
        Seed for RNG applied *before* sampling from QLF.
    gridseed : int
        Seed for RNG applied *after* sampling from QLF, but before
        sampling additional variables added to the grid.
    zin : sequence
        Optional input redshifts. If supplied only the apparent magnitudes 
        are sampled from the QLF.

    kwargs : dict
        Additional parameters passed to
        `LuminosityFunction.sample_from_fluxrange()`.
    '''
    qlfSeed = kwargs.pop('qlfseed',None)
    if qlfSeed:
        np.random.seed(qlfSeed)
    gridSeed = kwargs.pop('gridseed',None)
    M,m,z = qlf.sample_from_fluxrange(mRange,zRange,kcorr,**kwargs)
    M = AbsMagVar(FixedSampler(M),kcorr.restBand)
    m = AppMagVar(FixedSampler(m),kcorr.obsBand)
    z = RedshiftVar(FixedSampler(z))
    qlfGrid = QsoSimPoints([M,m,z],cosmo=qlf.cosmo,units='flux',
                           seed=gridSeed)
    return qlfGrid

def generateBEffEmissionLines(M1450,**kwargs):
    trendFn = kwargs.pop('EmissionLineTrendFilename','emlinetrends_v6')
    indy = kwargs.pop('EmLineIndependentScatter',False)
    noScatter = kwargs.pop('NoScatter',False)
    excludeLines = kwargs.pop('ExcludeLines',[])
    onlyLines = kwargs.pop('OnlyLines',None)
    minEw = kwargs.pop('minEw',None)
    seed = kwargs.pop('seed',None)
    verbose = kwargs.pop('verbose',0)
    if seed:
        np.random.seed(seed)
    M_i = M1450 - 1.486 + 0.596
    if verbose > 0:
        print('loading emission line template {}'.format(trendFn))
    lineCatalog = Table.read(os.path.join(datadir,trendFn+'.fits'))
    for line,scl in kwargs.get('scaleEWs',{}).items():
        try:
            i = np.where(lineCatalog['name']==line)[0][0]
        except IndexError:
            print('WARNING: {} not in line template'.format(line))
            continue
        lineCatalog['logEW'][i,:,1] += np.log10(scl)
    for line,scl in kwargs.get('scaleLogScatter',{}).items():
        try:
            i = np.where(lineCatalog['name']==line)[0][0]
        except IndexError:
            print('WARNING: {} not in line template'.format(line))
            continue
        logew = lineCatalog['logEW'][i,:,1]
        logew[1] = logew[0] - scl*(logew[0]-logew[1])
        logew[2] = logew[0] + scl*(logew[2]-logew[0])
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
    if minEw is not None:
        logEw = np.polyval(lineCatalog['logEW'][:,1].T,-25)
        useLines &= logEw > np.log10(minEw)
    #
    lineList = [ (BaldwinEffectSampler(l['wavelength'],M_i,x1),
                  BaldwinEffectSampler(l['logEW'],M_i,x2),
                  BaldwinEffectSampler(l['logWidth'],M_i,x3))
                 for l in lineCatalog[useLines] ]
    lines = BossDr9EmissionLineTemplateVar(lineList,
                                           lineCatalog['name'][useLines])
    return lines

def generateVdBCompositeEmLines(minEW=1.0,noFe=False,verbose=0):
    tmplfits = os.path.join(datadir,'simqso_templates.fits')
    all_lines = Table(fits.getdata(tmplfits,'VdB01CompEmLines'))
    # blended lines are repeated in the table
    l,li = np.unique(all_lines['OWave'],return_index=True)
    lines = all_lines[li]
    li = np.where(lines['EqWid'] > minEW)[0]
    lines = lines[li]
    #
    if noFe:
        isFe = lines['ID'].find('Fe') == 0
        lines = lines[~isFe]
    if verbose > 0:
        print('using the following lines from VdB template: ', end=' ')
        print(','.join(list(lines['ID'])))
    c = ConstSampler
    lineList = [ [c(l['OWave']),c(l['EqWid']),c(l['Width'])] for l in lines ]
    lines = GaussianEmissionLinesTemplateVar(lineList)
    lines.meta['LINEMODL'] = 'Fixed Vanden Berk et al. 2001 emission lines'
    return lines


class VW01FeTemplateGrid(object):
    def __init__(self,z,wave,fwhm=5000.,scales=None,useopt=True):
        self.fwhm = fwhm
        self.scales = scales
        self.useopt = useopt
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
        feTemplate = np.zeros_like(wave)
        if self.useopt:
            templnames = ['Fe_UVOPT_V01_T06_BR92','Fe2_UV191','Fe3_UV47']
        else:
            templnames = ['Fe_UVtemplt_B','Fe2_UV191','Fe3_UV47']
        with fits.open(os.path.join(datadir,'simqso_templates.fits')) as tmplfits:
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

