#!/usr/bin/env python

import os
import time
import numpy as np
from astropy.convolution import convolve,Gaussian1DKernel
from scipy.interpolate import interp1d,interp2d
from astropy import units as u

from pkg_resources import resource_filename
datadir = resource_filename('simqso', 'data')

def fixed_R_dispersion(lam1,lam2,R):
    '''Generate a wavelength grid at a fixed resolution d(log(lambda))^-1.

       Parameters
       ----------
       lam1,lam2 : float
           Wavelengths endpoints in Angstroms.
       R : float
           Spectral resolution.

       Returns
       -------
       wave : `~numpy.ndarray`
           Wavelength array between lam1 and lam2 with resolution R.

       Examples
       --------
       >>> from simqso.sqbase import fixed_R_dispersion
       >>> import numpy as np; np.set_printoptions(precision=2)
       >>> wave = fixed_R_dispersion(3000,3010,1000)
       >>> print wave
       [ 3000.    3003.    3006.01  3009.01  3012.02]
    '''
    loglam1 = np.log(lam1)
    loglam2 = np.log(lam2)
    dloglam = R**-1
    loglam = np.arange(loglam1,loglam2+dloglam,dloglam)
    return np.exp(loglam)

def deres(f,Rin,Rout,fout=None):
    '''
    Downgrade the resolution of a spectrum from Rin to Rout.

    Parameters
    ----------
    f : ndarray
        Input spectrum.
    Rin : float
        Resolution of input spectrum (as in R=lambda/dlambda).
    Rout : float
        Resolution of output spectrum, Rout<Rin.
    fout : ndarray
        Optional array for output spectrum. Updates f in-place if fout=f.
    '''
    assert Rout < Rin
    if fout is None:
        fout = np.zeros_like(f)
    kern = Gaussian1DKernel(Rin/Rout)
    fout[:] = convolve(f,kern,boundary='extend')
    return fout

def resample(x1,y1,x2):
    '''
    Resample function onto new grid using simple interpolation.
    '''
    resampfun = interp1d(x1,y1)
    return resampfun(x2)

def continuum_kcorr(obsBand,restBand,z,alpha_nu=-0.5):
    '''
    A simple power-law k-correction.

    Parameters
    ----------
    obsBand : str or float
        Observed band. Can be one of "SDSS-[ugriz]", "CFHT-[gri]", or a
        wavelength in Angstroms.
    restBand : str or float
        Rest-frame band. Can be one of "SDSS-[ugriz]", "CFHT-[gri]", or a
        wavelength in Angstroms.
    z : float or ndarray
        Emission redshift(s).
    alpha_nu : float
        Spectral index used to get k-correction, as f_nu ~ nu^alpha_nu.

    Returns
    -------
    k : ndarray
        K(z) is spectral k-correction for a simple power-law continuum.
    '''
    z = np.asarray(z)
    # CFHT: http://www.cfht.hawaii.edu/Science/mswg/filters.html
    effWave = {'SDSS-g':4670.,'SDSS-r':6165.,'SDSS-i':7471.,'SDSS-z':8918,
               'CFHT-g':4770.,'CFHT-r':6230.,'CFHT-i':7630.}
    try:
        obsWave = float(obsBand)
    except:
        obsWave = effWave[obsBand]
    try:
        restWave = float(restBand)
    except:
        restWave = effWave[restBand]
    # Following continuum K-corrections given in 
    #  Richards et al. 2006, AJ 131, 2766
    kcorr = ( -2.5*(1+alpha_nu)*np.log10(1+z) -
                 2.5*alpha_nu*np.log10(restWave/obsWave) )
    return kcorr

class SimKCorr(object):
    def __init__(self,obsBand,restBand):
        self.obsBand = obsBand
        self.restBand = restBand
    def __call__(self,m,z,inverse=False):
        raise NotImplementedError

class ContinuumKCorr(SimKCorr):
    def __init__(self,obsBand,restBand,alpha_nu=-0.5,effWaveBand=None):
        super(ContinuumKCorr,self).__init__(obsBand,restBand)
        self.alpha_nu = alpha_nu
        if effWaveBand is None:
            self._obsBand = self.obsBand
        else:
            self._obsBand = effWaveBand
    def __call__(self,m,z,inverse=False):
        return continuum_kcorr(self._obsBand,self.restBand,z,
                               alpha_nu=self.alpha_nu)

class EmissionLineKCorr(SimKCorr):
    def __init__(self,obsBand,restBand,datFile=None):
        super(EmissionLineKCorr,self).__init__(obsBand,restBand)
        if not obsBand.startswith('SDSS'):
            raise ValueError
        if datFile is None:
            datFile = os.path.join(datadir,'bossdr9kcorr_ugriz.npz')
        self.data = np.load(datFile)
        b_j = 'ugriz'.find(obsBand[-1])
        self.kcorr = self.data['kcorr'][...,b_j]
        self.mbins = self.data['mbins'][...,b_j]
        self.Mbins = self.data['Mbins']
        self.zbins = self.data['zbins']
        self.zz = np.tile(self.zbins,(len(self.Mbins),1))
        self.K_M = interp2d(self.Mbins,self.zbins,self.kcorr.transpose())
        self.K_m = interp2d(self.mbins,self.zz,self.kcorr)
    def __call__(self,m,z,inverse=False):
        Kfun = self.K_M if inverse else self.K_m
        if np.isscalar(m) or np.isscalar(z):
            return Kfun(m,z)
        return np.array([ Kfun(_m,_z)[0] for _m,_z in zip(m,z) ])
#        if inverse:
#            return griddata([self.Mbins,self.zbins],self.kcorr,(m,z))
#        else:

class AppToAbsMag(object):
    def __init__(self,cosmo,kcorr):
        self.cosmo = cosmo
        self.kcorr = kcorr
    def __call__(self,m,z,inverse=False):
        return self.cosmo.distmod(z).value + self.kcorr(m,z,inverse=inverse)

class Spectrum(object):
    '''
    Base class for one-dimensional spectra.

    Parameters
    ----------
    wave : `~numpy.ndarray`
        Input wavelength grid.
    f_lambda : `~numpy.ndarray`
        Input flux density. Default is None, which sets f_lambda[:] = 0.
    z : float
        Input redshift. Default is 0.0.
    '''
    def __init__(self,wave,f_lambda=None,z=0.0):
        self.wave = wave.astype(float) # XXX
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
        '''
        Set the redshift of the spectrum. *Does not modify the spectrum 
        itself*.
        '''
        self.z = z
    def clear(self):
        '''
        Clear the spectrum by zeroing the flux density vector and setting z=-1.
        '''
        self.z = -1.0
        self.f_lambda[:] = 0
    #
    def waveslice(self,w1,w2):
        '''
        Return a slice of the spectrum between two wavelengths.
        '''
        ii = np.where((self.wave > w1) & (self.wave < w2))[0]
        return Spectrum(self.wave[ii],self.f_lambda[ii],self.z)
    def resample(self,newWave):
        '''
        Resample the spectrum onto a new wavelength grid.
        '''
        newFlux = interp1d(self.wave,self.f_lambda,
                           bounds_error=False,fill_value=0.0)
        self.wave = newWave
        self.f_lambda = newFlux(newWave)
    def convolve_restframe(self,g,*args):
        r'''
        Convolves the spectrum with the input function as 
        f_lambda' = g(wave/(1+z),f_lambda).

        Optional \*args are passed to the convolution function.
        '''
        self.f_lambda = g(self.wave/(1+self.z),self.f_lambda,*args)

class TimerLog():
    '''
    Simple utility for tracking wall time execution for various steps of the 
    simulation.
    '''
    def __init__(self):
        self.stages = ['StartSimulation']
        self.times = [time.time()]
    def __call__(self,stage):
        self.stages.append(stage)
        self.times.append(time.time())
    def dump(self):
        self.__call__('Finish')
        stages = self.stages[1:]
        times = np.array(self.times[1:]) - self.times[0]
        #itimes = np.concatenate([[0,],np.diff(times)]) 
        itimes = np.diff(self.times)
        ftimes = itimes / times[-1]
        print('%20s %8s %8s %8s' % ('stage','time','elapsed','frac'))
        for t in zip(stages,itimes,times,ftimes):
            print('%20s %8.3f %8.3f %8.3f' % t)
        print()

