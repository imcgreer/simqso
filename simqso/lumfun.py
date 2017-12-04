#!/usr/bin/env python

import copy
import itertools
from collections import OrderedDict
import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import quad,dblquad,romberg,simps
from scipy.ndimage.filters import convolve
from scipy import optimize
from scipy.special import hyp2f1
from scipy.stats import poisson
from astropy.stats import poisson_conf_interval
from astropy.table import Table
import astropy.units as u

from .sqbase import AppToAbsMag

skyDeg2 = 41253.

def interp_dVdzdO(zrange,cosmo):
    r'''
    Interpolate the differential comoving solid volume element 
    :math:`(dV/dz){d\Omega}`
    over zrange = :math:`(z_1,z_2)`. Much faster than full calculation 
    without significant loss in accuracy.

    Parameters
    ----------
    zrange : tuple
        Redshift range for interpolation.
    cosmo : astropy.cosmology.Cosmology
    '''
    zz = np.arange(zrange[0]-0.01,zrange[1]+0.0251,0.025)
    diff_co_vol = [cosmo.differential_comoving_volume(z).value for z in zz]
    return interp1d(zz,diff_co_vol)

def doublePL_Lintegral(x,a,b):
    '''
    Indefinite integral of a double power law function 
    :math:`f(x,a,b) = (x^a + x^b)^{-1}`.
    '''
    return (b*x**(1-a) / ((a-1)*(a-b)) - 
             x**(1-a)*((a-b)* 
              hyp2f1(1.,(a-1)/(a-b),(a-1)/(a-b)+1,-x**(b-a)) + b) / 
               ((a-1)*(a-b)))

# handy conversions are given in Hopkins, Richards, & Hernquist 2007
# eqns. 6-8
def integrateDPL(Mrange,logPhiStar,MStar,alpha,beta):
    r'''
    Integrate a double power law luminosity function of the form

    .. math::
        \Phi(M,z){dM}{dz} = \frac{\Phi^*{dM}{dz}}{10^{0.4(\alpha+1)(M-M^*)} 
                                  + 10^{0.4(\beta+1)(M-M^*)}}

    over the range Mrange = :math:`(M_\mathrm{min}, M_\mathrm{max})`.
    '''
    PhiStar_M = 10**logPhiStar
    MM = np.asarray(Mrange)
    L_min,L_max = 10**(-0.4*(MM-MStar))
    try:
        Lsum = doublePL_Lintegral(L_max,-alpha,-beta) - \
                doublePL_Lintegral(L_min,-alpha,-beta) 
    except:
        Lsum = np.inf # proceed to the failure case below
    if not np.isfinite(Lsum):
        # if the analytic function failed to evaluate, revert to a numerical
        # integration
        Lsum,err = quad(lambda x: 1/(x**-alpha + x**-beta), L_min, L_max)
    return 1.0857 * PhiStar_M * Lsum

class QlfEvolParam(object):
    '''
    A redshift-evolving parameter in a luminosity function.

    Parameters
    ----------
    par : sequence
        initial values
    fixed : bool or sequence
        False means none, True means all, otherwise same shape as x
    z0 : float
        evaluate at z-z0; i.e., z0=-1 means (1+z), z0=6 means (z-6.0)
    '''
    def __init__(self,par,fixed=False,z0=0.):
        # this assures even float values are converted to length-1 arrays
        par = np.asarray(par).astype(np.float64) * np.ones(1)
        self.par = np.ma.array(par,mask=fixed)
        self.z0 = z0
    @staticmethod
    def _tostr(x,m):
        s = '%8g' % x
        if m:
            s = '[%s]' % s
        return s
    def __str__(self):
        return ','.join([self._tostr(p,m) 
                     for p,m in zip(self.par.data,self.par.mask)])
    def set(self,val,i=None):
        if self.par.mask.all():
            return
        if i is None:
            i = np.where(~self.par.mask)[0]
        if np.isscalar(i):
            n = 1
            i = np.array([i])
        else:
            n = len(i)
        if type(val) is list:
            self.par.data[i] = [ val.pop(0) for j in range(n) ]
        else:
            self.par.data[i] = val
    def get(self):
        return self.par.compressed()
    def iterfree(self):
        ii = np.where(~self.par.mask)[0]
        for i in ii:
            yield i,self.par.data[i]
    def fix(self,i=None):
        if i is None:
            i = np.s_[:]
        self.par[i] = np.ma.masked
    def free(self,i=None):
        if i is None:
            i = np.s_[:]
        self.par[i] = self.par.data[i]
    def _extract_par(self,par):
        rv = self.par.data.copy()
        if par is not None:
            n = (~self.par.mask).sum()
            rv[~self.par.mask] = [ par.pop(0) for i in range(n) ]
        return rv
    def eval_at_z(self,z,par=None):
        raise NotImplementedError

class PolyEvolParam(QlfEvolParam):
    '''
    A luminosity function parameter that evolves with redshift according 
    to a polynomial function.
    '''
    def eval_at_z(self,z,par=None):
        par = self._extract_par(par)
        return np.polyval(par,z-self.z0)


class LuminosityFunction(object):
    def __init__(self,cosmo=None):
        self.cosmo = cosmo
        self.set_scale('log')
        self.paramBounds = {}
    def __str__(self):
        s = ''
        for pname,p in self.params.items():
            s += '%15s:  %s\n' % (pname,str(p))
        return s
    def set_cosmology(self,cosmo):
        self.cosmo = cosmo
    @staticmethod
    def _resolvepar(p):
        if isinstance(p,QlfEvolParam):
            return p
        else:
            return PolyEvolParam(p)
    def _iterpars(self):
        for p in self.params.values():
            yield p
    def getpar(self):
        return np.concatenate([ p.get() for p in self._iterpars() ])
    def setpar(self,par):
        par = list(par)
        for p in self._iterpars():
            p.set(par)
    def eval_par_at_z(self,z,par=None):
        return [ p.eval_at_z(z,par) for p in self._iterpars() ]
    def logPhi(self,M,z,*args):
        raise NotImplementedError
    def Phi(self,M,z,*args):
        return 10**self.logPhi(M,z,*args)
    def __call__(self,M,z,par=None):
        return self._call(M,z,par)
    def set_scale(self,scale):
        if scale not in ['log','linear']:
            raise ValueError
        self.scale = scale
        if scale == 'log':
            self._call = self.logPhi
        else:
            self._call = self.Phi
    def copy(self):
        return copy.deepcopy(self)
    def set_param_bounds(self,paramName,paramBounds):
        self.paramBounds[paramName] = paramBounds
    def get_param_bounds(self,paramName):
        return self.paramBounds.get(paramName)

class DoublePowerLawLF(LuminosityFunction):
    def __init__(self,logPhiStar=None,MStar=None,alpha=None,beta=None,
                 **kwargs):
        '''each param is either a QlfEvolParam, or values to initialize
           a PolyEvolParam, which is the default
        '''
        super(DoublePowerLawLF,self).__init__(**kwargs)
        self.params = OrderedDict()
        self.params['logPhiStar'] = self._resolvepar(logPhiStar)
        self.params['MStar'] = self._resolvepar(MStar)
        self.params['alpha'] = self._resolvepar(alpha)
        self.params['beta'] = self._resolvepar(beta)
    def logPhi(self,M,z,par=None):
        if par is not None:
            par = list(par)
        logPhiStar,Mstar,alpha,beta = self.eval_par_at_z(z,par)
        if par is not None and len(par) > 0:
            raise ValueError
        return logPhiStar - \
                np.log10(10**(0.4*(alpha+1)*(M-Mstar)) + \
                         10**(0.4*( beta+1)*(M-Mstar)))
    def _sample(self,Mrange,zrange,p,**kwargs):
        zin = kwargs.pop('zin',None)
        verbose = kwargs.pop('verbose',0)
        eps_M,eps_z = 5e-2,2e-2
        nM = int(-np.diff(Mrange(zrange)) / eps_M)
        nz = int(np.diff(zrange) / eps_z)
        if zin is None:
            # integrate across redshift to get the dN/dz distribution
            skyfrac = kwargs.get('skyArea',skyDeg2) / skyDeg2
            dVdzdO = interp_dVdzdO(zrange,self.cosmo)
            phi_z = lambda z: integrateDPL(Mrange(z),
                                *self.eval_par_at_z(z,p)) * \
                                    dVdzdO(z)
            zbins = np.linspace(zrange[0],zrange[1],nz)
            zsamp = [quad(phi_z,*zr)[0] for zr in zip(zbins[:-1],zbins[1:])]
            zsamp = [0,] + zsamp
            Ntot = np.sum(zsamp)
            zfun = interp1d(np.cumsum(zsamp)/Ntot,zbins)
            Ntot = np.int(np.round(Ntot * skyfrac * 4*np.pi))
            if verbose > 0:
                print('integration returned ',Ntot,' objects')
        else:
            # redshifts supplied by user
            zfun = lambda x: zin
            Ntot = len(zin)
        x = np.random.random(Ntot)
        y = np.random.random(Ntot)
        z = zfun(x)
        M = np.zeros_like(z)
        for i in range(Ntot):
            Mr = Mrange(z[i])
            Mbins = np.linspace(Mr[0],Mr[1],nM)
            _p = self.eval_par_at_z(z[i],p)
            Msamp = [integrateDPL(Mr,*_p) for Mr in zip(Mbins[:-1],Mbins[1:])]
            Msamp = [0.,] + Msamp
            N_M = np.sum(Msamp)
            Mfun = interp1d(np.cumsum(Msamp)/N_M,Mbins)
            M[i] = Mfun(y[i])
            if verbose > 1 and Ntot > 1e4 and ((i+1)%(Ntot//10))==0:
                print(i+1,' out of ',Ntot)
        return M,z
    def _fast_sample(self,Mrange,zrange,p,**kwargs):
        verbose = kwargs.pop('verbose',0)
        if verbose > 1:
            print('using fast sample for QLF')
        skyfrac = kwargs.get('skyArea',skyDeg2) / skyDeg2
        eps_M,eps_z = 0.05,0.10
        magLimPad = 0.2
        full_Mrange = Mrange(zrange)
        nM = int(-np.diff(full_Mrange) / eps_M)
        nz = int(np.diff(zrange) / eps_z)
        Medges = np.linspace(full_Mrange[0],full_Mrange[1],nM)
        zedges = np.linspace(zrange[0],zrange[1],nz)
        # XXX shouldn't assume evenly spaced bins here
        dM = -np.diff(Medges)[0]
        dz = np.diff(zedges)[0]
        Mbins = Medges[:-1] + np.diff(Medges)/2
        zbins = zedges[:-1] + np.diff(zedges)/2
        Mlim_z = np.array([ Mrange(z)[0] for z in zbins ])
        dVdzdO = self.cosmo.differential_comoving_volume(zbins).value
        V_ij = dVdzdO * dz * dM * skyfrac * 4*np.pi
        Mi,zj = np.meshgrid(Mbins,zbins,indexing='ij')
        Phi_ij = self.Phi(Mi,zj)
        N_ij = Phi_ij * V_ij
        N_ij = poisson.rvs(N_ij)
        N_ij[Mi>Mlim_z+magLimPad] = 0
        ij = np.where(N_ij > 0)
        Mz = [ ( np.repeat(M,n), np.repeat(z,n) )
                  for M,z,n in zip(Mi[ij],zj[ij],N_ij[ij]) ]
        M,z = np.hstack(Mz)
        M += dM * (np.random.rand(len(M)) - 0.5)
        z += dz * (np.random.rand(len(M)) - 0.5)
        if verbose > 1:
            print('to generate {} quasars'.format(len(M)))
        return M,z
    def sample_from_fluxrange(self,mrange,zrange,kcorr,p=None,**kwargs):
        fast = kwargs.pop('fast_sample',False)
        m2M = AppToAbsMag(self.cosmo,kcorr)
        _mrange = np.array(mrange[::-1])
        _Mrange = lambda z: _mrange - m2M(_mrange,z)
        if fast:
            M,z = self._fast_sample(_Mrange,zrange,p,**kwargs)
        else:
            M,z = self._sample(_Mrange,zrange,p,**kwargs)
        m = M + m2M(M,z,inverse=True)
        return M,m,z
    def sample_from_Lrange(self,Mrange,zrange,p=None,**kwargs):
        _Mrange = lambda z: Mrange
        return self._sample(_Mrange,zrange,p,**kwargs)
    def _get_Lcdf_fun(self,Mrange,z,p):
        nM = 30
        Mbins = np.linspace(Mrange[0],Mrange[1],nM)
        logPhiStar,MStar,alpha,beta = self.eval_par_at_z(z,p)
        Lbins = 10**(-0.4*(Mbins-MStar))
        Lcdf = doublePL_Lintegral(Lbins[1:],-alpha,-beta) - \
                doublePL_Lintegral(Lbins[0],-alpha,-beta) 
        Lcdf /= Lcdf[-1]
        Lcdf = np.concatenate([[0.,],Lcdf])
        return interp1d(Lcdf,Mbins)
    def sample_at_flux_intervals(self,mrange,zbins,Nintervals,nPerBin,
                                 kcorr,p=None):
        m2M = AppToAbsMag(self.cosmo,kcorr)
        _mrange = np.array(mrange[::-1])
        medges = np.empty((Nintervals+1,len(zbins)))
        mgrid = np.empty((Nintervals,len(zbins),nPerBin))
        xedges = np.linspace(0.,1,Nintervals+1)
        for j,z in enumerate(zbins):
            Mrange = _mrange - m2M(_mrange,z)
            Lcdf_fun = self._get_Lcdf_fun(Mrange,z,p)
            Mvals = Lcdf_fun(xedges)[::-1]
            medges[:,j] = Mvals + m2M(Mvals,z,inverse=True)
            for i in range(Nintervals):
                x = xedges[i] + np.diff(xedges)[i]*np.random.random(nPerBin)
                Mx = Lcdf_fun(x)
                mgrid[i,j,:] = Mx + m2M(Mx,z,inverse=True)
        return medges,mgrid
    def integrate(self,mrange,zrange,kcorr,p=None):
        m2M = AppToAbsMag(self.cosmo,kcorr)
        dVdzdO = interp_dVdzdO(zrange,self.cosmo)
        _mrange = np.array(mrange)
        Mrange = lambda z: _mrange - m2M(_mrange,z)
        phi_z = lambda z: integrateDPL(Mrange(z),*self.eval_par_at_z(z,p)) * \
                                dVdzdO(z)
        nqso,err = quad(phi_z,*zrange)
        nqso *= 4*np.pi
        return nqso
    def ionizing_emissivity(self,z,Mrange,p=None,**kwargs):
        logPhiStar,MStar,alpha,beta = self.eval_par_at_z(z,p)
        # phi(L) integral is over (L/L*)^-alpha
        # Lphi(L) integral is over (L/L*)^-alpha-1, so send (alpha+1)
        x = integrateDPL(Mrange,logPhiStar,MStar,alpha+1,beta+1)
        if True:
            # until astropy provides AB mag -> Lnu conversion
            c = 4.*np.pi*(10*u.pc.to(u.cm))**2
            LStar_nu = c * 10**(-0.4*(MStar + 48.6))
        if True:
            # until I get something more flexible in here
            # ... use the power-law conversion in .spectrum
            break_wave = kwargs.get('break_wave',1100.)
            alpha1 = kwargs.get('alpha1',-0.5)
            alpha2 = kwargs.get('alpha2',-1.5)
            #print 'warning: using SED model (%.2f,%.1f,%.2f)' % \
            #       (alpha2,break_wave,alpha1)
            l912 = (1450./break_wave)**alpha1 * (break_wave/912.)**alpha2
        # for now return e1450, e912
        return LStar_nu * x, LStar_nu * l912 * x


class SinglePowerLawLF(LuminosityFunction):
    def __init__(self,logPhiStar=None,alpha=None,**kwargs):
        super(SinglePowerLawLF,self).__init__(**kwargs)
        self.params = OrderedDict()
        self.params['logPhiStar'] = self._resolvepar(logPhiStar)
        self.params['alpha'] = self._resolvepar(alpha)
        self.Mref = -26.0
    def logPhi(self,M,z,par=None):
        if par is not None:
            par = list(par)
        logPhiStar,alpha = self.eval_par_at_z(z,par)
        if par is not None and len(par) > 0:
            raise ValueError
        return logPhiStar - \
                np.log10(10**(0.4*(alpha+1)*(M-self.Mref)))



class SchechterLF(LuminosityFunction):
    def __init__(self,logPhiStar=None,MStar=None,alpha=None,**kwargs):
        super(SchechterLF,self).__init__(**kwargs)
        self.params = OrderedDict()
        self.params['logPhiStar'] = self._resolvepar(logPhiStar)
        self.params['MStar'] = self._resolvepar(MStar)
        self.params['alpha'] = self._resolvepar(alpha)
    def logPhi(self,M,z,par=None):
        if par is not None:
            par = list(par)
        logPhiStar,Mstar,alpha = self.eval_par_at_z(z,par)
        if par is not None and len(par) > 0:
            raise ValueError
        # -0.0357 = log10( ln10/2.5 )
        return ( logPhiStar - 0.4*(alpha+1)*(M-Mstar) 
                     - 10**(-0.4*(M-Mstar))/np.log(10) - 0.0357 )


# Fitting routines, originally from 
#      https://github.com/imcgreer/QLFz4/blob/master/qlffit.py

def arr_between(a,b):
    return np.logical_and(a>=b[0],a<b[1])

class QuasarSurvey(object):
    r'''
    A collection of quasars formed with uniform selection criteria.

    Parameters
    ----------
    m : apparent magnitudes of objects
    z : redshifts
    m_lim : limiting apparent magnitude of survey
    skyArea : area of survey in deg^2
    m2M : f(m,z,inverse=False)
        function taking apparent mag and redshift as arguments, along with a 
        keyword "inverse", and returning the conversion from apparent mag to 
        absolute mag, or the reverse if inverse=True. Must include both 
        cosmological and k-corrections, i.e., 
        :math:`M = m - \mathrm{m2M}(m,z) = m - DM(z) - K(m,z)`
        and  :math:`m = M + \mathrm{m2M}(M,z,\mathrm{inverse=True})`
        Allows for luminosity-dependent k-corrections.
    '''
    def __init__(self,m,z,m_lim,skyArea,m2M):
        self.m = m
        self.z = z
        self.m_lim = m_lim
        self.skyArea = skyArea
        self.skyFraction = skyArea/41252.96
        self.area_srad =  self.skyFraction * 4*np.pi
        self.N = len(m)
        # convert apparent to absolute magnitudes
        self.m2M = m2M
        self.m2M_val = m2M(m,z)
        self.M = m - self.m2M_val
    def set_selection_function(self,selfun):
        self.selfun = selfun
        self.p_Mz = lambda M,z: self.selfun(M,z,absMag=True)
        self.weights = np.clip(self.selfun(self.m,self.z),1e-20,1.0)**-1
    def Nofz(self,zedges):
        N = np.empty(zedges.shape[0]-1)
        Ncorr = np.empty_like(N)
        for i,z1,z2 in zip(itertools.count(),zedges[:-1],zedges[1:]):
            ii = np.where(arr_between(self.z,(z1,z2)))[0]
            N[i] = len(ii)
            Ncorr[i] = np.sum(self.weights[ii])
        return N,Ncorr
    def take(self,ii):
        rv = copy.copy(self)
        for k in ['m','z','M','weights']:
            rv.__dict__[k] = rv.__dict__[k][ii]
        return rv
    def __getitem__(self,index):
        if type(index) is np.ndarray:
            return self.take(index)
        else:
            return (self.m[index],self.z[index],self.M[index])
    @staticmethod
    def init_lf_table(Mbins,zbins):
        lfShape = Mbins.shape + zbins.shape
        lfTab = Table(masked=True)
        lfTab['absMag'] = Mbins.astype(np.float32)
        lfTab['counts'] = np.zeros(lfShape,dtype=np.float32)
        lfTab['rawCounts'] = np.zeros(lfShape,dtype=np.int32)
        lfTab['countUnc'] = np.zeros(lfShape,dtype=np.float32)
        lfTab['filled'] = np.zeros(lfShape,dtype=np.bool)
        lfTab['phi'] = np.zeros(lfShape,dtype=np.float64)
        lfTab['rawPhi'] = np.zeros(lfShape,dtype=np.float64)
        lfTab['sigPhi'] = np.zeros(lfShape,dtype=np.float64)
        return lfTab
    def getinbounds(self,Medges,zedges):
        # identify which bins are within the flux limit by converting the
        # the luminosity bins to fluxes
        Mbounds,zbounds = np.meshgrid(Medges,zedges,indexing='ij')
        mbounds = Mbounds + self.m2M(Mbounds,zbounds,inverse=True)
        # if M is outside the definition of the k-correction, m2M returns
        # nan. This prevents a warning from the comparison to nan.
        mbounds[np.isnan(mbounds)] = np.inf
        inbounds = mbounds < self.m_lim
        # this sums the bin edges 2x2: 
        #   4=full covg, 0=no covg, otherwise partial
        inbounds = convolve(inbounds.astype(int),np.ones((2,2)))[:-1,:-1]
        return inbounds
    def calcBinnedLF(self,Medges,zedges,**kwargs):
        '''
        Calculate binned luminosity function from the stored survey data.

        Parameters
        ----------
        Medges : array defining bin edges in absolute mag
        zedges : array defining bin edges in redshift
        '''
        confinterval = kwargs.get('confinterval','root-n')
        # kind of hacky to access cosmo through m2M... XXX
        dVdzdO = interp_dVdzdO(zedges,self.m2M.cosmo)
        #
        Mbins = Medges[:-1] + np.diff(Medges)/2
        zbins = zedges[:-1] + np.diff(zedges)/2
        lfShape = Mbins.shape + zbins.shape
        # assign data points to bins and trim out-of-bounds objects
        Mi = np.digitize(self.M,Medges) - 1
        zi = np.digitize(self.z,zedges) - 1
        ii = np.where( (Mi>=0) & (Mi<len(Mbins)) &
                       (zi>=0) & (zi<len(zbins)) )[0]
        # do the counting in bins
        lf = self.init_lf_table(Mbins,zbins)
        np.add.at( lf['rawCounts'], (Mi[ii],zi[ii]),                1   )
        np.add.at(    lf['counts'], (Mi[ii],zi[ii]), self.weights[ii]   )
        np.add.at(  lf['countUnc'], (Mi[ii],zi[ii]), self.weights[ii]**2)
        #
        inbounds = self.getinbounds(Medges,zedges)
        lf['filled'][:] = (inbounds==4)
        # calculate bin volumes by integrating dVdM = (dV/dz)dzdM
        #   ... note if there were many redshift bins, could save time
        #       by only calculating dV once for each filled bin within
        #       each redshift slice
        binVol = np.zeros(lfShape)
        for i,j in zip(*np.where(inbounds > 0)):
            Mlim = lambda z: np.clip(self.m_lim-self.m2M(self.m_lim,z),
                                     Medges[i],Medges[i+1])
            binVol[i,j],_ = dblquad(lambda M,z: dVdzdO(z),
                                    zedges[j],zedges[j+1],
                                    lambda z: Medges[i],Mlim)
        # calculate luminosity function from ~ counts/volume
        mask = (lf['rawCounts']==0) | (binVol == 0)
        binVol = np.ma.array(binVol * self.area_srad, mask=mask)
        lf['phi'] = np.ma.divide(lf['counts'],binVol)
        lf['rawPhi'] = np.ma.divide(lf['rawCounts'],binVol)
        # --- only works for the symmetric ones ---
        sighi = ( poisson_conf_interval(lf['countUnc'],
                                            interval=confinterval)[1]
                           - lf['countUnc'] )
        lf['sigPhi'] = np.ma.divide(sighi,binVol)
        return lf

class QLFIntegrator(object):
    def __init__(self,Mrange,zrange,dVdzdO):
        self.Mrange = Mrange
        self.zrange = zrange
        self.dVdzdO = dVdzdO
        self.int_kwargs = {}

class FullQLFIntegrator(QLFIntegrator):
    def __init__(self,Mrange,zrange,dVdzdO,**kwargs):
        super(FullQLFIntegrator,self).__init__(Mrange,zrange,dVdzdO)
        self.nM = kwargs.pop('nM',20)
        self.nz = kwargs.pop('nz',10)
        self.int_kwargs.setdefault('epsabs',kwargs.pop('epsabs',1e-3))
        self.int_kwargs.setdefault('epsrel',kwargs.pop('epsrel',1e-3))
        self.zz = np.linspace(self.zrange[0],self.zrange[1],self.nz)
        self.MM = np.linspace(self.Mrange[0],self.Mrange[1],self.nM)
    def __call__(self,Phi_Mz,p_Mz,par):
        #
        integrand = lambda M,z: Phi_Mz(M,z,par) * p_Mz(M,z) * self.dVdzdO(z)
        lfsum = 0
        for z1,z2 in zip(self.zz[:-1],self.zz[1:]):
            for M1,M2 in zip(self.MM[:-1],self.MM[1:]):
                intp,err = dblquad(integrand, z1, z2,
                                   lambda z: M1,lambda z: M2,
                                   **self.int_kwargs)
                lfsum += intp
        return lfsum

class FastQLFIntegrator(QLFIntegrator):
    def __init__(self,Mrange,zrange,dVdzdO,**kwargs):
        super(FastQLFIntegrator,self).__init__(Mrange,zrange,dVdzdO)
        self.int_kwargs.setdefault('divmax',kwargs.pop('divmax',20))
        self.int_kwargs.setdefault('tol',kwargs.pop('epsabs',1e-3))
        self.int_kwargs.setdefault('rtol',kwargs.pop('epsrel',1e-3))
    def __call__(self,Phi_Mz,p_Mz,par):
        #
        integrand = lambda M,z: Phi_Mz(M,z,par) * p_Mz(M,z) * self.dVdzdO(z)
        inner = lambda z: romberg(integrand,*self.Mrange,args=(z,),
                                  **self.int_kwargs)
        outer = romberg(inner,*self.zrange,**self.int_kwargs)
        return outer

class FasterQLFIntegrator(QLFIntegrator):
    def __init__(self,Mrange,zrange,dVdzdO,**kwargs):
        super(FasterQLFIntegrator,self).__init__(Mrange,zrange,dVdzdO)
        self.minProb = kwargs.pop('minProb',1e-3)
        in_MBinW = kwargs.pop('MBinWidth',0.1)
        in_zBinW = kwargs.pop('zBinWidth',0.05)
        self.nM = int( np.diff(self.Mrange) / in_MBinW ) + 1 
        self.nz = int( np.diff(self.zrange) / in_zBinW ) + 1
        #
        self.Medges = np.linspace(self.Mrange[0],self.Mrange[1],self.nM)
        self.zedges = np.linspace(self.zrange[0],self.zrange[1],self.nz)
        self.MBinW = np.diff(self.Medges)[0]
        self.zBinW = np.diff(self.zedges)[0]
        #
        self.dV = self.dVdzdO(self.zedges)
        self.Mi,self.zi = np.meshgrid(self.Medges,self.zedges,indexing='ij')
        #
        self.p_Mz_cache = {}
        self.lowProbMask = {}
    def _get_p_Mz_grid(self,p_Mz):
        p_Mz_grid = self.p_Mz_cache.get(p_Mz)
        if p_Mz_grid is None:
            p_Mz_grid = p_Mz(self.Mi,self.zi)
            self.p_Mz_cache[p_Mz] = p_Mz_grid
            self.lowProbMask[p_Mz] = p_Mz_grid > self.minProb
        return p_Mz_grid,self.lowProbMask[p_Mz]
    def __call__(self,Phi_Mz,p_Mz,par):
        #
        p_Mz_grid,mask = self._get_p_Mz_grid(p_Mz)
        Phi_Mz_grid = Phi_Mz(self.Mi,self.zi,par)
        #
        lfsum_z = simps(Phi_Mz_grid * p_Mz_grid * self.dV, dx=self.zBinW)
        lfsum = simps(lfsum_z, dx=self.MBinW)
        return lfsum

def joint_qlf_likelihood_fun(par,surveys,lfintegrator,Phi_Mz,verbose):
    min_prob = 1e-3
    first_term,second_term = 0.0,0.0
    for s in surveys:
        # first term: sum over each observed quasar
        p_Mizi = s.weights**-1
        ii = np.where(p_Mizi > min_prob)[0]
        prod = p_Mizi[ii] * Phi_Mz(s.M[ii],s.z[ii],par) 
        first_term += -2*np.sum(np.log(prod))
        # second term: integral of LF over available volume
        lfsum = lfintegrator(Phi_Mz,s.p_Mz,par)
        second_term += 2 * s.area_srad * lfsum
    if verbose:
        print('testing ',par,first_term,second_term)
    return first_term + second_term

class FitMethod(object):
    def __init__(self):
        pass
    def __call__(self,*args,**kwargs):
        return self.routine(*args,**kwargs)
    def set_bounds(self,exclude_list=[]):
        pass

class NelderMeadFit(FitMethod):
    def __init__(self,verbose=False):
        self.routine = optimize.fmin
        self.args = ()
        self.kwargs = {'full_output':True,'xtol':1e-3,'ftol':1e-3,
                       'disp':verbose}

class JointQLFFitter(object):
    def __init__(self,Mrange,zrange,cosmo,qlfModel,**kwargs):
        self.likefun = joint_qlf_likelihood_fun
        self.Mrange = Mrange
        self.zrange = zrange
        self.dVdzdO = interp_dVdzdO(zrange,cosmo)
        self.qlfModel = qlfModel
        self.qlfModel.set_scale('linear')
        self.fitMethod = kwargs.get('fit_method',NelderMeadFit())
        self.set_integrate_mode(kwargs.get('integrate_mode','fast'),
                                kwargs.get('integrate_kwargs',{}))
        self.verbose = kwargs.get('verbose',False)
    def set_integrate_mode(self,mode,integrate_kwargs={}):
        self.integrate_mode = mode
        self.integrate_kwargs = integrate_kwargs
        if self.integrate_mode == 'full':
            self.lfintegrator = FullQLFIntegrator(self.Mrange,self.zrange,
                                                  self.dVdzdO,
                                                  **self.integrate_kwargs)
        elif self.integrate_mode == 'fast':
            self.lfintegrator = FastQLFIntegrator(self.Mrange,self.zrange,
                                                  self.dVdzdO,
                                                  **self.integrate_kwargs)
        elif self.integrate_mode == 'reallyfast':
            self.lfintegrator = FasterQLFIntegrator(self.Mrange,self.zrange,
                                                    self.dVdzdO,
                                                    **self.integrate_kwargs)
        else:
            raise ValueError
    def fit(self,surveys,qlfModel=None,initVals=None):
        if qlfModel is None:
            qlfModel = self.qlfModel
        if initVals is None:
            initVals = list(qlfModel.getpar())
        likefunArgs = (surveys,self.lfintegrator,qlfModel,self.verbose)
        res = self.fitMethod(self.likefun,initVals,*self.fitMethod.args,
                             args=likefunArgs,**self.fitMethod.kwargs)
        self.lastFit = res
        return res
    def getModel(self):
        rv = self.qlfModel.copy()
        rv.setpar(self.lastFit[0])
        return rv
    def getS(self,surveys,qlfModel=None,par=None):
        if qlfModel is None:
            qlfModel = self.qlfModel
        if par is None:
            par = qlfModel.getpar()
        likefunArgs = (surveys,self.lfintegrator,qlfModel,self.verbose)
        return self.likefun(par,*likefunArgs)
    def varyFitParam(self,paramName,surveys,ntry=None,logRange=None):
        if ntry is None:
            ntry = 50
        # XXX all of this is not right if these params have more than one
        #     free value
        if logRange is None:
            logRange = {
              'logPhiStar':(-1.5,0.0), 'MStar':(-1.5,0.0),
              'alpha':(-2.0,0.3), 'beta':(-2.0,1.0),
            }[paramName]
        logbins = logRange + (ntry,)
        #
        S0 = self.getS(surveys)
        print('S0 is ',S0,' at ',self.qlfModel.params[paramName].get())
        rv = {}
        #
        for i,pval0 in self.qlfModel.params[paramName].iterfree():
            fitvals = [(pval0,S0)]
            qlfModel = self.qlfModel.copy()
            print('trying %s[#%d]' % (paramName,i))
            for sgn in [-1,1]:
                delv = sgn*np.logspace(*logbins)
                for dv in delv:
                    qlfModel.params[paramName].set(pval0+dv,i=i)
                    qlfModel.params[paramName].fix(i)
                    S = self.fit(surveys,qlfModel=qlfModel)[1]
                    qlfModel.params[paramName].free(i)
                    if sgn < 0:
                        fitvals.insert(0, (pval0+dv, S) )
                    else:
                        fitvals.append(   (pval0+dv, S) )
                    print(' '.join(['%.3f']*6) % (pval0,S0,pval0+dv,S,dv,S-S0))
                    if S-S0 > 10:
                        # this is more than 3 sigma
                        break
            rv[i] = np.array(fitvals)
        return rv
    def sampleModels(self,sigParam,surveys,n=100):
        S0 = self.getS(surveys)
        par0 = self.qlfModel.getpar()
        qlfModel = self.qlfModel.copy()
        S = np.zeros(n)
        allpar = np.zeros((n,len(par0)))
        for i in range(n):
            par = par0 + sigParam*np.random.normal(size=len(sigParam))
            S[i] = self.getS(surveys,qlfModel,par)
            allpar[i] = par
        return Table(dict(par=allpar,dS=(S-S0)))


def deltaPhi(z,cosmo2,cosmo1):
    '''n/V' = (n/V)*(V/V')'''
    dV1 = cosmo1.differential_comoving_volume(z)
    dV2 = cosmo2.differential_comoving_volume(z)
    return dV2/dV1

def deltaLogPhi(z,cosmo2,cosmo1):
    '''log(n/V') = log[(n/V)*(V/V')] = log(n/V) + log(V/V')'''
    return np.log10(deltaPhi(z,cosmo2,cosmo1))

def deltaM(z,cosmo2,cosmo1):
    '''m = M + DM = M' + DM'
       --> M' = M + (DM - DM')'''
    DM1 = cosmo1.distmod(z).value
    DM2 = cosmo2.distmod(z).value
    return DM2 - DM1

