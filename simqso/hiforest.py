#!/usr/bin/env python

from __future__ import print_function

import os
from collections import OrderedDict,namedtuple
import multiprocessing
from functools import partial
import numpy as np
import scipy.stats as stats
import scipy.constants as const
from astropy.io import fits
from astropy.table import Table,vstack,hstack

from .sqbase import datadir,fixed_R_dispersion,resample

# shorthands
exp,sqrt,log = np.exp,np.sqrt,np.log
c_kms = const.c/1e3
sqrt_pi = sqrt(np.pi)
sigma_c = 6.33e-18 # cm^-2
fourpi = 4*np.pi

def _getlinelistdata():
    # Line list obtained from Prochaska's XIDL code
    # https://svn.ucolick.org/xidl/trunk/Spec/Lines/all_lin.fits
    linelist = fits.getdata(os.path.join(datadir,'all_lin.fits'))
    Hlines = np.array([i for i in range(linelist.size) 
                           if b'HI' in linelist.ION[i]])
    transitionParams = {}
    for n,idx in enumerate(Hlines[::-1],start=2):
        transitionParams[n] = (linelist.WREST[idx],
                               linelist.F[idx],
                               linelist.GAMMA[idx])
    return transitionParams

transitionParams = _getlinelistdata()

# default is to go up to 32->1
default_lymanseries_range = (2,33)

def generate_los(model,zmin,zmax):
    '''
    Given a model for the distribution of absorption systems, generate
    a random line-of-sight populated with absorbers.
    returns (z,logNHI,b) for each absorption system.
    '''
    abs_dtype = [('z',np.float32),('logNHI',np.float32),('b',np.float32)]
    absorbers = []
    for component,p in model.items():
        if zmin > p['zrange'][1] or zmax < p['zrange'][0]:
            # outside the redshift range of this forest component
            continue
        # parameters for the forest component (LLS, etc.) absorber distribution
        NHImin,NHImax = p['logNHrange']
        NHImin,NHImax = 10**NHImin,10**NHImax
        z1 = max(zmin,p['zrange'][0])
        z2 = min(zmax,p['zrange'][1])
        beta = p['beta'] 
        mbeta1 = -beta+1
        gamma1 = p['gamma'] + 1
        # expectation for the number of absorbers at this redshift
        #  (inverting n(z) = N0*(1+z)^gamma)
        N = (p['N0']/gamma1) * ( (1+z2)**gamma1 - (1+z1)**gamma1 )
        # sample from a Poisson distribution for <N>
        n = stats.poisson.rvs(N,size=1)[0]
        # invert the dN/dz CDF to get the sample redshifts
        x = np.random.random_sample(n)
        z = (1+z1)*((((1+z2)/(1+z1))**gamma1 - 1)*x + 1)**(1/gamma1) - 1
        # invert the NHI CDF to get the sample column densities
        x = np.random.random_sample(n)
        NHI = NHImin*(1 + x*((NHImax/NHImin)**mbeta1 - 1))**(1/mbeta1)
        #
        try: 
            # fixed b
            b = np.array([p['b']]*n,dtype=np.float32)
        except KeyError:
            # dn/db ~ b^-5 exp(-(b/bsig)^-4) (Hui & Rutledge 1999)
            bsig = p['bsig']
            bmin,bmax = p['brange']
            bexp = lambda b: exp(-(b/bsig)**-4)
            x = np.random.random_sample(n)
            b = bsig*(-np.log((bexp(bmax)-bexp(bmin))*x + bexp(bmin)))**(-1./4)
        #
        absorber = np.empty(n,dtype=abs_dtype)
        absorber['z'] = z
        absorber['logNHI'] = np.log10(NHI)
        absorber['b'] = b
        absorbers.append(absorber)
    absorbers = np.concatenate(absorbers)
    # return sorted by redshift
    return absorbers[absorbers['z'].argsort()]

def voigt(a,x):
    '''Tepper-Garcia 2006, footnote 4 (see erratum)'''
    x2 = x**2
    Q = 1.5/x2
    H0 = exp(-x2)
    return H0 - (a/sqrt_pi)/x2 * (H0*H0*(4*x2*x2 + 7*x2 + 4 + Q) - Q - 1)

def sum_of_voigts(wave,tau_lam,c_voigt,a,lambda_z,b,tauMin,tauMax):
    '''
    Given arrays of parameters, compute the summed optical depth
    spectrum of absorbers using Voigt profiles.
    Uses the Tepper-Garcia 2006 approximation for the Voigt function.
    '''
    umax = np.clip(sqrt(c_voigt * (a/sqrt_pi)/tauMin),5.0,np.inf)
    # ***assumes constant velocity bin spacings***
    dv = (wave[1]-wave[0])/(0.5*(wave[0]+wave[1])) * c_kms
    du = dv/b
    bnorm = b/c_kms
    npix = (umax/du).astype(np.int32)
    for i in range(len(a)):
        w0 = np.searchsorted(wave,lambda_z[i])
        i1 = max(0,w0-npix[i])
        i2 = min(len(wave),w0+npix[i])
        if np.all(tau_lam[i1:i2] > tauMax):
            continue
        # the clip is to prevent division by zero errors
        u = np.abs((wave[i1:i2]/lambda_z[i]-1)/bnorm[i]).clip(1e-5,np.inf)
        tau_lam[i1:i2] += c_voigt[i] * voigt(a[i],u)
    return tau_lam

# from http://stackoverflow.com/questions/42558/python-and-the-singleton-pattern
class Singleton:
    def __init__(self,decorated):
        self._decorated = decorated
    def Instance(self,*args,**kwargs):
        try:
            inst = self._instance
            #self._argcheck(*args)
        except AttributeError:
            self._instance = self._decorated(*args,**kwargs)
            inst = self._instance
        return inst
    def __call__(self):
        raise TypeError('Must be accessed through "Instance()".')
    def __instancecheck__(self,inst):
        return isinstance(inst,self._decorated)
    #def _argcheck(self,*args):
    #    raise NotImplementedError

@Singleton
class VoigtTable(object):
    '''
    Lookup table of Voigt profiles use to precompute low-density absorbers.
    '''
    def __init__(self,*args,**kwargs):
        self._init_table(*args,**kwargs)
    def _argcheck(self,*args):
        assert self.dv == args[0]
    def _init_table(self,*args,**kwargs):
        wave, = args
        # ***assumes constant velocity bin spacings***
        dv = (wave[1]-wave[0])/(0.5*(wave[0]+wave[1])) * c_kms
        self.wave0 = wave[0]
        self.npix = len(wave)
        self.dv = dv
        self.dv_c = dv/c_kms
        #
        na = kwargs.get('fastvoigt_na',20)
        loga_min = kwargs.get('fastvoigt_logamin',-8.5)
        loga_max = kwargs.get('fastvoigt_logamax',-3.0)
        gamma = kwargs.get('fastvoigt_gamma',1.5)
        nb = kwargs.get('fastvoigt_nb',20)
        u_range = kwargs.get('fastvoigt_urange',10)
        # define the bins in Voigt a parameter using exponential spacings
        alpha = (loga_max - loga_min) / na**gamma
        self.logabins = np.array([loga_max - alpha*n**gamma 
                                      for n in range(na)])
        # define the bins in b
        self.bbins = np.linspace(10.,100.,nb)
        # 
        self.xv = {}
        for j,b in enumerate(self.bbins):
            # offset slightly to avoid division by zero error
            self.xv[j] = np.arange(1e-5,u_range,dv/b)
        self.dx = np.array([len(self.xv[j])-1 for j in range(len(self.bbins))])
        self.voigt_tab = {}
        for i in range(na):
            self.voigt_tab[i] = {}
            for j in range(nb):
                vprof = voigt(10**self.logabins[i],self.xv[j])
                self.voigt_tab[i][j] = np.concatenate([vprof[::-1][1:],vprof])
    def sum_of_voigts(self,a,b,wave,c_voigt,tau_lam):
        ii = np.argmin(np.abs(np.log10(a)[:,np.newaxis] -
                       self.logabins[np.newaxis,:]),axis=1)
        jj = np.argmin(np.abs(b[:,np.newaxis]-self.bbins[np.newaxis,:]),axis=1)
        wc = np.round((np.log(wave) - np.log(self.wave0))/self.dv_c)
        wc = wc.astype(np.int32)
        dx = self.dx[jj]
        w1,w2 = wc-dx,wc+dx+1
        x1,x2 = np.zeros_like(dx),2*dx+1
        # off left edge of spectrum
        ll = np.where(w1<0)[0]
        x1[ll] = -w1[ll]
        w1[ll] = 0
        # off right edge of spectrum
        ll = np.where(w2>self.npix)[0]
        x2[ll] = self.npix - w1[ll]
        w2[ll] = self.npix
        # within the spectrum!
        ll = np.where(~((w2<0)|(w1>=self.npix)|(w2-w1<=0)))[0]
        # now loop over the absorbers and add the tabulated voigt profiles
        for i,j,k in zip(ii[ll],jj[ll],ll):
            tau_lam[w1[k]:w2[k]] += \
                              c_voigt[k] * self.voigt_tab[i][j][x1[k]:x2[k]]
        return tau_lam

def fast_sum_of_voigts(wave,tau_lam,c_voigt,a,lambda_z,b,
                       tauMin,tauMax,tauSplit):
    '''
    Given arrays of parameters, compute the summed optical depth
    spectrum of absorbers using Voigt profiles.
    Uses the Tepper-Garcia 2006 approximation for the Voigt function
    for large optical depth systems (defined by tauSplit), and
    a lookup table for low optical depth systems.
    '''
    voigttab = VoigtTable.Instance(wave)
    # split out strong absorbers and do full calc
    ii = np.where(c_voigt >= tauSplit)[0]
    tau_lam = sum_of_voigts(wave,tau_lam,
                            c_voigt[ii],a[ii],lambda_z[ii],b[ii],
                            tauMin,tauMax)
    ii = np.where(c_voigt < tauSplit)[0]
    tau_lam = voigttab.sum_of_voigts(a[ii],b[ii],lambda_z[ii],
                                     c_voigt[ii],tau_lam)
    return tau_lam

def sum_of_continuum_absorption(wave,tau_lam,NHI,z1,tauMin,tauMax):
    '''
    Compute the summed optical depth for Lyman continuum blanketing
    given a series of absorbers with column densities NHI and
    redshifts z1 (=1+z).
    '''
    tau_c_lim = sigma_c*NHI
    lambda_z_c = 912.*z1
    ii = np.where((lambda_z_c > wave[0]) & (tau_c_lim > tauMin))[0]
    # sort by decreasing column density to start with highest tau systems
    ii = ii[NHI[ii].argsort()[::-1]]
    # ending pixel (wavelength at onset of continuum absorption)
    i_end = np.searchsorted(wave,lambda_z_c[ii],side='right')
    # starting pixel - wavelength where tau drops below tauMin
    wave_start = (tauMin/tau_c_lim[ii])**0.333 * wave[i_end]
    i_start = np.searchsorted(wave,wave_start)
    # now do the sum
    for i,i1,i2 in zip(ii,i_start,i_end):
        # ... only if pixels aren't already saturated
        if np.any(tau_lam[i1:i2] < tauMax):
            l1l0 = wave[i1:i2]/lambda_z_c[i]
            tau_lam[i1:i2] += tau_c_lim[i]*l1l0*l1l0*l1l0
    return tau_lam

def calc_tau_lambda(wave,los,**kwargs):
    '''
    Compute the absorption spectrum, in units of optical depth, for
    a series of absorbers along a line-of-sight (los).
    '''
    lymanseries_range = kwargs.get('lymanseries_range',
                                   default_lymanseries_range)
    tauMax = kwargs.get('tauMax',15.0)
    tauMin = kwargs.get('tauMin',1e-5)
    tau_lam = kwargs.get('tauIn',np.zeros_like(wave))
    fast = kwargs.get('fast',True)
    tauSplit = kwargs.get('fast_tauSplit',1.0)
    # arrays of absorber properties
    NHI = 10**los['logNHI']
    z1 = 1 + los['z']
    b = los['b']
    # first apply continuum blanketing. the dense systems will saturate
    # a lot of the spectrum, obviating the need for calculations of
    # discrete transition profiles
    tau_lam = sum_of_continuum_absorption(wave,tau_lam,NHI,z1,tauMin,tauMax)
    # now loop over Lyman series transitions and add up Voigt profiles
    for transition in range(*lymanseries_range):
        # transition properties
        lambda0,F,Gamma = transitionParams[transition]
        # Doppler width
        nu_D = b / (lambda0*1e-13)
        # Voigt a parameter
        a = Gamma / (fourpi*nu_D)
        # wavelength of transition at absorber redshift
        lambda_z = lambda0*z1
        # coefficient of absorption strength (central tau)
        c_voigt = 0.014971475 * NHI * F / nu_D
        # all the values used to calculate tau, now just needs line profile
        if fast:
            tau_lam = fast_sum_of_voigts(wave,tau_lam,
                                         c_voigt,a,lambda_z,b,
                                         tauMin,tauMax,tauSplit)
        else:
            tau_lam = sum_of_voigts(wave,tau_lam,
                                    c_voigt,a,lambda_z,b,
                                    tauMin,tauMax)
    return tau_lam

class IGMTransmissionGrid(object):
    '''
    Generate a library of forest transmission spectra, by mapping an array 
    of emission redshifts to a set of sightlines.

    Parameters
    ----------
    wave : `~numpy.ndarray`
        Input wavelength grid (must be at fixed resolution!).
    z_em : `~numpy.ndarray`
        Array containing emission redshifts.
    nlos : int
        Number of lines-of-sight to generate.
    losMap : sequence 
        Optional mapping from z_em to LOS. Must have the same number of 
        elements and be in the range 0..nlos-1.
        If not provided and nlos>0, losMap is randomly generated.

    Returns
    -------
    spectra: dict
    T : `~numpy.ndarray` 
        transmission spectra with shape (N(z),N(wave))
    z : `~numpy.ndarray` 
        emission redshift for each spectrum
    losMap : `~numpy.ndarray` 
        map of z_em <-> line-of-sight
    wave : `~numpy.ndarray` 
        input wavelength grid
    voigtcache : bool
        use a lookup table of Voigt profiles to speed computation (def: True)
    '''
    def __init__(self,wave,forestModel,numSightLines,**kwargs):
        self.specWave = wave
        self.forestModel = forestModel
        self.numSightLines = numSightLines
        self.verbose = kwargs.get('verbose',0)
        self.nosortz = kwargs.get('nosortz',False)
        self.subsample = kwargs.get('subsample',True)
        self.seed = kwargs.get('seed')
        self.voigtkwargs = {'fast':kwargs.pop('voigtcache',True)}
        # pad the lower redshift by just a bit
        self.zmin = wave.min() / 1215.7 - 1.01
        self.zmax = kwargs.get('zmax',10)
        # Generate the lines-of-sight first, to preserve random generator order
        if self.verbose:
            print("Generating {} sightlines".format(self.numSightLines))
            if self.verbose > 1:
                print('... using random seed {}'.format(self.seed))
        if self.seed is not None:
            np.random.seed(self.seed)
        self.sightLines = [ generate_los(self.forestModel,self.zmin,self.zmax) 
                              for i in range(self.numSightLines) ]
        # default is 10 km/s
        forestRmin = kwargs.get('Rmin',3e4)
        logwave = log(wave)
        dloglam = np.diff(logwave)
        if not np.allclose(dloglam,dloglam[0]):
            raise ValueError("Must have constant dloglam")
        specR = dloglam[0]**-1
        self.nRebin = np.int(np.ceil(forestRmin/specR))
        self.forestR = specR * self.nRebin
        # go a half pixel below the minimum wavelength
        wavemin = exp(logwave[0]-0.5/specR)
        # go well beyond LyA to get maximum wavelength
        wavemax = 1250*(1+self.zmax)
        wavemax = min(wave[-1],1250*(1+self.zmax))
        self.nSpecPix = np.searchsorted(wave,wavemax,side='right')
        # now make sure it is an integer multiple 
        wavemax = wave[self.nSpecPix-1]
        self.nPix = self.nSpecPix * self.nRebin
        self.forestWave = exp( log(wavemin) + 
                                 self.forestR**-1*np.arange(self.nPix) )
        dloglam = self.forestR**-1
        self.forestWave = exp( log(wavemin) +  dloglam*np.arange(self.nPix) )
        #
        self.tau = np.zeros(self.nPix)
        if not self.subsample:
            self.allT = []
        self.reset()
    def reset(self):
        self.currentSightLineNum = -1
    def next_spec(self,sightLine,z,**kwargs):
        if self.currentSightLineNum != sightLine:
            if self.verbose > 1:
                print('finished sightline ',self.currentSightLineNum)
#            self.currentSightLine = generate_los(self.forestModel,
#                                                 self.zmin,self.zmax) 
            self.currentSightLine = self.sightLines[sightLine]
            self.currentSightLineNum = sightLine
            self.tau[:] = 0.0
            self.zi = 0
        zi1 = self.zi
        los = self.currentSightLine
        zi2 = np.searchsorted(los['z'],min(z,self.zmax))
        if self.verbose > 1:
            print("extending sightline {} to z={:.4f}".format(sightLine,z))
        if zi2 < zi1:
            raise ValueError("must generate sightline in increasing redshift")
        self.zi = zi2
        tau = calc_tau_lambda(self.forestWave,los[zi1:zi2],tauIn=self.tau,
                              **self.voigtkwargs)
        T = exp(-tau).reshape(-1,self.nRebin).mean(axis=1)
        self.T = T.astype(np.float32)
        if not self.subsample:
            self.allT.append(self.T)
        return self.T
    def current_spec(self,sightLine,z,**kwargs):
        if self.subsample:
            return self.T
        else:
            return self.allT[sightLine]
    def all_spec(self,losMap,z_em,**kwargs):
        if len(losMap) != len(z_em):
            raise ValueError
        if self.nosortz:
            zi = np.arange(len(z_em))
        else:
            zi = z_em.argsort()
        T = np.vstack( [ self.next_spec(losMap[i],z_em[i],**kwargs) 
                           for i in zi ] )
        return Table(dict(T=T[zi.argsort()].astype(np.float32),
                          z=z_em.astype(np.float32),
                          sightLine=losMap.astype(np.int32)))
    def write(self,fileName,outputDir,tspec=None,
              losMap=None,z_em=None,**kwargs):
        '''Save transmissionspectra to a FITS file.'''
        if tspec is None:
            if losMap is None or z_em is None:
                raise ValueError("Must pass losMap and z")
            tspec = self.all_spec(losMap,z_em,**kwargs)
        logwave = np.log(self.specWave[:2])
        dloglam = np.diff(logwave)
        tspec.meta['CD1_1'] = float(dloglam)
        tspec.meta['CRPIX1'] = 1
        tspec.meta['CRVAL1'] = logwave[0]
        tspec.meta['CRTYPE1'] = 'LOGWAVE'
        tspec.meta['IGMNLOS'] = self.numSightLines
        tspec.meta['IGMMODL'] = str(self.forestModel)
        tspec.meta['IGMRES'] = self.forestR
        for k,v in kwargs.get('meta',{}).items():
            tspec.meta[k] = v
        if not fileName.endswith('.fits') or fileName.endswith('.fits.gz'):
            fileName += '.fits'
        tspec.write(os.path.join(outputDir,fileName),overwrite=True)

# for now just duck-typing this
class CachedIGMTransmissionGrid(object):
    def __init__(self,fileName,outputDir='.'):
        if not (fileName.endswith('.fits') or fileName.endswith('.fits.gz')):
            fileName += '.fits'
        fn = os.path.join(outputDir,fileName)
        self.tspec = tspec = Table.read(fn)
        hdr = fits.getheader(fn,1)
        nwave = tspec['T'].shape[1]
        wi = np.arange(nwave)
        logwave = hdr['CRVAL1'] + hdr['CD1_1']*(wi-(hdr['CRPIX1']-1))
        self.specWave = exp(logwave)
        self.numSightLines = hdr['IGMNLOS']
        self.losIndex = { tuple(losNum_z):i for i,losNum_z 
                                       in enumerate(tspec['sightLine','z']) }
        self.losMap = self.tspec['sightLine']
    def next_spec(self,sightLine,z,**kwargs):
        return self.current_spec(sightLine,z,**kwargs)
    def current_spec(self,sightLine,z,**kwargs):
        # z is saved as float32 and need to match type
        i = self.losIndex[(sightLine,np.float32(z))]
        return self.tspec['T'][i]

def generate_binned_forest(fileName,forestModel,nlos,zbins,waverange,R,
                           outputDir='.',**kwargs):
    wave = fixed_R_dispersion(*tuple(waverange+(R,)))
    z = np.tile(zbins[:,np.newaxis],nlos).transpose()
    ii = np.arange(nlos)
    losMap = np.tile(ii[:,np.newaxis],len(zbins))
    fGrid = IGMTransmissionGrid(wave,forestModel,nlos,**kwargs)
    tspec = fGrid.all_spec(losMap.ravel(),z.ravel())
    if fileName is None:
        return tspec
    else:
        fGrid.write(fileName,outputDir,tspec=tspec,
                meta={'ZBINS':','.join(['%.3f'%_z for _z in zbins])})

def _get_forest_mags(forestModel,zbins,waverange,R,photoMap,n,**kwargs):
    wave = fixed_R_dispersion(*tuple(waverange+(R,)))
    grid = generate_binned_forest(None,forestModel,n,zbins,waverange,R,
                                  **kwargs)
    nBands = len(photoMap.getBandpasses())
    #
    fGrid = grid.group_by('sightLine')
    wi = np.arange(fGrid['T'].shape[-1],dtype=np.float32)
    fGrid['dmag'] = np.zeros((1,nBands),dtype=np.float32)
    fGrid['fratio'] = np.zeros((1,nBands),dtype=np.float32)
    #
    fakespec = namedtuple('fakespec','wave,f_lambda')
    refspec = fakespec(wave,np.ones_like(wave))
    refmags,reffluxes = photoMap.calcSynPhot(refspec)
    #
    for snum,sightLine in zip(fGrid.groups.keys['sightLine'],fGrid.groups):
        for i,z in enumerate(zbins):
            spec = fakespec(wave,sightLine['T'][i])
            mags,fluxes = photoMap.calcSynPhot(spec)
            dmag = mags - refmags
            dmag[fluxes<=0] = 99
            sightLine['dmag'][i] = dmag
            sightLine['fratio'][i] = fluxes.clip(0,np.inf) / reffluxes
        if ( (snum+1) % 10 ) == 0:
            try:
                pid = multiprocessing.current_process().name.split('-')[1]
            except:
                pid = '--'
            print('[%2s] completed %d sightlines' % (pid,snum+1))
    del fGrid['z','T']
    return fGrid

def generate_grid_forest(fileName,forestModel,nlos,zbins,waverange,R,
                         photoMap,outputDir='.',nproc=1,**kwargs):
    n = nlos // nproc
    if nproc == 1:
        _map = map
    else:
        pool = multiprocessing.Pool(nproc)
        _map = pool.map
    forest_generator = partial(_get_forest_mags,forestModel,zbins,
                               waverange,R,photoMap,**kwargs)
    _nlos = np.repeat(n,nproc)
    _nlos[-1] += nlos - np.sum(_nlos)
    fGrids = _map(forest_generator,_nlos)
    for i in range(1,len(fGrids)):
        fGrids[i]['sightLine'] += fGrids[i-1]['sightLine'].max() + 1
    fGrid = vstack(fGrids)
    fGrid.meta['ZBINS'] = ','.join(['%.3f'%_z for _z in zbins])
    fGrid.meta['BANDS'] = ','.join(photoMap.getBandpasses())
    fGrid.write(os.path.join(outputDir,fileName),overwrite=True)
    if nproc > 1:
        pool.close()

class GridForest(object):
    def __init__(self,fileName,simBands,median=False):
        self.simBands = np.array(simBands)
        self.data = Table.read(fileName).group_by('sightLine')
        self.numSightLines = len(self.data.groups)
        zbins = self.data.meta['ZBINS'].split(',')
        self.zbins = np.array(zbins).astype(np.float32)
        self.dz = np.diff(self.zbins)
        self.bands = np.array(self.data.meta['BANDS'].split(','))
        self.ii = np.array([ np.where(b==self.simBands)[0][0]
                                for b in self.bands ])
        shp = (self.numSightLines,len(self.zbins),-1)
        self.dmag = np.array(self.data['dmag']).reshape(shp)
        self.frat = np.array(self.data['fratio']).reshape(shp)
        if median:
            self.dmag = np.median(self.dmag,axis=0)[None,:,:]
            self.frat = np.median(self.frat,axis=0)[None,:,:]
            self.numSightLines = 1
        self.dmdz = np.diff(self.dmag,axis=1) / self.dz[:,None]
        self.dfdz = np.diff(self.frat,axis=1) / self.dz[:,None]
    def get(self,losNum,z):
        zi = np.digitize(z,self.zbins) - 1
        if np.any( (zi<0) | (zi >= len(self.zbins)-1) ):
            print("WARNING: qso z range {:.3f}|{:.3f} ".format(
                         z.min(),z.max()), end=' ')
            print("outside forest grid {:.3f}|{:.3f}".format(
                         self.zbins[0],self.zbins[-1]))
            zi = zi.clip(0,len(self.zbins)-2)
        dz = z - self.zbins[zi]
        dmag = self.dmag[losNum,zi] + self.dmdz[losNum,zi]*dz[:,None]
        frat = self.frat[losNum,zi] + self.dfdz[losNum,zi]*dz[:,None]
        return self.ii,dmag,frat

# for now just duck-typing this
class MeanIGMTransmissionGrid(object):
    def __init__(self,fileName,wave,outputDir='.'):
        if not fileName.endswith('.fits') or fileName.endswith('.fits.gz'):
            fileName += '.fits'
        self.outWave = wave
        fn = os.path.join(outputDir,fileName)
        tspec = Table.read(fn)
        hdr = fits.getheader(fn,1)
        nwave = tspec['T'].shape[1]
        wi = np.arange(nwave)
        logwave = hdr['CRVAL1'] + hdr['CD1_1']*(wi-(hdr['CRPIX1']-1))
        self.specWave = exp(logwave)
        self.wi = np.searchsorted(self.outWave,self.specWave[-1])
        nlos = hdr['IGMNLOS']
        self.numSightLines = 1
        self.zBins = np.array(list(map(float,hdr['ZBINS'].split(','))))
        self.meanT = tspec['T'].reshape(nlos,-1,nwave).mean(axis=0)
    def next_spec(self,sightLine,z,**kwargs):
        return self.spec(z)
    def current_spec(self,sightLine,z,**kwargs):
        return self.spec(z)
    def spec(self,z):
        zi = np.searchsorted(self.zBins,z)
        T = self.meanT[zi].clip(0,1) # XXX clip is there for bad vals
        T = resample(self.specWave,T,self.outWave[:self.wi])
        return T

