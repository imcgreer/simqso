#!/usr/bin/env python

import os
import numpy as np
import scipy.stats as stats
import scipy.constants as const
from astropy.io import fits

# shorthands
exp,sqrt = np.exp,np.sqrt
c_kms = const.c/1e3
sqrt_pi = sqrt(np.pi)
sigma_c = 6.33e-18 # cm^-2
fourpi = 4*np.pi

def _getlinelistdata():
	# Line list obtained from Prochaska's XIDL code
	# https://svn.ucolick.org/xidl/trunk/Spec/Lines/all_lin.fits
	datadir = os.path.split(__file__)[0]+'/data/'
	linelist = fits.getdata(datadir+'all_lin.fits')
	Hlines = np.array([i for i in range(linelist.size) 
	                       if 'HI' in linelist.ION[i]])
	LySeries = {}
	for n in range(Hlines.size):
		LySeries[n+2] = Hlines[-1-n]
	return linelist,LySeries

linelist,LymanSeries = _getlinelistdata()

# default is to go up to 32->1
default_lymanseries_range = (2,33)

Fan99_model = {
  'forest':{'zrange':(0.0,6.0),
            'logNHrange':(13.0,17.3),
            'N0':50.3,
            'gamma':2.3,
            'beta':1.41,
            'b':30.0},
     'LLS':{'zrange':(0.0,6.0),
            'logNHrange':(17.3,20.5),
            'N0':0.27,
            'gamma':1.55,
            'beta':1.25,
            'b':70.0},
     'DLA':{'zrange':(0.0,6.0),
            'logNHrange':(20.5,22.0),
            'N0':0.04,
            'gamma':1.3,
            'beta':1.48,
            'b':70.0},
}

WP11_model = {
 'forest0':{'zrange':(0.0,1.5),
            'logNHrange':(12.0,19.0),
            'gamma':0.2,
            'beta':1.55,
            'B':0.0170,
            'N0':340.,
            'brange':(10.,100.),
            'bsig':24.0},
 'forest1':{'zrange':(1.5,4.6),
            'logNHrange':(12.0,14.5),
            'gamma':2.04,
            'beta':1.50,
            'B':0.0062,
            'N0':102.0,
            'brange':(10.,100.),
            'bsig':24.0},
 'forest2':{'zrange':(1.5,4.6),
            'logNHrange':(14.5,17.5),
            'gamma':2.04,
            'beta':1.80,
            'B':0.0062,
            'N0':4.05,
            'brange':(10.,100.),
            'bsig':24.0},
 'forest3':{'zrange':(1.5,4.6),
            'logNHrange':(17.5,19.0),
            'gamma':2.04,
            'beta':0.90,
            'B':0.0062,
            'N0':0.051,
            'brange':(10.,100.),
            'bsig':24.0},
    'SLLS':{'zrange':(0.0,4.6),
            'logNHrange':(19.0,20.3),
            'N0':0.0660,
            'gamma':1.70,
            'beta':1.40,
            'brange':(10.,100.),
            'bsig':24.0},
     'DLA':{'zrange':(0.0,4.6),
            'logNHrange':(20.3,22.0),
            'N0':0.0440,
            'gamma':1.27,
            'beta':2.00,
            'brange':(10.,100.),
            'bsig':24.0},
}

forestModels = {'Fan1999':Fan99_model,
                'Worseck&Prochaska2011':WP11_model}

def generate_los(model,zmin,zmax):
	'''Given a model for the distribution of absorption systems, generate
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
	'''Given arrays of parameters, compute the summed optical depth
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
	#	raise NotImplementedError
@Singleton
class VoigtTable:
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
		# now loop over the absorbers and add the tabled voigt profiles
		for i,j,k in zip(ii[ll],jj[ll],ll):
			tau_lam[w1[k]:w2[k]] += \
			                  c_voigt[k] * self.voigt_tab[i][j][x1[k]:x2[k]]
		return tau_lam

def fast_sum_of_voigts(wave,tau_lam,c_voigt,a,lambda_z,b,
                       tauMin,tauMax,tauSplit):
	'''Given arrays of parameters, compute the summed optical depth
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
	'''Compute the summed optical depth for Lyman continuum blanketing
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
	'''Compute the absorption spectrum, in units of optical depth, for
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
		lambda0 = linelist.WREST[LymanSeries[transition]]
		F = linelist.F[LymanSeries[transition]]
		Gamma = linelist.GAMMA[LymanSeries[transition]]
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

def generate_spectra(wave,z_em,los,**kwargs):
	'''Generate a transmission spectrum along a line-of-sight (los)
	   given by a series of discrete absorbers. The returned spectra
	   have the same dispersion as the input (wave), but are generated
	   on a higher resolution grid (given by 'Rmin').
	   The spectra are calculated at discrete redshift intervals given
	   by z_em; i.e., the return value is a stack of transmission spectra
	   for a single line-of-sight, with each row corresponding to a
	   redshift in z_em.
	   Returns: array with shape (Nz,Nwave)
	'''
	# default is 10 km/s
	forestRmin = kwargs.get('Rmin',3e4)
	specR = (0.5*(wave[0]+wave[1]))/(wave[1]-wave[0])
	nrebin = np.int(np.ceil(forestRmin/specR))
	forestR = specR * nrebin
	# go a half pixel below the minimum wavelength
	wavemin = exp(np.log(wave[0])-0.5/specR)
	# go well beyond LyA to get maximum wavelength
	wavemax = min(wave[-1],1250*(1+z_em.max()))
	npix = np.searchsorted(wave,wavemax,side='right')
	fwave = exp(np.log(wavemin)+forestR**-1*np.arange(npix*nrebin))
	# only need absorbers up to the maximum redshift
	los = los[los['z']<z_em.max()]
	zi = np.concatenate([[0,],np.searchsorted(los['z'],z_em)])
	#
	tspec = np.ones(z_em.shape+wave.shape)
	#
	tau = np.zeros_like(fwave)
	for i in range(1,len(zi)):
		zi1,zi2 = zi[i-1],zi[i]
		tau = calc_tau_lambda(fwave,los[zi1:zi2],tauIn=tau,**kwargs)
		T = exp(-tau).reshape(-1,nrebin)
		tspec[i-1,:npix] = np.average(T,weights=fwave.reshape(-1,nrebin),
		                              axis=1)
	return tspec

def generate_N_spectra(wave,z_em,nlos,**kwargs):
	'''Generate a library of forest transmission spectra, randomly mapping 
	   an array of emission redshifts to a set of lines-of-sight.
	    wave - the input dispersion (forest is calculated on super-sampled grid)
	    z_em - the list of emission redshifts
	    nlos - the number of lines-of-sight to generate
	   If nlos == -1, each emission redshift has an independent line-of-sight
	   If losMap is provided in kwargs, this is the mapping from z_em to LOS,
	    i.e., losMap has the same number of entries as z_em, and has elements 
	    in the range 0..nlos-1
	   Otherwise, losMap is generated randomly.
	   Returns dictionary with
	    T = transmission array (Nz,Nwave)
	    losMap = line-of-sight to z_em mapping (Nz)
	    z = z_em
	    wave = wave
	'''
	forestModel = kwargs.get('ForestModel','Worseck&Prochaska2011')
	if type(forestModel) is str:
		forestModel = forestModels[forestModel]
	zrange = kwargs.get('zRange')
	if zrange is None:
		zmin,zmax = 0.0,z_em.max()
	else:
		zmin,zmax = zrange
	losMap = kwargs.get('losMap')
	if nlos == -1:
		# each emission redshift gets its own line-of-sight
		nlos = len(z_em)
		losMap = np.arange(nlos)
	# Generate the lines-of-sight first, to preserve random generator order
	linesofsight = [generate_los(forestModel,zmin,zmax) for i in range(nlos)]
	if losMap is None:
		# map each emission redshift to a randomly chosen line-of-sight
		losMap = np.random.randint(0,nlos,z_em.shape[0])
	# generate spectra for each line-of-sight
	specAll = np.zeros(z_em.shape+wave.shape)
	for losNum,los in enumerate(linesofsight):
		ii = np.where(losMap == losNum)[0]
		if len(ii)==0:
			continue
		zi = z_em[ii].argsort()
		spec = generate_spectra(wave,z_em[ii[zi]],los,**kwargs)
		specAll[ii,:] = spec[zi.argsort()]
		print 'finished LOS #%d' % (losNum+1)
	return dict(T=specAll,losMap=losMap,z=z_em.copy(),wave=wave.copy())

def generate_grid_spectra(wave,zbins,nlos,**kwargs):
	'''Generate spectra on a grid at discrete redshift samplings zbins.
	   nlos sets the number of times the grid is repeated along independent
	    lines-of-sight.
	   e.g., if zbins = [2.0,2.5,3.0] and nlos = 2, the returned spectra
	    include two lines-of-sight with samplings at z = 2, 2.5, and 3.
	'''
	zem = np.tile(zbins,nlos)
	losMap = np.repeat(np.arange(nlos),len(zbins))
	# This is needed for the absorber list to be reproducible.
	seed = kwargs.get('GridSeed',1)
	np.random.seed(seed)
	sp = generate_N_spectra(wave,zem,nlos,losMap=losMap,**kwargs)
	sp['nLOS'] = nlos
	sp['zbins'] = zbins
	sp['seed'] = seed
	return sp

def generate_spectra_from_grid(wave,z_em,tgrid,**kwargs):
	'''Given an input grid of transmission spectra, calculate output spectra
	   at redshift intervals given by z_em. This is used to 'extend' and grid
	   spectrum to align with new redshift samplings.
	   E.g., if tgrid has samples at z=[2.0,2.5.,3.0] and z_em=[2.2,2.7],
	    then the returned spectra are sampled at z=2.2 and z=2.7, where the
	    first spectrum is generated by taking the input grid spectrum at z=2.0
	    and extending it to z=2.2 by adding additional absorbers.
	   Similarly, the z=2.7 spectrum is generated from the z=2.5 grid spectrum.
	   This is useful for quickly generating forest spectra at arbitrary
	   redshifts without having to do the full calculation.
	'''
	forestModel = kwargs.get('ForestModel','Worseck&Prochaska2011')
	if type(forestModel) is str:
		forestModel = forestModels[forestModel]
	zrange = kwargs.get('zRange')
	if zrange is None:
		zmin,zmax = 0.0,tgrid['zbins'].max()
	else:
		zmin,zmax = zrange
	specAll = np.zeros(z_em.shape+wave.shape)
	# map each emission redshift to a line-of-sight, or a predefined
	#  mapping if provided
	nlos = tgrid['nLOS']
	losMap = kwargs.get('losMap',np.random.randint(0,nlos,z_em.shape[0]))
	T = tgrid['T'].reshape(nlos,len(tgrid['zbins']),-1)
	np.random.seed(tgrid['seed'])
	# generate spectra for each line-of-sight
	for losNum in range(nlos):
		ii = np.where(losMap == losNum)[0]
		los = generate_los(forestModel,zmin,zmax)
		if len(ii)==0:
			# need to do it here, because all los'es must be generated for
			# the random number generation to proceed in the correct order
			continue
		jj = np.digitize(z_em[ii],tgrid['zbins'])
		for j in range(len(tgrid['zbins'])):
			zi = np.where(jj-1==j)[0]
			if len(zi)==0:
				continue
			# generate spectra needs emission redshifts to be increasing
			zs = z_em[ii[zi]].argsort()
			# only use the absorbers starting at the redshift bin edge
			los_ii = np.where(los['z'] > tgrid['zbins'][j])[0]
			# add up the absorber spectra and then multiply them into the
			# spectrum for the redshift bin j
			spec = generate_spectra(wave,z_em[ii[zi[zs]]],los[los_ii],**kwargs)
			specAll[ii[zi],:] = T[losNum,j][np.newaxis,:] * spec[zs.argsort()]
	return dict(T=specAll,losMap=losMap,z=z_em.copy(),wave=wave.copy())

def save_spectra(spec,forestName):
	'''Save a spectrum to a FITS file.'''
	wave = spec['wave']
	npix = len(wave)
	nobj = len(spec['z'])
	spec_dtype = [('T','(%d,)f4'%npix),('z','f4'),('losMap','i4')]
	ftab = np.empty(nobj,dtype=spec_dtype)
	for k,fmt in spec_dtype:
		ftab[k] = spec[k]
	logwave = np.log(wave[:2])
	# XXX should figure out the standard way
	hdu = fits.new_table(ftab)
	hdu.header.update('CD1_1',np.diff(logwave)[0])
	hdu.header.update('CRPIX1',1)
	hdu.header.update('CRVAL1',logwave[0])
	hdu.header.update('CRTYPE1','LOGWAVE')
	if 'zbins' in spec:
		hdu.header.update('NLOS',spec['nLOS'])
		hdu.header.update('ZBINS',','.join('%.3f'%z for z in spec['zbins']))
		hdu.header.update('GRIDSEED',spec['seed'])
	hdu.writeto(forestName+'.fits.gz',clobber=True)

def load_spectra(forestName):
	'''Load a spectrum from a FITS file.'''
	spec,hdr = fits.getdata(forestName+'.fits.gz',header=True)
	nwave = spec['T'].shape[1]
	wave = np.arange(nwave)
	logwave = hdr['CRVAL1'] + hdr['CD1_1']*(wave-(hdr['CRPIX1']-1))
	wave = exp(logwave)
	rv = dict(T=spec['T'],losMap=spec['losMap'],z=spec['z'],wave=wave)
	if 'ZBINS' in hdr:
		rv['nLOS'] = hdr['NLOS']
		rv['zbins'] = np.array(hdr['ZBINS'].split(',')).astype(np.float)
		rv['seed'] = hdr['GRIDSEED']
	return rv

