#!/usr/bin/env python

import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import simps
from scipy.signal import convolve
from scipy.stats import norm,lognorm,expon
from astropy.table import Table
from astropy import units as u

from .sqbase import datadir
from .spectrum import Spectrum


##############################################################################
# Samplers
##############################################################################

class Sampler(object):
	def __init__(self,low,high):
		self.low = low
		self.high = high
	def sample(self,n):
		raise NotImplementedError
	def resample(self,qsoData,**kwargs):
		pass
	def __call__(self,n):
		return self.sample(n)
	def __str__(self):
		s = str((self.low,self.high))
		return s

class FixedSampler(Sampler):
	def __init__(self,vals):
		self.low = None
		self.high = None
		self.vals = vals
	def sample(self,n):
		return self.vals

class NullSampler(Sampler):
	def __init__(self):
		pass
	def sample(self,n):
		return None

class ConstSampler(Sampler):
	def __init__(self,*val):
		self.low = None
		self.high = None
		self.val = val
	def sample(self,n):
		return np.repeat(self.val,n).reshape(-1,n)

class UniformSampler(Sampler):
	def sample(self,n):
		return np.linspace(self.low,self.high,n)

class GridSampler(Sampler):
	def __init__(self,low,high,nbins=None,stepsize=None):
		if nbins is None and stepsize is None:
			raise ValueError("Must specify nbins or stepsize")
		super(GridSampler,self).__init__(low,high)
		if stepsize:
			nbins = (high - low) / stepsize
		self.nbins = nbins+1
	def sample(self,n):
		arr = np.linspace(self.low,self.high,n*self.nbins)
		return arr.reshape(self.nbins,n)

class CdfSampler(Sampler):
	def _init_cdf(self):
		self.cdf_low = self.rv.cdf(self.low)
		self.cdf_high = self.rv.cdf(self.high)
	def _getpoints(self,x):
		return self.cdf_low + (self.cdf_high-self.cdf_low)*x
	def _sample(self,x):
		return self.rv.ppf(x)
	def sample(self,n):
		x = np.random.random(n)
		return self._sample(self._getpoints(x))

class PowerLawSampler(CdfSampler):
	def __init__(self,low,high,a):
		self.rv = self
		super(PowerLawSampler,self).__init__(low,high)
		self.a = a
	def cdf(self,x):
		x1,x2,a = self.low,self.high,self.a
		if np.any(x<x1) or np.any(x>x2):
			raise ValueError
		return (x**(a+1) - x1**(a+1)) / (a+1)
	def ppf(self,y):
		if np.any(y<0) or np.any(y>1):
			raise ValueError
		x1,x2,a = self.low,self.high,self.a
		return np.power( (x2**(a+1)-x1**(a+1))*y + x1**(a+1), (a+1)**-1 )

class GaussianSampler(CdfSampler):
	def __init__(self,mean,sigma,low=-np.inf,high=np.inf):
		super(GaussianSampler,self).__init__(low,high)
		self.mean = mean
		self.sigma = sigma
		self._reset()
		self._init_cdf()
	def _reset(self):
		self.rv = norm(loc=self.mean,scale=self.sigma)

class LogNormalSampler(CdfSampler):
	def __init__(self,mean,sigma,low=-np.inf,high=np.inf):
		super(LogNormalSampler,self).__init__(low,high)
		self.mean = mean
		self.sigma = sigma
		self.rv = lognorm(loc=self.mean,scale=self.sigma)
		self._init_cdf()

class ExponentialSampler(CdfSampler):
	def __init__(self,scale,low=0,high=np.inf):
		super(ExponentialSampler,self).__init__(low,high)
		self.scale = scale
		self._reset()
		self._init_cdf()
	def _reset(self):
		self.rv = expon(scale=self.scale)

class DoublePowerLawSampler(Sampler):
	def __init__(self,a,b,x0,low=-np.inf,high=np.inf):
		super(DoublePowerLawSampler,self).__init__(low,high)
		self.a = a
		self.b = b
		self.x0 = x0
	def sample(self,n):
		raise NotImplementedError

class LinearTrendWithAsymScatterSampler(Sampler):
	def __init__(self,coeffs,pts,low=-np.inf,high=np.inf):
		super(LinearTrendWithAsymScatterSampler,self).__init__(low,high)
		self.coeffs = coeffs
		self.npts = len(pts)
		self._reset(pts)
	def _reset(self,pts):
		xmn,xlo,xhi = [ np.polyval(c,pts) for c in self.coeffs ]
		siglo = np.clip(xmn-xlo,1e-10,np.inf)
		sighi = np.clip(xhi-xmn,1e-10,np.inf)
		self.loSampler = GaussianSampler(xmn,siglo,
		                                 low=self.low,high=self.high)
		self.hiSampler = GaussianSampler(xmn,sighi,
		                                 low=self.low,high=self.high)
	def _sample(self,x):
		# XXX this doesn't seem to pick correctly
		xlo = self.loSampler._sample(self.loSampler._getpoints(x))
		xhi = self.hiSampler._sample(self.hiSampler._getpoints(x))
		return np.choose(x<0.5,[xlo,xhi])

class BaldwinEffectSampler(LinearTrendWithAsymScatterSampler):
	def __init__(self,coeffs,absMag,x=None,low=-np.inf,high=np.inf):
		super(BaldwinEffectSampler,self).__init__(coeffs,absMag,
		                                          low=low,high=high)
		self.x = x
	def sample(self,n=None):
		if n is None:
			n = len(self.x)
		elif n != self.npts:
			raise ValueError("BaldwinEffectSampler input does not match "
			                 "preset (%d != %d)" % (n,self.npts))
		if self.x is None:
			# save the x values for reuse
			self.x = np.random.random(n)
		return self._sample(self.x)
	def resample(self,absMag,**kwargs):
		self._reset(absMag)
		return self._sample()



##############################################################################
# Simulation variables
##############################################################################

class QsoSimVar(object):
	def __init__(self,sampler,name=None):
		self.sampler = sampler
		if name is not None:
			self.name = name
	def __call__(self,n):
		return self.sampler(n)
	def __str__(self):
		return str(self.sampler)

class MultiDimVar(QsoSimVar):
	nDim = None
	def _recurse_pars(self,_pars,depth):
		if depth > 0:
			return [ self._recurse_pars(p,depth-1) for p in _pars ]
		else:
			return self.sampler(*_pars)
	def _recurse_call(self,samplers,n,depth):
		if depth > 0:
			return [ self._recurse_call(sampler,n,depth-1) 
			             for sampler in samplers ]
		else:
			return samplers(n)
	def _init_samplers(self,pars):
		self.samplers = self._recurse_pars(pars,self.nDim)
	def __call__(self,n):
		arr = self._recurse_call(self.samplers,n,self.nDim)
		arr = np.squeeze(arr)
		if n==1:
			# hmm...
			arr = arr[...,np.newaxis]
		return np.rollaxis(arr,-1)

class SpectralFeatureVar(object):
	def add_to_spec(self,spec,par):
		raise NotImplementedError

class AppMagVar(QsoSimVar):
	name = 'appMag'
	def __init__(self,sampler,band=None):
		super(AppMagVar,self).__init__(sampler)
		self.obsBand = band

class AbsMagVar(QsoSimVar):
	name = 'absMag'
	def __init__(self,sampler,restWave=None):
		'''if restWave is none then bolometric'''
		super(AbsMagVar,self).__init__(sampler)
		self.restWave = restWave

class RedshiftVar(QsoSimVar):
	name = 'z'

class ContinuumVar(QsoSimVar,SpectralFeatureVar):
	def add_to_spec(self,spec,par,**kwargs):
		spec.f_lambda[:] += self.render(spec.wave,spec.z,par,**kwargs)
		return spec

def _Mtoflam(lam0,M,z,DM):
	nu0 = (lam0 * u.Angstrom).to(u.Hz,equivalencies=u.spectral()).value
	fnu0 = 10**(-0.4*(M+DM(z)+48.599934))
	flam0 = nu0*fnu0/lam0
	return flam0/(1+z)

class BrokenPowerLawContinuumVar(ContinuumVar,MultiDimVar):
	nDim = 1
	name = 'slopes'
	def __init__(self,sampler,slopePars,breakPts):
		super(BrokenPowerLawContinuumVar,self).__init__(sampler)
		self.breakPts = np.asarray(breakPts).astype(np.float32)
		self._init_samplers(slopePars)
	def render(self,wave,z,slopes,fluxNorm=None):
		spec = np.zeros_like(wave)
		w1 = 1
		spec[0] = 1.0
		z1 = 1 + z
		alpha_lams = -(2+np.asarray(slopes)) # a_nu --> a_lam
		# add a breakpoint beyond the red edge of the spectrum in order
		# to fill using the last power law slope if necessary
		breakpts = np.concatenate([self.breakPts,[wave[-1]+1]])
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

class EmissionLineVar(QsoSimVar,SpectralFeatureVar):
	def add_to_spec(self,spec,par,**kwargs):
		spec.f_lambda[:] *= self.render(spec.wave,spec.z,par,**kwargs)
		return spec

class GaussianEmissionLineVar(EmissionLineVar):
	def __init__(self,linePars,name):
		super(GaussianEmissionLineVar,self).__init__(None,name=name)
	def render(self,wave,z,par):
		# XXX copied below
		emspec = np.ones_like(wave)
		lineWave,eqWidth,sigma = par * (1+z)
		A = eqWidth/(np.sqrt(2*np.pi)*sigma)
		twosig2 = 2*sigma**2
		nsig = (np.sqrt(-2*np.log(1e-3/A))*np.array([[-1.],[1]])).T
		i1,i2 = np.searchsorted(wave,lineWave+nsig*sigma)
		emspec[i1:i2] += A*np.exp(-(wave[i1:i2]-lineWave)**2 / twosig2)
		return emspec

class GaussianEmissionLinesTemplateVar(EmissionLineVar,MultiDimVar):
	nDim = 2
	name = 'emLines'
	def __init__(self,sampler,linePars):
		super(GaussianEmissionLinesTemplateVar,self).__init__(sampler)
		self._init_samplers(linePars)
	def render(self,wave,z,emlines):
		emspec = np.ones_like(wave)
		lineWave,eqWidth,sigma = emlines.T * (1+z)
		A = eqWidth/(np.sqrt(2*np.pi)*sigma)
		twosig2 = 2*sigma**2
		nsig = (np.sqrt(-2*np.log(1e-3/A))*np.array([[-1.],[1]])).T
		ii = np.where(np.logical_and(lineWave>wave[0],lineWave<wave[-1]))[0]
		for i in ii:
			i1,i2 = np.searchsorted(wave,lineWave[i]+nsig[i]*sigma[i])
			emspec[i1:i2] += A[i]*np.exp(-(wave[i1:i2]-lineWave[i])**2
			                                   / twosig2[i])
		return emspec

class BossDr9EmissionLineTemplateVar(GaussianEmissionLinesTemplateVar):
	'''translates the log values'''
	def __call__(self,n=None):
		lpar = super(BossDr9EmissionLineTemplateVar,self).__call__(n)
		lpar[...,1:] = np.power(10,lpar[...,1:])
		return lpar

class DustExtinctionVar(QsoSimVar,SpectralFeatureVar):
	pass

class SMCDustExtinctionVar(DustExtinctionVar):
	name = 'smcDustEBV'

class CalzettiDustExtinctionVar(DustExtinctionVar):
	name = 'calzettiDustEBV'

class BlackHoleMassVar(QsoSimVar):
	name = 'logBhMass'

class EddingtonRatioVar(QsoSimVar):
	name = 'logEddRatio'

class AbsMagFromAppMagVar(AbsMagVar):
	def __init__(self,appMag,m2M,restWave=None):
		absMag = m2M(appMag)
		sampler = FixedSampler(absMag)
		super(AbsMagFromAppMagVar,self).__init__(sampler,restWave)

class AbsMagFromBHMassEddRatioVar(AbsMagVar):
	def __init__(self,logBhMass,logEddRatio,restWave=None):
		eddLum = 1.26e38 * 10**logBhMass
		lum = 10**logEddRatio * eddLum
		BC1450 = 5.0 # rough value from Richards+06
		lnu1450 = lum / BC1450
		M1450 = magnitude_AB_from_L_nu(lnu1450/2e15)
		sampler = FixedSampler(M1450)
		super(AbsMagFromBHMassEddRatioVar,self).__init__(sampler,restWave)

class SynMagVar(QsoSimVar):
	name = 'synMag'

class SynFluxVar(QsoSimVar):
	name = 'synFlux'


##############################################################################
# Simulation grids
##############################################################################

class QsoSimObjects(object):
	def __init__(self,qsoVars=None,cosmo=None,units=None):
		self.qsoVars = qsoVars
		self.cosmo = cosmo
		self.units = units
	def setCosmology(self,cosmodef):
		if type(cosmodef) is dict:
			self.cosmo = cosmology.FlatLambdaCDM(**cosmodef)
		elif isinstance(cosmodef,cosmology.FLRW):
			self.cosmo = cosmodef
		elif cosmodef is None:
			self.cosmo = cosmology.get_current()
		else:
			raise ValueError
	def __iter__(self):
		for obj in self.data:
			yield obj
	def __getattr__(self,name):
		try:
			return self.data[name]
		except KeyError:
			raise AttributeError("no attribute "+name)
	def addVar(self,var):
		self.qsoVars.append(var)
		vals = var(self.nObj)
		if vals is not None:
			self.data[var.name] = vals
	def addVars(self,newVars):
		for var in newVars:
			self.addVar(var)
	def getSpectralFeatures(self):
		return [ var for var in self.qsoVars 
		               if isinstance(var,SpectralFeatureVar) ]
	def distMod(self,z):
		return self.cosmo.distmod(z).value
	def read(self,gridFile):
		self.data = Table.read(gridFile)
		self.nObj = len(self.data)

class QsoSimPoints(QsoSimObjects):
	def __init__(self,qsoVars,n=None,**kwargs):
		super(QsoSimPoints,self).__init__(qsoVars,**kwargs)
		data = { var.name:var(n) for var in qsoVars }
		self.data = Table(data)
		self.nObj = len(self.data)
	def __str__(self):
		return str(self.data)

class QsoSimGrid(QsoSimObjects):
	def __init__(self,qsoVars,nPerBin,**kwargs):
		super(QsoSimGrid,self).__init__(qsoVars,**kwargs)
		self.gridShape = tuple( v.sampler.nbins-1 for v in qsoVars ) + \
		                    (nPerBin,)
		# for grid variables this returns the edges of the bins
		axes = [ var(1) for var in qsoVars ]
		self.gridEdges = np.meshgrid(*axes,indexing='ij')
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
		return np.asarray(self.data[name]).reshape(self.gridShape)
	def __str__(self):
		s = "grid dimensions: "+str(self.gridShape)+"\n"
		s += str(self.gridEdges)+"\n"
		s += str(self.data)
		return s



def generateQlfPoints(qlf,mRange,zRange,m2M,cosmo,band,**kwargs):
	m,z = qlf.sample_from_fluxrange(mRange,zRange,m2M,cosmo,**kwargs)
	m = AppMagVar(FixedSampler(m),band=band)
	z = RedshiftVar(FixedSampler(z))
	return QsoSimPoints([m,z],units='flux',cosmo=cosmo)

def generateBEffEmissionLines(M1450,**kwargs):
	trendFn = kwargs.get('EmissionLineTrendFilename','emlinetrends_v6')
	indy = kwargs.get('EmLineIndependentScatter',False)
	noScatter = kwargs.get('NoScatter',False)
	excludeLines = kwargs.get('ExcludeLines',[])
	M_i = M1450 - 1.486 + 0.596
	lineCatalog = Table.read(datadir+trendFn+'.fits')
	for line,scl in kwargs.get('scaleEWs',{}).items():
		i = np.where(lineCatalog['name']==line)[0][0]
		lineCatalog['logEW'][i,:] += np.log10(scl)
	if noScatter:
		for k in ['wavelength','logEW','logWidth']:
			lineCatalog[k][:,1:] = lineCatalog[k][:,[0]]
	if indy:
		x1 = x2 = x3 = None
	else:
		x1 = np.random.random(len(M_i))
		x2 = np.random.random(len(M_i))
		x3 = np.random.random(len(M_i))
	lineList = [ ((l['wavelength'],M_i,x1),
	              (l['logEW'],M_i,x2),
	              (l['logWidth'],M_i,x3))
	             for l in lineCatalog 
	               if l['name'] not in excludeLines ]
	lines = BossDr9EmissionLineTemplateVar(BaldwinEffectSampler,lineList)
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
	lineList = [ [(l['OWave'],l['EqWid'],l['Width'])] for l in lines ]
	lines = GaussianEmissionLinesTemplateVar(ConstSampler,lineList)
	return lines


class VW01FeTemplateGrid(object):
	def __init__(self,z,wave,fwhm=5000.,scales=None):
		self.zbins = np.arange(z.min(),z.max()+0.005,0.005)
		self.feGrid = np.empty((self.zbins.shape[0],wave.shape[0]))
		# the Fe template is an equivalent width spectrum
		wave0,ew0 = self._restFrameFeTemplate(fwhm,scales)
		for i,zbin in enumerate(self.zbins):
			# in units of EW - no (1+z) when redshifting
			rfEW = interp1d(wave0*(1+zbin),ew0,kind='slinear',
			                bounds_error=False,fill_value=0.0)
			self.feGrid[i] = rfEW(wave)
		self.zi = np.searchsorted(self.zbins,z)
	def _loadVW01Fe(self,wave):
		fepath = datadir+'VW01_Fe/'
		feTemplate = np.zeros_like(wave)
		#for fn in ['Fe_UVtemplt_B.asc','Fe2_UV191.asc','Fe3_UV47.asc']:
		for fn in ['Fe_UVOPT_V01_T06_BR92.asc','Fe2_UV191.asc','Fe3_UV47.asc']:
			w,f = np.loadtxt(fepath+fn,unpack=True)
			spec = interp1d(w,f,kind='slinear')
			w1,w2 = np.searchsorted(wave,[w[0],w[-1]])
			feTemplate[w1:w2] += spec(wave[w1:w2])
		# continuum parameters given in VW01 pg. 6
		a_nu = -1.9
		fcont = 3.45e-14 * (wave/1500.)**(-2-a_nu)
		w1 = np.searchsorted(wave,1716.)
		a_nu = -1.0
		fcont[w1:] = 3.89e-14 * (wave[w1:]/1500.)**(-2-a_nu)
		feTemplate /= fcont
		return feTemplate
	def _restFrameFeTemplate(self,FWHM_kms,feScalings):
		#wave = np.logspace(np.log(1075.),np.log(3089.),5000,base=np.e)
		wave = np.logspace(np.log(1075.),np.log(7500.),5000,base=np.e)
		feTemplate = self._loadVW01Fe(wave)
		# rescale segments of the Fe template
		if feScalings is None:
			#feScalings = [(0,3500,1.0),]
			feScalings = [(0,7500,1.0),]
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
	def get(self,idx):
		return self.feGrid[self.zi[idx]]

