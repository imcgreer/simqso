#!/usr/bin/env python

import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import simps
from scipy.signal import convolve
from scipy.stats import norm,lognorm
from astropy.table import Table

# XXX
from astropy.io.fits import getdata
from astropy.io import ascii as ascii_io

from .sqbase import datadir,mag2lum
from . import dustextinction


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

class ConstSampler(Sampler):
	def __init__(self,val):
		self.low = None
		self.high = None
		self.val = val
	def sample(self,n):
		return np.repeat(self.val,n)

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
	def __init__(self,coeffs,absMag,low=0,high=np.inf):
		super(BaldwinEffectSampler,self).__init__(coeffs,absMag,
		                                          low=low,high=high)
	def sample(self,n):
		if n != self.npts:
			raise ValueError("BaldwinEffectSampler input does not match "
			                 "preset (%d != %d)" % (n,self.npts))
		# save the x values for reuse
		self.x = np.random.random(n)
		return self._sample(self.x)
	def resample(self,absMag,**kwargs):
		self._reset(absMag)
		return self._sample(self.x)



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

class AppMagVar(QsoSimVar):
	name = 'appMag'
	def __init__(self,sampler,band=None):
		super(AppMagVar,self).__init__(sampler)
		self.band = band

class AbsMagVar(QsoSimVar):
	name = 'absMag'
	def __init__(self,sampler,restWave=None):
		'''if restWave is none then bolometric'''
		super(AbsMagVar,self).__init__(sampler)
		self.restWave = restWave

class RedshiftVar(QsoSimVar):
	name = 'z'

class ContinuumVar(QsoSimVar):
	pass

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
		return np.rollaxis(np.array(arr),-1)

class BrokenPowerLawContinuumVar(ContinuumVar,MultiDimVar):
	nDim = 1
	def __init__(self,sampler,slopePars,name='slopes'):
		super(BrokenPowerLawContinuumVar,self).__init__(sampler,name=name)
		self._init_samplers(slopePars)

class EmissionLineVar(QsoSimVar):
	pass

class GaussianEmissionLineVar(EmissionLineVar):
	def __init__(self,linePars,name='emLines'):
		super(GaussianEmissionLineVar,self).__init__(None,name=name)

class GaussianEmissionLinesTemplateVar(EmissionLineVar,MultiDimVar):
	nDim = 2
	def __init__(self,sampler,linePars,name='emLines'):
		super(GaussianEmissionLinesTemplateVar,self).__init__(sampler,
		                                                      name=name)
		self._init_samplers(linePars)

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


##############################################################################
# Simulation grids
##############################################################################

class QsoSimObjects(object):
	def __init__(self,qsoVars=None):
		self.qsoVars = qsoVars
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
		self.data[var.name] = var(self.nObj)
	def addVars(self,newVars):
		for var in newVars:
			self.addVar(var)
	def getPars(self,obj,varType):
		pars = [ obj[var.name] 
		           for var in self.qsoVars
			         if isinstance(var,varType) ]
		return pars
	# XXX this really doesn't belong here...
	def updateMags(self,m):
		dm = m - self.appMag
		print '--> delta mag mean = %.7f, rms = %.7f, |max| = %.7f' % \
		              (dm.mean(),dm.std(),np.abs(dm).max())
		self.absMag[:] -= dm
		return np.abs(dm).max()
	def read(self,gridFile):
		self.data = Table.read(gridFile)
		self.nObj = len(self.data)

class QsoSimPoints(QsoSimObjects):
	def __init__(self,qsoVars,n=None):
		super(QsoSimPoints,self).__init__(qsoVars)
		data = { var.name:var(n) for var in qsoVars }
		self.data = Table(data)
		self.nObj = len(self.data)
	def __str__(self):
		return str(self.data)

class QsoSimGrid(QsoSimObjects):
	def __init__(self,qsoVars,nPerBin):
		super(QsoSimGrid,self).__init__(qsoVars)
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



def generateQlfPoints(qlf,mRange,zRange,m2M,cosmo,band='i',**kwargs):
	m,z = qlf.sample_from_fluxrange(mRange,zRange,m2M,cosmo,**kwargs)
	m = AppMagVar(FixedSampler(m),band=band)
	z = RedshiftVar(FixedSampler(z))
	return QsoSimPoints([m,z])

def generateBEffEmissionLines(absMag,**kwargs):
	trendFn = kwargs.get('EmissionLineTrendFilename','emlinetrends_v6')
	#fixed = kwargs.get('fixLineProfiles',False)
	#minEW = kwargs.get('minEW',0.0)
	#indy = kwargs.get('EmLineIndependentScatter',False)
	lineCatalog = Table.read(datadir+trendFn+'.fits')
	lineList = [ ((l['wavelength'],absMag),
	              (l['logEW'],absMag),
	              (l['logWidth'],absMag))
	             for l in lineCatalog ]
	lines = GaussianEmissionLinesTemplateVar(BaldwinEffectSampler,lineList)
	return lines




class FixedVdBcompositeEMLineGrid(object):
	def __init__(self,M,z,minEW=1.0,noFe=False):
		self.minEW = minEW
		self.all_lines = ascii_io.read(datadir+
		                            'VandenBerk2001_AJ122_549_table2.txt')
		# blended lines are repeated in the table.  
		l,li = np.unique(self.all_lines['OWave'],return_index=True)
		self.unique_lines = self.all_lines[li]
		li = np.where(self.unique_lines['EqWid'] > minEW)[0]
		self.lines = self.unique_lines[li]
		if noFe:
			isFe = self.lines['ID'].find('Fe') == 0
			self.lines = self.lines[~isFe]
		print 'using the following lines from VdB template: ',self.lines['ID']
	def update(self,M,z):
		# lines are fixed in luminosity and redshift
		return
	def addLine(self,name,rfwave,eqWidth,profileWidth):
		self.lines = np.resize(self.lines,(self.lines.size+1,))
		self.lines = self.lines.view(np.recarray)
		self.lines.OWave[-1] = rfwave
		self.lines.EqWid[-1] = eqWidth
		self.lines.Width[-1] = profileWidth
		self.lines.ID[-1] = name
	def addSBB(self):
		# XXX this is a hack!
		self.addLine('SBBhack',3000.,300.,500.)
	def _idmap(self,name):
		if name.startswith('Ly'):
			return 'Ly'+{'A':'{alpha}','B':'{beta}',
			             'D':'{delta}','E':'{epsilon}'}[name[-1]]
		else:
			return name
	def set(self,name,**kwargs):
		li = np.where(self._idmap(name) == self.lines.ID)[0]
		if len(li) == 0:
			print self.lines.ID
			raise ValueError
		for k,v in kwargs.items():
			if k == 'width':
				self.lines['Width'][li] = v
			elif k == 'eqWidth':
				self.lines['EqWid'][li] = v
			elif k == 'wave':
				self.lines['OWave'][li] = v
	def __iter__(self):
		'''returns wave, equivalent width, gaussian sigma'''
		while True:
			yield self.get(None)
	def get(self,idx):
		return self.lines['OWave'],self.lines['EqWid'],self.lines['Width']
	def getTable(self,hdr):
		'''Return a Table of all parameters and header information'''
		hdr['LINEMODL'] = 'Fixed Vanden Berk et al. 2001 emission lines'
		return None

class VW01FeTemplateGrid(object):
	def __init__(self,M,z,wave,fwhm=5000.,scales=None):
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
	def update(self,M,z):
		# template fixed in luminosity and redshift
		return
	def get(self,idx):
		return self.feGrid[self.zi[idx]]
	def getTable(self,hdr):
		'''Return a Table of all parameters and header information'''
		hdr['FETEMPL'] = 'Vestergaard & Wilkes 2001 Iron Emission'
		return None

class VariedEmissionLineGrid(object):
	def __init__(self,M1450,z,**kwargs):
		#trendfn = kwargs.get('EmissionLineTrendFilename','emlinetrends_v5',)
		trendfn = kwargs.get('EmissionLineTrendFilename','emlinetrends_v6',)
		self.fixed = kwargs.get('fixLineProfiles',False)
		self.minEW = kwargs.get('minEW',0.0)
		indy = kwargs.get('EmLineIndependentScatter',False)
		# emission line trends are calculated w.r.t. M_i, so
		#  convert M1450 to M_i(z=0) using Richards+06 eqns. 1 and 3
		M_i = M1450 - 1.486 + 0.596
		# load the emission line trend data, and restrict to lines for which
		# the mean EW trend at the lowest luminosity exceeds minEW
		t = getdata(datadir+trendfn+'.fits')
		maxEW = 10**(t['logEW'][:,0,1]+t['logEW'][:,0,0]*M_i.max())
		ii = np.where(maxEW > self.minEW)[0]
		self.lineTrends = t[ii]
		self.nLines = len(ii)
		# random deviate used to sample line profile values - if independent
		# scatter, each line gets its own random scatter, otherwise each 
		# object gets a single deviation and the lines are perfectly correlated
		nx = self.nLines if indy else 1
		xshape = z.shape + (nx,)
		self.xv = {}
		for k in ['wavelength','logEW','logWidth']:
			self.xv[k] = np.random.standard_normal(xshape)
		# store the line profile values in a structured array with each
		# element having the same shape as the input grid
		nf4 = str(z.shape)+'f4'
		self.lineGrids = np.zeros(self.nLines,
		                      dtype=[('name','S15'),('wavelength',nf4),
		                             ('eqWidth',nf4),('width',nf4)])
		self.lineGrids['name'] = self.lineTrends['name']
		self._edit_trends(**kwargs)
		self.update(M1450,z)
	def _edit_trends(self,**kwargs):
		# the optional argument 'scaleEWs' allows the caller to arbitrarily
		# rescale both the output equivalent widths and the input scatter
		# to the EW distribution for individual lines. construct the
		# scaling vector here
		self.sigscl = np.ones(self.nLines)
		self.ewscl = np.ones(self.nLines)
		for line,scl in kwargs.get('scaleEWs',{}).items():
			i = np.where(self.lineGrids['name']==line)[0][0]
			if type(scl) is tuple:
				self.ewscl[i],self.sigscl[i] = scl
			else:
				self.ewscl[i] = scl
	def update(self,M1450,z):
		# derive the line profile parameters using the input luminosities
		# and redshifts
		M_i = M1450 - 1.486 + 0.596
		M = M_i[...,np.newaxis]
		# loop over the three line profile parameters and obtain the
		# randomly sampled values using the input trends
		for k in ['wavelength','logEW','logWidth']:
			a = self.lineTrends[k][...,0]
			b = self.lineTrends[k][...,1]
			meanVal = a[...,0]*M + b[...,0]
			siglo = a[...,1]*M + b[...,1] - meanVal
			sighi = a[...,2]*M + b[...,2] - meanVal
			x = self.xv[k]
			sig = np.choose(x<0,[sighi,-siglo])
			if k=='logEW':
				sig *= self.sigscl
			v = meanVal + x*sig
			if k.startswith('log'):
				v[:] = 10**v
				k2 = {'logEW':'eqWidth','logWidth':'width'}[k]
			else:
				k2 = k
			if k=='logEW':
				v *= self.ewscl
			self.lineGrids[k2] = np.rollaxis(v,-1)
	def get(self,idx):
		# shape of lineGrids is (Nlines,)+gridshape, and idx is an index
		# into the grid, so add a dummy slice along the first axis
		idx2 = (np.s_[:],)+idx
		return (self.lineGrids['wavelength'][idx2],
		        self.lineGrids['eqWidth'][idx2],
		        self.lineGrids['width'][idx2])
	def getTable(self,hdr):
		'''Return a Table of all parameters and header information'''
		hdr['LINEMODL'] = 'Log-linear trends with luminosity'
		hdr['LINENAME'] = ','.join(self.lineGrids['name'])
		flatgrid = np.product(self.lineGrids['wavelength'].shape[1:])
		def squash(a):
			return a.reshape(-1,flatgrid).transpose()
		t = Table({'lineWave':squash(self.lineGrids['wavelength']),
		           'lineEW':squash(self.lineGrids['eqWidth']),
		           'lineWidth':squash(self.lineGrids['width']),})
		return t

class FixedDustGrid(object):
	def __init__(self,M,z,dustModel,E_BmV):
		self.dustModel = dustModel
		self.E_BmV = E_BmV
		self.dust_fn = dustextinction.dust_fn[dustModel]
	def update(self,M,z):
		# fixed in luminosity and redshift
		return
	def get(self,idx):
		return self.dust_fn,self.E_BmV
	def getTable(self,hdr):
		'''Return a Table of all parameters and header information'''
		hdr['DUSTMODL'] = self.dustModel
		hdr['FIXEDEBV'] = self.E_BmV
		return None

class ExponentialDustGrid(object):
	def __init__(self,M,z,dustModel,E_BmV_scale,fraction=1):
		self.dustModel = dustModel
		self.E_BmV_scale = E_BmV_scale
		self.dust_fn = dustextinction.dust_fn[dustModel]
		if fraction==1:
			self.EBVdist = np.random.exponential(E_BmV_scale,M.shape)
		else:
			print 'using dust LOS fraction ',fraction
			self.EBVdist = np.zeros_like(M).astype(np.float32)
			N = fraction * M.size
			ii = np.random.randint(0,M.size,(N,))
			self.EBVdist.flat[ii] = np.random.exponential(E_BmV_scale,(N,)).astype(np.float32)
	def update(self,M,z):
		# fixed in luminosity and redshift
		return
	def get(self,idx):
		return self.dust_fn,self.EBVdist[idx]
	def getTable(self,hdr):
		'''Return a Table of all parameters and header information'''
		t = Table({'E(B-V)':self.EBVdist.flatten()})
		hdr['DUSTMODL'] = self.dustModel
		hdr['EBVSCALE'] = self.E_BmV_scale
		return t

