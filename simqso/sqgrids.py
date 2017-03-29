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

from .sqbase import datadir
from .spectrum import Spectrum
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
	def resample(self,*args,**kwargs):
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

class IndexSampler(Sampler):
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
			nbins = int( np.floor((high - low) / stepsize) )
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
		xlo = self.loSampler._sample(self.loSampler._getpoints(x))
		xhi = self.hiSampler._sample(self.hiSampler._getpoints(x))
		return np.clip(np.choose(x>0.5,[xlo,xhi]),0,np.inf)

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
	def resample(self,qsoGrid,**kwargs):
		self._reset(qsoGrid.absMag)



##############################################################################
# Simulation variables
##############################################################################

class QsoSimVar(object):
	def __init__(self,sampler,name=None):
		self.sampler = sampler
		if name is not None:
			self.name = name
		self.update = False
		self.meta = {}
	def __call__(self,n):
		return self.sampler(n)
	def resample(self,*args,**kwargs):
		self.sampler.resample(*args,**kwargs)
	def __str__(self):
		return str(self.sampler)
	def updateMeta(self,meta):
		for k,v in self.meta.items():
			meta[k] = v

class MultiDimVar(QsoSimVar):
	# obviously these should be combined...
	def _recurse_call(self,samplers,n):
		if isinstance(samplers,Sampler):
			return samplers(n)
		else:
			return [ self._recurse_call(sampler,n) for sampler in samplers ]
	def _recurse_resample(self,samplers,*args,**kwargs):
		if isinstance(samplers,Sampler):
			samplers.resample(*args,**kwargs)
		else:
			for sampler in samplers:
				self._recurse_resample(sampler,*args,**kwargs)
	def __call__(self,n):
		arr = self._recurse_call(self.sampler,n)
		return np.rollaxis(np.array(arr),-1)
	def resample(self,*args,**kwargs):
		self._recurse_resample(self.sampler,*args,**kwargs)

class SpectralFeatureVar(object):
	def add_to_spec(self,spec,par,**kwargs):
		spec.f_lambda[:] += self.render(spec.wave,spec.z,par,**kwargs)
		return spec

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
	pass

def _Mtoflam(lam0,M,z,DM):
	nu0 = (lam0 * u.Angstrom).to(u.Hz,equivalencies=u.spectral()).value
	fnu0 = 10**(-0.4*(M+DM(z)+48.599934))
	flam0 = nu0*fnu0/lam0
	return flam0/(1+z)

class BrokenPowerLawContinuumVar(ContinuumVar,MultiDimVar):
	name = 'slopes'
	def __init__(self,samplers,breakPts):
		super(BrokenPowerLawContinuumVar,self).__init__(samplers)
		self.breakPts = np.asarray(breakPts).astype(np.float32)
		self.meta['CNTBKPTS'] = ','.join(['%.1f' % b for b in self.breakPts])
	def render(self,wave,z,slopes,fluxNorm=None):
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
	def render(self,wave,z,par):
		return render_gaussians(wave,z,np.array([par]))

class GaussianLineEqWidthVar(EmissionFeatureVar):
	'''this is an arguably kludgy way of making it possible to include
	   line EW as a variable in grids, by reducing the line to a single
	    parameter'''
	def __init__(self,sampler,name,wave0,width0):
		super(GaussianLineEqWidthVar,self).__init__(sampler,name)
		self.wave0 = wave0
		self.width0 = width0
	def render(self,wave,z,ew0):
		return render_gaussians(wave,z,
		                        np.array([[self.wave0,ew0,self.width0]]))

class GaussianEmissionLinesTemplateVar(EmissionFeatureVar,MultiDimVar):
	name = 'emLines'
	def render(self,wave,z,lines):
		return render_gaussians(wave,z,lines)

class BossDr9EmissionLineTemplateVar(GaussianEmissionLinesTemplateVar):
	'''translates the log values'''
	def __init__(self,samplers,lineNames):
		super(BossDr9EmissionLineTemplateVar,self).__init__(samplers)
		self.lineNames = lineNames
		self.meta['LINEMODL'] = 'BOSS DR9 Log-linear trends with luminosity'
		self.meta['LINENAME'] = ','.join(lineNames)
	def __call__(self,n=None):
		lpar = super(BossDr9EmissionLineTemplateVar,self).__call__(n)
		lpar[...,1:] = np.power(10,lpar[...,1:])
		return lpar

class FeTemplateVar(EmissionFeatureVar):
	def __init__(self,feGrid,name=None):
		super(FeTemplateVar,self).__init__(NullSampler())
		self.feGrid = feGrid
	def render(self,wave,z,par):
		return self.feGrid.get(z)

class HIAbsorptionVar(QsoSimVar,SpectralFeatureVar):
	def __init__(self,forest,name=None):
		super(HIAbsorptionVar,self).__init__(IndexSampler())
		self.forest = forest
		self.nforest = len(forest['wave'])
	def add_to_spec(self,spec,i,**kwargs):
		spec.f_lambda[:self.nforest] *= self.forest['T'][i]
		return spec

class DustExtinctionVar(QsoSimVar,SpectralFeatureVar):
	@staticmethod
	def dustCurve(name):
		return dustextinction.dust_fn[name]
	def add_to_spec(self,spec,ebv,**kwargs):
		spec.convolve_restframe(self.dustCurve(self.dustCurveName),ebv)
		return spec

class SMCDustVar(DustExtinctionVar):
	name = 'smcDustEBV'
	dustCurveName = 'SMC'
	meta = {'DUSTMODL':'SMC'}

class CalzettiDustVar(DustExtinctionVar):
	name = 'calzettiDustEBV'
	dustCurveName = 'CalzettiSB'
	meta = {'DUSTMODL':'Calzetti Starburst'}

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
	def addData(self,data):
		self.data = hstack([self.data,data])
	def getVars(self,varType=QsoSimVar):
		return filter(lambda v: isinstance(v,varType),self.qsoVars)
	def resample(self):
		for var in self.qsoVars:
			# how to more reasonably know what variables are needed to
			# be passed down? the nominal use case is updating variables
			# which depend on absMag. for now just passing the whole object...
			if var.update:
				var.resample(self)
				self.data[var.name] = var(self.nObj)
	def distMod(self,z):
		return self.cosmo.distmod(z).value
	def read(self,gridFile):
		self.data = Table.read(gridFile)
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
	def __init__(self,qsoVars,n=None,**kwargs):
		super(QsoSimPoints,self).__init__(qsoVars,**kwargs)
		data = { var.name:var(n) for var in qsoVars }
		self.data = Table(data)
		self.nObj = len(self.data)
		self.gridShape = (self.nObj,)
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
		self.zbins = np.arange(z.min(),z.max()+0.005,0.005)
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

