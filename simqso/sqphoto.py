#!/usr/bin/env python

import numpy as np
from collections import OrderedDict
from scipy.interpolate import interp1d
from scipy.integrate import simps
from scipy.constants import c
c_Angs = c*1e10

from astropy.io import fits

from .sqbase import datadir

_default_bands = {
  'SDSS':list('ugriz'),
  'CFHT':list('ugriz'),
  'UKIDSS':list('YJHK'),
  'WISE':['W1','W2','W3','W4'],
  'GALEX':list('fn'),
}

softening_parameter = np.array([1.4,0.9,1.2,1.8,7.4])*1e-10

def nmgy2abmag(_b,f,df=None):
	ii = np.where(f>0)
	mag = 99.99 + np.zeros_like(f)
	mag[ii] = 22.5 - 2.5*np.log10(f[ii])
	if df is None:
		return mag
	else:
		err = np.zeros_like(mag)
		err[ii] = 1.0857 * df[ii]/f[ii]
		return mag,err

def abmag2nmgy(_b,m):
	return 10**(-0.4*(m - 22.5))

def nmgy2asinhmag(_b,f,df=None):
	b = softening_parameter['ugriz'.find(_b)]
	mag = -1.0857*(np.arcsinh(1e-9*f/(2*b)) + np.log(b))
	if df is None:
		return mag
	else:
		err = 1.0857 * 1e-9*df/(2*b) / np.sqrt(1+((1e-9*f)/(2*b))**2)
		return mag,err

def asinhmag2nmgy(_b,m):
	b = softening_parameter['ugriz'.find(_b)]
	return 2*b*np.sinh(m/(-1.0857) - np.log(b)) / 1e-9

# Vega...

#
# SDSS photometry model
#

_sdss_phot_pars = {
  'gain':[     1.6,   3.925,  4.7225,    4.86,    4.76, ],
  'darkVariance':[ 9.45625, 1.63125, 1.16125,    6.25,   1.105, ],
  'sky':[ 1.33136, 1.70364, 4.35521, 8.10383, 25.3321, ],
  'skyErr':[ 0.00657128, 0.00254991, 0.00409365, 0.00670879, 0.0255355, ],
  'nEffPsf':[ 36.4706, 32.1649, 27.2765,  24.854, 25.9643, ],
  'nMgyPerCount':[ 0.00981, 0.00378, 0.00507, 0.00662, 0.0337, ],
}

class sdssPhotoUnc(object):
	'''sdss_photo_unc(b,f_nmgy)
	   In a given SDSS band b, provide the uncertainty for a given flux in 
	   nanomaggies (f_nmgy) based on the distribution of observing conditions.
	   --> Currently underestimates true scatter by using gaussians for the
	       scatter, whereas the true distributions generally have long tails
		   to higher values for sky, nEff, etc.
	   see http://classic.sdss.org/dr7/algorithms/fluxcal.html for details
	'''
	def __init__(self,b):
		i = 'ugriz'.find(b)
		self.pixArea = 0.396**2 # pix -> arcsec^2
		self.gain = _sdss_phot_pars['gain'][i]
		self.darkVar = _sdss_phot_pars['darkVariance'][i]
		self.skyMean = _sdss_phot_pars['sky'][i]
		self.skyStd = {'u':0.4,'g':0.4,'r':1.2,'i':2.0,'z':5.0}[b]
		self.skyMin = {'u':0.6,'g':1.0,'r':2.2,'i':3.2,'z':8.3}[b]
		#skyErr_nmgy = _sdss_phot_pars['skyErr'][i] # not used...
		# nEffPsf distribution is roughly similar in all bands
		self.npixMean = _sdss_phot_pars['nEffPsf'][i]
		self.npixStd = 5.0
		self.npixMin = 10.0
		self.c2fMean = _sdss_phot_pars['nMgyPerCount'][i]
		self.c2fStd = {'u':2.3e-3,'g':3.9e-4,'r':2.7e-4,
		               'i':3.7e-4,'z':5.6e-3}[b]
		self.c2fMin = {'u':5.7e-3,'g':2.3e-3,'r':3.5e-3,
		               'i':4.7e-3,'z':1.4e-2}[b]
		# add in a global photometric calibration error term
		self.calibrationError = 0.015
	def __call__(self,f_nmgy):
		shape = f_nmgy.shape
		gain = self.gain
		pixArea = self.pixArea
		darkVar = self.darkVar
		sky_nmgy_asec2 = np.clip(np.random.normal(self.skyMean,
		                                          self.skyStd,shape),
		                         self.skyMin,np.inf)
		npix = np.clip(np.random.normal(self.npixMean,self.npixStd,shape),
		               self.npixMin,np.inf)
		c2f = np.clip(np.random.normal(self.c2fMean,self.c2fStd,shape),
		              self.c2fMin,np.inf)
		df = np.sqrt( f_nmgy*(c2f/gain) + 
		               sky_nmgy_asec2*pixArea*npix*(c2f/gain) +
		                darkVar*npix*(c2f/gain)**2 +
		                 (self.calibrationError*f_nmgy)**2 )
		return df

class empiricalPhotoUnc(object):
	'''approximation only valid in sky-dominated regime'''
	def __call__(self,f_nmgy):
		shape = f_nmgy.shape
		magAB = nmgy2abmag(self.b,f_nmgy)
		scatter = np.clip(self.scatter_a*magAB + self.scatter_b, 0.01, np.inf)
		b = self.b + scatter*np.random.normal(size=shape)
		log_dm = 2.5*(self.a*magAB - b)
		dm = np.clip(10**log_dm, self.err_floor, np.inf)
		return f_nmgy * dm / 1.0857

class ukidsslasPhotoUnc(empiricalPhotoUnc):
	def __init__(self,b):
		UKIDSS_LAS_terms = np.array([[0.13616,3.1360,0.029],
		                             [0.14665,3.3081,0.043],
		                             [0.14429,3.2105,0.040],
		                             [0.15013,3.3053,0.028]])
		self.b = b
		i = 'YJHK'.find(b)
		self.a,self.b,self.scatter_b = UKIDSS_LAS_terms[i]
		# ignoring magnitude-dependent scatter since all useful fluxes are
		# in the sky-dominated regime
		self.scatter_a = 0.0
		# scatter seems to be slightly overestimated
		self.scatter_b *= 0.9
		# calibration uncertainty floor
		self.err_floor = 0.015

class sdssStripe82PhotoUnc(empiricalPhotoUnc):
	'''this fails at m<~18 when SDSS detections are no longer sky-dominated,
	   but not really interested in bright objects on the Stripe...'''
	def __init__(self,b):
		stripe82terms = np.array([[0.15127,3.8529,0.00727,-0.1308],
                                  [0.15180,4.0233,0.00486,-0.0737],
                                  [0.14878,3.8970,0.00664,-0.1077],
                                  [0.14780,3.8024,0.00545,-0.0678],
                                  [0.14497,3.5437,0.00715,-0.1121]])
		self.b = b
		i = 'ugriz'.find(b)
		self.a,self.b,self.scatter_a,self.scatter_b = stripe82terms[i]
		# calibration uncertainty floor
		self.err_floor = 0.015

class cfhtlsWidePhotoUnc(empiricalPhotoUnc):
	'''as with Stripe82, not valid at bright magnitudes (m<~19)'''
	def __init__(self,b):
		cfhtlswideterms = np.array([[0.16191,4.4005,0.037],
                                    [0.15508,4.3392,0.034],
                                    [0.15902,4.3399,0.015],
                                    [0.15721,4.2786,0.028],
                                    [0.16092,4.1967,0.034]])
		self.b = b
		i = 'ugriz'.find(b)
		self.a,self.b,self.scatter_b = cfhtlswideterms[i]
		# ignoring magnitude-dependent scatter since all useful fluxes are
		# in the sky-dominated regime
		self.scatter_a = 0.0
		# calibration uncertainty floor
		self.err_floor = 0.015

def load_photo_map(params):
	bandpasses = OrderedDict()
	filterdata = fits.open(datadir+'filtercurves.fits')
	for photDesc in params['PhotoSystems']:
		try:
			photSys,survey,bands = photDesc
		except ValueError:
			photSys,survey = photDesc
			try:
				bands = _default_bands[photSys]
			except:
				raise ValueError('%s not a valid photo system' % photSys)
		for band in bands:
			bpName = '-'.join([photSys,survey,band])
			bpExt = '-'.join([photSys,band])
			fdat = filterdata[bpExt].data
			fcurv = interp1d(fdat.lam,fdat.Rlam,
			                 bounds_error=False,fill_value=0.0,kind='slinear')
			# precompute the bandpass normalization
			norm = simps(fdat.Rlam/fdat.lam, fdat.lam)
			bandpasses[bpName] = dict(Rlam=fcurv,norm=norm,data=fdat)
	# XXX this loop is repeated until I converge on a mapping framework
	cdfmap = fits.getdata(datadir+'photomap_cdfs.fits')
	mapObserved = {}
	magSys = {}
	for photDesc in params['PhotoSystems']:
		try:
			photSys,survey,bands = photDesc
		except ValueError:
			photSys,survey = photDesc
			bands = _default_bands[photSys]
		if photSys == 'SDSS' and survey == 'Legacy':
			for band in bands:
				bpName = '-'.join([photSys,survey,band])
				mapObserved[bpName] = sdssPhotoUnc(band) 
				magSys[bpName] = 'asinh'
		elif photSys == 'SDSS' and survey == 'Stripe82':
			for band in bands:
				bpName = '-'.join([photSys,survey,band])
				mapObserved[bpName] = sdssStripe82PhotoUnc(band) 
				magSys[bpName] = 'AB'
		elif photSys == 'UKIDSS' and survey == 'LAS':
			for band in bands:
				bpName = '-'.join([photSys,survey,band])
				mapObserved[bpName] = ukidsslasPhotoUnc(band) 
				magSys[bpName] = 'AB'
		elif photSys == 'CFHT' and survey == 'CFHTLSWide':
			for band in bands:
				bpName = '-'.join([photSys,survey,band])
				mapObserved[bpName] = cfhtlsWidePhotoUnc(band) 
				magSys[bpName] = 'AB'
		else:
			for band in bands:
				bpName = '-'.join([photSys,survey,band])
				bppfx = '_'.join([photSys,survey,band])
				iend = np.where(cdfmap[bppfx+'_cdf']>=1)[0][0]
				cdffun = interp1d(cdfmap[bppfx+'_cdf'][:iend+1],
				                  cdfmap[bppfx+'_df'][:iend+1])
				mapObserved[bpName] = \
				       lambda flux: cdffun(np.random.random(flux.shape))
				magSys[bpName] = 'AB'
	return dict(bandpasses=bandpasses,mapObserved=mapObserved,magSys=magSys)

def getPhotoCache(wave,photoMap):
	photoCache = {}
	for b,bp in photoMap['bandpasses'].items():
		bpdata = bp['data']
		i1,i2 = np.searchsorted(wave,bpdata['lam'][[0,-1]],side='right')
		if i1==i2:
			lam,Rlam,dlam = 0.0,0.0,0.0
		else:
			lam = wave[i1:i2]
			dlam = np.diff(wave[i1:i2])
			dlam = np.concatenate([dlam,[dlam[-1],]])
			Rlam = bp['Rlam'](wave[i1:i2])
		photoCache[b] = {'ii':(i1,i2),'lam_Rlam_dlam':lam*Rlam*dlam}
	return photoCache

conv_Slam_to_Snu = 1/(c_Angs * 3631e-23)

def calcSynPhot(spec,photoMap,photoCache=None,mags=None,fluxes=None):
	if mags is None:
		mags = np.zeros(len(photoMap['bandpasses']))
	if fluxes is None:
		fluxes = np.zeros(len(photoMap['bandpasses']))
	if photoCache is None:
		photoCache = getPhotoCache(spec.wave,photoMap)
	for j,b in enumerate(photoMap['bandpasses']):
		fnorm = photoMap['bandpasses'][b]['norm']
		i1,i2 = photoCache[b]['ii']
		lamRlamdlam = photoCache[b]['lam_Rlam_dlam']
		flam = spec.f_lambda[i1:i2]
		flux = np.sum(flam*lamRlamdlam) / fnorm
		fluxes[j] = flux * conv_Slam_to_Snu 
		if fluxes[j] == 0:
			mags[j] = 99.99
		else:
			mags[j] = min(-2.5*np.log10(fluxes[j]),99.99)  # AB mag
	fluxes *= 1e9 # nanomaggies
	return mags,fluxes

def calcObsPhot(synFlux,photoMap):
	obsFlux = np.empty_like(synFlux)
	obsFluxErr = np.empty_like(synFlux)
	obsMag = np.empty_like(synFlux)
	obsMagErr = np.empty_like(synFlux)
	gridShape = synFlux.shape[:-1]
	for j,b in enumerate(photoMap['bandpasses']):
		obsFluxErr[...,j] = photoMap['mapObserved'][b](synFlux[...,j])
		obsFlux[...,j] = synFlux[...,j] + \
		                   obsFluxErr[...,j]*np.random.randn(*gridShape)
		if photoMap['magSys'][b]=='AB':
			obsMag[...,j],obsMagErr[...,j] = nmgy2abmag(b,obsFlux[...,j],
			                                            obsFluxErr[...,j])
		elif photoMap['magSys'][b]=='asinh':
			obsMag[...,j],obsMagErr[...,j] = nmgy2asinhmag(b,obsFlux[...,j],
			                                               obsFluxErr[...,j])
		else:
			raise ValueError
	return {'obsFlux':obsFlux,'obsFluxErr':obsFluxErr,
	        'obsMag':obsMag,'obsMagErr':obsMagErr}

