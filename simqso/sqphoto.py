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

def sdss_photo_unc(b,f_nmgy):
	'''sdss_photo_unc(b,f_nmgy)
	   In a given SDSS band b, provide the uncertainty for a given flux in 
	   nanomaggies (f_nmgy) based on the distribution of observing conditions.
	   --> Currently underestimates true scatter by using gaussians for the
	       scatter, whereas the true distributions generally have long tails
		   to higher values for sky, nEff, etc.
	   see http://classic.sdss.org/dr7/algorithms/fluxcal.html for details
	'''
	i = 'ugriz'.find(b)
	shape = f_nmgy.shape
	pixArea = 0.396**2 # pix -> arcsec^2
	gain = _sdss_phot_pars['gain'][i]
	darkVar = _sdss_phot_pars['darkVariance'][i]
	sky_nmgy_asec2 = np.clip(np.random.normal(_sdss_phot_pars['sky'][i],
	              {'u':0.4,'g':0.4,'r':1.2,'i':2.0,'z':5.0}[b],shape),
	              {'u':0.6,'g':1.0,'r':2.2,'i':3.2,'z':8.3}[b],np.inf)
	skyErr_nmgy = _sdss_phot_pars['skyErr'][i] # not used...
	# nEffPsf distribution is roughly similar in all bands
	npix = np.clip(np.random.normal(_sdss_phot_pars['nEffPsf'][i],5.0,shape),
	               10.0,np.inf)
	c2f = np.clip(np.random.normal(_sdss_phot_pars['nMgyPerCount'][i],
	        {'u':2.3e-3,'g':3.9e-4,'r':2.7e-4,'i':3.7e-4,'z':5.6e-3}[b],shape),
	        {'u':5.7e-3,'g':2.3e-3,'r':3.5e-3,'i':4.7e-3,'z':1.4e-2}[b],np.inf)
	df = np.sqrt( f_nmgy*(c2f/gain) + 
	               sky_nmgy_asec2*pixArea*npix*(c2f/gain) +
	                darkVar*npix*(c2f/gain)**2 )
	return df

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
				mapObserved[bpName] = lambda flux: sdss_photo_unc(band,flux)
				magSys[bpName] = 'asinh'
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

