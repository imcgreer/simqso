#!/usr/bin/env python

import numpy as np
from astropy.io import fits
from collections import OrderedDict
from scipy.interpolate import interp1d
from scipy.integrate import simps
from scipy.constants import c
c_Angs = c*1e10

from .sqbase import datadir

_default_bands = {
  'SDSS':list('ugriz'),
  'CFHT':list('ugriz'),
  'UKIDSS':list('YJHK'),
  'WISE':['W1','W2','W3','W4'],
  'GALEX':list('fn'),
}

def load_photo_map(params):
	bandpasses = OrderedDict()
	filterdata = fits.open(datadir+'filtercurves.fits')
	for photsys in params['PhotoSystems']:
		bands = params.get('PhotoBands',_default_bands).get(photsys)
		for band in bands:
			bpname = photsys+'-'+band
			fdat = filterdata[bpname].data
			fcurv = interp1d(fdat.lam,fdat.Rlam,
			                 bounds_error=False,fill_value=0.0,kind='slinear')
			# precompute the bandpass normalization
			norm = simps(fdat.Rlam/fdat.lam, fdat.lam)
			bandpasses[bpname] = dict(Rlam=fcurv,norm=norm,data=fdat)
	mapObserved = None
	return dict(bandpasses=bandpasses,mapObserved=mapObserved)

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



