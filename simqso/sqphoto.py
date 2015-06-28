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

def load_photo_map(params):
	bandpasses = OrderedDict()
	filterdata = fits.open(datadir+'filtercurves.fits')
	photoBands = params.get('PhotoBands',_default_bands)
	for photsys in params['PhotoSystems']:
		bands = photoBands.get(photsys,_default_bands[photsys])
		for band in bands:
			bpname = photsys+'-'+band
			fdat = filterdata[bpname].data
			fcurv = interp1d(fdat.lam,fdat.Rlam,
			                 bounds_error=False,fill_value=0.0,kind='slinear')
			# precompute the bandpass normalization
			norm = simps(fdat.Rlam/fdat.lam, fdat.lam)
			bandpasses[bpname] = dict(Rlam=fcurv,norm=norm,data=fdat)
	#
	cdfmap = fits.getdata(datadir+'photomap_cdfs.fits')
	mapObserved = {}
	for photsys,errmodel in params['PhotoErrorModel'].items():
		bands = photoBands.get(photsys,_default_bands[photsys])
		for band in bands:
			bpname = photsys+'-'+band
			bppfx = '_'.join([photsys,errmodel,band])
			iend = np.where(cdfmap[bppfx+'_cdf']>=1)[0][0]
			mapObserved[bpname] = interp1d(cdfmap[bppfx+'_cdf'][:iend+1],
			                               cdfmap[bppfx+'_df'][:iend+1])
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

def calcObsPhot(synFlux,photoMap):
	x = np.random.random(synFlux.shape)
	obsFlux = np.empty_like(synFlux)
	obsFluxErr = np.empty_like(synFlux)
	gridShape = synFlux.shape[:-1]
	for j,b in enumerate(photoMap['bandpasses']):
		obsFluxErr[...,j] = photoMap['mapObserved'][b](x[...,j])
		obsFlux[...,j] = synFlux[...,j] + \
		                   obsFluxErr[...,j]*np.random.randn(*gridShape)
	obsMag = 99.99 + np.zeros_like(obsFlux)
	obsMagErr = np.zeros_like(obsFlux)
	ii = np.where(obsFlux>0)
	obsMag[ii] = 22.5 - 2.5*np.log10(obsFlux[ii])
	obsMagErr[ii] = 1.0856*(obsFluxErr[ii]/obsFlux[ii])
	return {'obsFlux':obsFlux,'obsFluxErr':obsFluxErr,
	        'obsMag':obsMag,'obsMagErr':obsMagErr}

