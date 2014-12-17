#!/usr/bin/env python

from astropy.io import fits
from collections import OrderedDict
from scipy.interpolate import interp1d
from scipy.integrate import simps

from .sqbase import datadir

_default_bands = {
  'SDSS':list('ugriz'),
  'CFHT':list('ugriz'),
  'UKIDSS':list('YJHK'),
  'WISE':['W1','W2','W3','W4'],
  'GALEX':list('fn'),
}

def load_photo_map(params):
	phmap = OrderedDict()
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
			phmap[bpname] = dict(Rlam=fcurv,norm=norm,data=fdat)
	return phmap

