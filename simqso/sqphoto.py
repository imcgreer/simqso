#!/usr/bin/env python

def _load_SDSS(*args,**kwargs):
	bands, = args
	if bands is None:
		bands = 'ugriz'

sysdef = {
  'SDSS':_load_SDSS,
}

def load_photo_map(params):
	phmap = {}
	for photsys in params['PhotoSystems']:
		# will default to None if PhotoBands is not present
		bands = params.get('PhotoBands',{}).get(photsys)
		phmap[photsys] = sysdef[photsys](bands)
	return phmap

