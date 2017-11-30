#!/usr/bin/env python

import os,sys
import numpy as np
from sklearn.mixture import GaussianMixture
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy import units as u

def make_coreqso_table(dr14qso,ebosstarg):
	if isinstance(dr14qso,basestring):
		dr14qso = Table.read(dr14qso)
	if isinstance(ebosstarg,basestring):
		ebosstarg = Table.read(ebosstarg)
	#
	dr14coo = SkyCoord(dr14qso['RA'],dr14qso['DEC'],unit=u.deg)
	# restrict to CORE quasar targets
	ii = np.where(ebosstarg['EBOSS_TARGET1'] & (1<<10) > 0)[0]
	ebosstarg = ebosstarg[ii]
	ebosstargcoo = SkyCoord(ebosstarg['RA'],ebosstarg['DEC'],unit=u.deg)
	# now identify confirmed quasars from DR14 in the target list
	m1,m2,sep,_ = dr14coo.search_around_sky(ebosstargcoo,2*u.arcsec)
	# for some reason there is a repeated entry...
	_,ii = np.unique(m1,return_index=True)
	dr14qso = dr14qso[m2[ii]]
	# just a sanity check
	jj = np.where(dr14qso['EXTINCTION']>0)[0]
	assert np.allclose(dr14qso['EXTINCTION'][jj],
	                   ebosstarg['EXTINCTION'][m1[ii[jj]]],atol=1e-3)
	# extract all the WISE columns from targeting
	wisecols = ['W1_MAG','W1_MAG_ERR',
	            'W1_NANOMAGGIES','W1_NANOMAGGIES_IVAR',
	            'W2_NANOMAGGIES','W2_NANOMAGGIES_IVAR',
	            'HAS_WISE_PHOT']
	# overwriting the DR14Q flux fields because they have invalid entries
	for k in wisecols + ['EXTINCTION','PSFFLUX','PSFFLUX_IVAR']:
		dr14qso[k] = ebosstarg[k][m1[ii]]
	dr14qso.write('ebosscore_dr14q.fits',overwrite=True)

def get_column_ratio(a,j):
	col = a[:,[j]].copy()
	a /= col
	return col,np.delete(a,j,1)

class eBossQsos(object):
	def __init__(self,fileName='ebosscore_dr14q.fits',zrange=None):
		dat = Table.read(fileName)
		if zrange is not None:
			dat = dat[np.logical_and(dat['Z']>zrange[0],dat['Z']<zrange[1])]
		self.data = dat
		self.set_specz_col('Z')
		self._extract_fluxes()
	def set_specz_col(self,colname):
		self.speczName = colname
		self.specz = self.data[colname]
	def _extract_fluxes(self,ratios='byref',refNum=3):
		# SDSS
		sdssFluxes = np.array(self.data['PSFFLUX'])
		extCorr = np.array(self.data['EXTINCTION'])
		self.sdssFluxes = np.ma.array(sdssFluxes*10**(0.4*extCorr))
		# XXX need the galex ext values
		# Galex
		fuv = np.ma.array(self.data['FUV'],mask=self.data['FUV_IVAR']==0)
		nuv = np.ma.array(self.data['NUV'],mask=self.data['NUV_IVAR']==0)
		self.galexFluxes = np.ma.vstack([fuv,nuv]).transpose()
		# UKIDSS
		self.ukidssFluxes = np.ma.vstack(
		         [ np.ma.array(self.data[b+'FLUX'],
		                       mask=self.data['UKIDSS_MATCHED']==0)
		              for b in 'YJHK' ] ).transpose()
		# units are W/m^2/Hz, 1e26 converts to Jy and 5.44 to nanomaggie
		self.ukidssFluxes *= 1e26*10**5.44
		# WISE (convert from Vega)
		# using extinction conversions from ebosstarget_qso_selection.pro
		w1 = np.ma.array(self.data['W1_NANOMAGGIES'],
		                 mask=self.data['HAS_WISE_PHOT']!='T')
		w1 *= 10**(0.4*(0.184*extCorr[:,2]/2.285))
		w2 = np.ma.array(self.data['W2_NANOMAGGIES'],
		                 mask=self.data['HAS_WISE_PHOT']!='T')
		w2 *= 10**(0.4*(0.113*extCorr[:,2]/2.285))
		self.wiseVegaFluxes = np.ma.vstack([w1,w2]).transpose()
		w1 *= 10**(-0.4*2.699)
		w2 *= 10**(-0.4*3.339)
		self.wiseFluxes = np.ma.vstack([w1,w2]).T
	def get_optwise(self):
		f_WISE = np.average(self.wiseVegaFluxes,axis=1,weights=[1,0.5])
		f_opt = np.average(self.sdssFluxes[:,1:4],axis=1,weights=[1,0.8,0.6])
		return f_opt,f_WISE
	def extract_features(self,featureset=['sdss','z'],
	                     refband='i',ratios='byref'):
		fluxes = []
		names = []
		if 'galex' in featureset:
			fluxes.append(self.galexFluxes)
			names.extend(['f','n'])
		if 'sdss' in featureset:
			fluxes.append(self.sdssFluxes)
			names.extend(list('ugriz'))
		if 'ukidss' in featureset:
			fluxes.append(self.ukidssFluxes)
			names.extend(list('YJHK'))
		if 'wise' in featureset:
			fluxes.append(self.wiseFluxes)
			names.extend(['W1','W2'])
		#
		fluxes = np.ma.hstack(fluxes)
		j = names.index(refband)
		if ratios=='byref':
			refFlux,fluxes = get_column_ratio(fluxes,j)
			names = [refband] + [ b+'/'+refband for b in names ]
		elif ratios=='neighboring':
			refFlux = fluxes[:,[j]]
			fluxes = fluxes[:,:-1]/fluxes[:,1:]
			names = [refband] + [ b1+'/'+b2 
			                       for b1,b2 in zip(names[:-1],names[1:]) ]
		else:
			raise ValueError
		features = [refFlux,fluxes]
		if 'z' in featureset:
			zfeat = np.ma.array(self.specz)[:,None]
			features = [zfeat] + features
			names = ['z_q'] + names
		return np.ma.hstack(features),names

def prep_simqsos(simqsos,refband='i'):
	j = 3 # XXX
	ii = np.where(simqsos['selected'])[0]
	fluxes = np.array(simqsos['obsFlux'][ii,:5]) # XXX only sdss for now
	refFlux,fratios = get_column_ratio(fluxes,j)
	X = np.ma.hstack([simqsos['z'][ii,None],refFlux,fratios])
	return X

def model_selection(simqsos,refband='i'):
	X = prep_simqsos(simqsos,refband)
	cv_types = ['spherical', 'tied', 'diag', 'full']
	n_components = range(5,31,5)
	bic = []
	for cv_type in cv_types:
		for ncomp in n_components:
			gmm = GaussianMixture(ncomp,covariance_type=cv_type)
			gmm.fit(X)
			bic.append(gmm.bic(X))
			print cv_type,ncomp,bic[-1]
	return bic

def fit_simqsos(simqsos,ncomp=15,refband='i'):
	X = prep_simqsos(simqsos,refband)
	gmm = GaussianMixture(ncomp)
	return gmm.fit(X)

def test(fn,qsos=None):
	simqsos = Table.read(fn)
	if qsos is None:
		qsos = eBossQsos()
	features,names = qsos.extract_features()
	print names
	fit = fit_simqsos(simqsos)
	s = fit.score(features)
	print fn,s

if __name__=='__main__':
	#make_coreqso_table(sys.argv[1],sys.argv[2])
	test(sys.argv[1])

