#!/usr/bin/env python

import os,sys
from copy import deepcopy
from collections import defaultdict
import numpy as np
from scipy.stats import binned_statistic
import matplotlib.pyplot as plt
from matplotlib import ticker
from astropy.table import Table,hstack
from simqso import sqgrids as grids

import ebossmodels
import ebosscore
import ebossfit

def percfun(pval):
	return lambda x: np.percentile(x,pval)

def calc_colorz(z,clrs,pvals,zedges):
	zbins = zedges[:-1] + np.diff(zedges)/2
	colorz = np.zeros((len(pvals),clrs.shape[-1],len(zbins)),dtype=np.float32)
	for i,pval in enumerate(pvals):
		clrz,_,_ = binned_statistic(z,clrs.T,percfun(pval),zedges)
		colorz[i] = clrz
	return colorz

def sim_colorz(simqsos,pvals,zedges,ratios=True,refBand=None):
	b = ebosscore.BandIndexes(simqsos)
	c_ii = [ b(_b) for _b in list('ugri')+['W1'] ]
	fluxk = 'Flux' if ratios else 'Mag'
	colorz = {}
	for which in ['syn','obs']:
		z = simqsos['z']
		flux = simqsos[which+fluxk]
		if which=='obs':
			flux = flux[simqsos['selected']]
			z = z[simqsos['selected']]
		if ratios:
			if refBand is None:
				clrs = flux[:,:-1] / flux[:,1:]
			else:
				clrs = flux / flux[:,[refBand]]
				clrs = np.delete(flux,refBand,1)
		else:
			clrs = -np.diff(flux,axis=1)
		colorz[which] = calc_colorz(z,clrs[:,c_ii],pvals,zedges)
	tab = Table(colorz)
	return tab

def ebosscore_colorz(coreqsos,pvals,zedges):
	features,names = coreqsos.extract_features(featureset=['sdss','wise'],
	                                            ratios='neighboring')
	print names
	clrs = features[:,1:]
	clrs = np.delete(clrs,-2,1)
	colorz = calc_colorz(coreqsos.specz,clrs,pvals,zedges)
	return Table(dict(ebosscore=colorz))

fratio_yrange = {
  'u/g':(-0.3,1.3), 'g/r':(0.1,1.3), 'r/i':(0.5,1.2), 'i/z':(0.5,1.3),
  'W1/W2':(0.4,1.3),
}

def colorz_compare(simqsos,coreqsos):
	zedges = np.linspace(0.9,4.0,32)
	zbins = zedges[:-1] + np.diff(zedges)/2
	pvals = [25,50,75]
	colorz = sim_colorz(simqsos,pvals,zedges)
	colorz2 = ebosscore_colorz(coreqsos,pvals,zedges)
	colorz = hstack([colorz,colorz2])
	bandnames = list('ugriz') #+ ['W1','W2']
	colornames = [ b1+'-'+b2 for b1,b2 in zip(bandnames[:-1],bandnames[1:]) ]
	colornames += ['W1-W2'] # otherwise get z-W1
	plt.figure(figsize=(10,7))
	plt.subplots_adjust(0.075,0.05,0.97,0.97,0.2,0.13)
	for j in range(colorz['syn'].shape[1]):
		plt.subplot(3,2,j+1)
		for which in ['syn','obs','ebosscore']:
			c = {'syn':'C0','obs':'C1','ebosscore':'C2'}[which]
			plt.fill_between(zbins,colorz[which][0,j],colorz[which][2,j],
			                 color=c,alpha=0.3)
			plt.plot(zbins,colorz[which][1,j],c=c,ls='-')
		plt.xlim(0.85,4.05)
		if False:
			yr = [ (-0.7,4.2), (-0.3,1.7), (-0.15,0.5), (-0.2,0.5), (-0.15,0.75) ]
		else:
			yr = [ (-0.3,1.3), (0.1,1.3), (0.5,1.2), (0.5,1.3), (0.4,1.3) ]
		plt.ylim(*yr[j])
		plt.ylabel(colornames[j])
	plt.subplot(3,2,6)
	plt.hist(simqsos['z'],zbins,log=True)
	plt.hist(simqsos['z'][simqsos['selected']],zbins,log=True)
	plt.hist(coreqsos.specz,zbins,log=True,alpha=0.5)
	plt.xlim(0.85,4.05)
	plt.ylabel('n(z)')

def get_colorz_bins(mrange=(-27,-23),zrange=(0.9,4.0),nm=7,nz=500):
	mbins = np.linspace(*tuple(mrange+(nm,)))
	zbins = np.linspace(*tuple(zrange+(nz,)))
	return mbins,zbins

def model_colorz_tracks(model,forestFile,**kwargs):
	mbins,zbins = get_colorz_bins(**kwargs)
	nm,nz = len(mbins),len(zbins)
	M,z = np.meshgrid(mbins,zbins,indexing='ij')
	M = grids.AbsMagVar(grids.FixedSampler(M.flatten()),restWave=1450)
	z = grids.RedshiftVar(grids.FixedSampler(z.flatten()))
	qsos = grids.QsoSimPoints([M,z],
	                          cosmo=ebosscore.dr9cosmo,
	                          units='luminosity')
	qsos = ebosscore.runsim(model,None,forestFile,qsos,
	                        medianforest=True,const=True,nophot=True)
	synmags = np.array(qsos.data['synMag'].reshape(nm,nz,-1))
	synclrs = -np.diff(synmags,axis=-1)
	synfluxes = np.array(qsos.data['synFlux'].reshape(nm,nz,-1))
	synfratios = synfluxes[...,:-1] / synfluxes[...,1:]
	return dict(mbins=mbins,zbins=zbins,
	            synmags=synmags,syncolors=synclrs,
	            synfluxes=synfluxes,synfratios=synfratios,
	            qsos=qsos)

def colorz_param_trends(modelName):
	forestFile ='sdss_forest_grid.fits'
	tab = Table()
	#
	def add_entry(tab,nm,s,cz):
		for k1,k2 in [('mags','synmags'),('clrs','syncolors'),
		              ('fluxes','synfluxes'),('fratio','synfratios')]:
			k = '_'.join([nm,s,k1])
			tab[k] = cz[k2]
	#
	model = deepcopy(ebossmodels.qso_models[modelName])
	cname = model['continuum']
	for s in [-1.8,-1.5,-1.2]:
		model['continuum'] = deepcopy(ebossmodels.cont_models[cname])
		model['continuum'][0][0] = (s,0.3)
		print s,model
		cz = model_colorz_tracks(model,forestFile)
		add_entry(tab,'fuv','%+4.1f'%s,cz)
		print
	#
	model = deepcopy(ebossmodels.qso_models[modelName])
	cname = model['continuum']
	for s in [-0.2,-0.4,-0.6]:
		model['continuum'] = deepcopy(ebossmodels.cont_models[cname])
		model['continuum'][0][1] = (s,0.3)
		print s,model
		cz = model_colorz_tracks(model,forestFile)
		add_entry(tab,'uv','%+4.1f'%s,cz)
		print
	#
	model = deepcopy(ebossmodels.qso_models[modelName])
	for l in ['LyA','CIV','MgII','Hbeta','HA','Pa']:
		for scl in [0.5,1.0,2.0]:
			model['emlines'] = {'scaleEWs':{}}
			if l == 'Hbeta':
				model['emlines']['scaleEWs'][l] = scl
			elif l == 'Pa':
				scl = scl**2
				for c in ['alpha','beta']:
					model['emlines']['scaleEWs'][l+c] = scl
			else:
				for c in 'bn':
					model['emlines']['scaleEWs'][l+c] = scl
			print l,model
			cz = model_colorz_tracks(model,forestFile)
			add_entry(tab,l,'%3.1f'%scl,cz)
		print
	#
	if 'dustem' in ebossmodels.qso_models[modelName]:
		model = deepcopy(ebossmodels.qso_models[modelName])
		dustnm = model['dustem']
		for f in [0.5,1.0,2.0]:
			model['dustem'] = deepcopy(ebossmodels.dustem_models[dustnm])
			for c in model['dustem']:
				pars = model['dustem'][c]
				model['dustem'][c] = [(pars[0][0]*f,None)] + pars[1:]
			print f,model
			cz = model_colorz_tracks(model,forestFile)
			add_entry(tab,'hotdust','%3.1f'%f,cz)
			print
	return cz['mbins'],cz['zbins'],tab

def plot_trends(modelName,trendFile,coreqsos):
	mbins,zbins = get_colorz_bins()
	eboss_zedges = np.linspace(0.9,4.0,32)
	eboss_zbins = eboss_zedges[:-1] + np.diff(eboss_zedges)/2
	pvals = [25,50,75]
	obs_colorz = ebosscore_colorz(coreqsos,pvals,eboss_zedges)['ebosscore']
	trends = Table.read(trendFile)
	pars = defaultdict(list)
	for c in trends.colnames:
		name,val,which = c.split('_')
		if which=='clrs':
			pars[name].append(val)
	#nrow,ncol = 5,2
	nrow,ncol = 3,2
	m_i = 3
	asmags = False
	f = '-' if asmags else '/'
	clrnames = [b1+f+b2 for b1,b2 in zip('ugri','griz')] + ['W1'+f+'W2']
	pclrs = [ 'C%d'%i for i in range(1,10,2) ]
	for p,vals in pars.items():
		#plt.figure(figsize=(7.5,8))
		plt.figure(figsize=(7.5,5.5))
		plt.subplots_adjust(0.10,0.07,0.98,0.94,0.28)
		plt.figtext(0.5,0.99,p,ha='center',va='top',size=15)
		for pnum,j in enumerate([0,1,2,3,-1],start=1):
			ax = plt.subplot(nrow,ncol,pnum)
			for _i,v in enumerate(vals):
				if asmags:
					k = '_'.join([p,v,'clrs'])
				else:
					k = '_'.join([p,v,'fratio'])
				ax.plot(zbins,trends[k][m_i,:,j],label=v,c=pclrs[_i])
			c = 'C0'
			obs_cz = obs_colorz[:,j]
			if asmags:
				obs_cz = -2.5*np.log10(obs_cz.clip(1e-2,np.inf))
			ax.fill_between(eboss_zbins,obs_cz[0],obs_cz[2],
			                color=c,alpha=0.3)
			ax.plot(eboss_zbins,obs_cz[1],c=c,ls='-')
			ax.set_ylabel(clrnames[pnum-1])
			ax.set_ylim(fratio_yrange[clrnames[pnum-1]])
			plt.setp(ax.get_xticklabels()+ax.get_yticklabels(),fontsize=9)
			ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.2))
			if j==0: plt.legend()
		plt.savefig(modelName+'_'+p+'.pdf')
		plt.close()

def plot_model_trends(model='all'):
	coreqsos = ebossfit.eBossQsos() 
	if model=='all':
		models = ebossmodels.qso_models.keys()
	else:
		models = [model]
	for modelName in models:
		m,z,trends = colorz_param_trends(modelName)
		trendfn = modelName+'_trends.fits'
		trends.write(trendfn,overwrite=True)
		plot_trends(modelName,trendfn,coreqsos)

def model_spectrum(model,**kwargs):
	from simqso.sqbase import fixed_R_dispersion
	from simqso.sqrun import buildQsoSpectrum
	wave = fixed_R_dispersion(0.1e4,10e4,500)
	m,z = np.array([-25]),np.array([1.0])
	M = grids.AbsMagVar(grids.FixedSampler(m),restWave=1450)
	z = grids.RedshiftVar(grids.FixedSampler(z))
	qsos = grids.QsoSimPoints([M,z],
	                          cosmo=ebosscore.dr9cosmo,
	                          units='luminosity')
	qsos = ebosscore.runsim(model,None,None,qsos,wave=wave,
	                        const=True,nophot=True)
	specFeatures = qsos.getVars(grids.SpectralFeatureVar)
	spec,comp = buildQsoSpectrum(wave,qsos.cosmo,specFeatures,
	                             qsos.data[0],save_components=True)
	spec.wave /= 2.0
	return spec,comp,qsos

def model_spectrum_z(model,**kwargs):
	from simqso.sqbase import datadir
	from astropy.io import fits
	#
	filterdata = fits.open(datadir+'filtercurves.fits')
	spec,comp,qsos = model_spectrum(model,**kwargs)
	#
	wscl = 1e-4
	fscl = 1e12
	rflam = spec.wave
	#
	filts = ['SDSS-'+b for b in 'UGRIZ'] + \
	        ['UKIDSS-'+b for b in 'JHK'] + \
	        ['WISE-'+b for b in ['W1','W2']]
	contFeatures = qsos.getVars(grids.ContinuumVar)
	plt.ioff()
	for z in np.linspace(1.0,4.0,11):
		plt.figure()
		plt.subplots_adjust(0.1,0.1,0.98,0.94)
		ax = plt.subplot(111)
		for cf in contFeatures:
			k = cf.name
			plt.plot(rflam*wscl,rflam*comp[k].f_lambda*fscl,label=k)
			if isinstance(cf,grids.BrokenPowerLawContinuumVar):
				plcont = comp[k].f_lambda
		for k in ['emLines','fetempl']:
			plt.plot(rflam*wscl,
			         0.5*rflam*(1+comp[k].f_lambda)*fscl*plcont,label=k)
		plt.plot(rflam*wscl,rflam*spec.f_lambda*fscl,label='total')
		for filt in filts:
			fdat = filterdata[filt].data
			ffscl = 1/fdat['Rlam'].max()
			plt.fill_between(fdat['lam']/(1+z)*wscl,
			                 fdat['Rlam']*ffscl,1e-1,alpha=0.2,label=None)
		plt.xscale('log')
		plt.yscale('log')
		plt.ylim(1e-1,2.5)
		plt.title(r'$z=%.2f$'%z)
		ax.set_xticks([0.09,0.12,0.15,0.19,0.28,0.50,0.65,1.0,2.0])
		plt.xlim(0.3/(1+z),6.0/(1+z))
		ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
		ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
		plt.legend(ncol=3,fontsize=9,loc='upper right')
		plt.xlabel('wave [$\mu$m]')
		plt.ylabel(r'$\lambda{f}_\lambda$')
		plt.savefig('meanspec_z%.1f.pdf'%z)
		plt.close()
	plt.ion()

def compare_model_spec():
	plt.figure()
	for name,model in ebossmodels.qso_models.items():
		spec,comp,_ = model_spectrum(model)
		i = np.searchsorted(spec.wave,1450)
		plt.plot(spec.wave/1e4,
		         spec.wave*spec.f_lambda/spec.f_lambda[i],
		         label=name)
	plt.legend()
	plt.xscale('log')
	plt.yscale('log')

if __name__=='__main__':
	import argparse
	parser = argparse.ArgumentParser(
	                          description='run eboss color-z simulations.')
	parser.add_argument('fitsfile',nargs='?',type=str,help='input file name')
	parser.add_argument('--forest',type=str,default='sdss_forest_grid.fits',
	    help='file containing forest grid (default:sdss_forest_grid.fits')
	parser.add_argument('-m','--model',type=str,default='bossdr9',
	    help='name of quasar model')
	parser.add_argument('--trends',action="store_true",
	    help='show parameter color-z mean trends instead of running sim')
	parser.add_argument('--tracks',action="store_true",
	    help='show color-z mean trends instead of running sim')
	args = parser.parse_args()
	if args.trends:
		plot_model_trends(model=args.model)
	elif args.tracks:
		model = ebossmodels.qso_models[args.model]
		cz = model_colorz_tracks(model,args.forest)
	else:
		simqsos = Table.read(args.fitsfile)
		coreqsos = ebossfit.eBossQsos()
		colorz_compare(simqsos,coreqsos)
		plt.savefig(args.fitsfile.replace('.fits','_colorz.pdf'))

