#!/usr/bin/env python

import os,sys
import subprocess
import multiprocessing
import numpy as np
from scipy.stats import binned_statistic
import matplotlib.pyplot as plt
from astropy.cosmology import FlatLambdaCDM
from astropy.table import Table,hstack
from simqso import sqbase,sqrun,sqmodels,sqphoto,lumfun,hiforest
from simqso import sqgrids as grids
from simqso.sqbase import continuum_kcorr,fixed_R_dispersion

import ebossmodels

def percfun(pval):
	return lambda x: np.percentile(x,pval)

def calc_colorz(z,clrs,pvals,zedges):
	zbins = zedges[:-1] + np.diff(zedges)/2
	colorz = np.zeros((len(pvals),clrs.shape[-1],len(zbins)),dtype=np.float32)
	for i,pval in enumerate(pvals):
		clrz,_,_ = binned_statistic(z,clrs.T,percfun(pval),zedges)
		colorz[i] = clrz
	return colorz

def sim_colorz(simqsos,pvals,zedges):
	b = BandIndexes(simqsos)
	c_ii = [ b(_b) for _b in list('ugri')+['W1'] ]
	colorz = {}
	for which in ['syn','obs']:
		z = simqsos['z']
		mag = simqsos[which+'Mag']
		if which=='obs':
			mag = mag[simqsos['selected']]
			z = z[simqsos['selected']]
		clrs = -np.diff(mag,axis=1)
		colorz[which] = calc_colorz(z,clrs[:,c_ii],pvals,zedges)
	tab = Table(colorz)
	return tab

def ebosscore_colorz(coreqsos,pvals,zedges):
	deredMags = coreqsos['PSFMAG'] - coreqsos['EXTINCTION']
	z = coreqsos['Z']
	clrs = -np.diff(deredMags)
	colorz = calc_colorz(z,clrs,pvals,zedges)
	return Table(dict(ebosscore=colorz))

def colorz(simqsos,coreqsos):
	zedges = np.linspace(0.9,4.0,32)
	zbins = zedges[:-1] + np.diff(zedges)/2
	pvals = [25,50,75]
	colorz = sim_colorz(simqsos,pvals,zedges)
	colorz2 = ebosscore_colorz(coreqsos,pvals,zedges)
	colorz = hstack([colorz,colorz2])
	bandnames = list('ugriz') #+ ['W1','W2']
	colornames = [ b1+'-'+b2 for b1,b2 in zip(bandnames[:-1],bandnames[1:]) ]
	colornames += ['W1-W2'] # otherwise get z-W1
	plt.figure(figsize=(10,8))
	plt.subplots_adjust(0.07,0.05,0.97,0.97,0.2,0.1)
	for j in range(colorz['syn'].shape[1]):
		plt.subplot(3,2,j+1)
		for which in ['syn','obs','ebosscore']:
			if which=='ebosscore' and j==4:  continue # XXX
			c = {'syn':'C0','obs':'C1','ebosscore':'C2'}[which]
			plt.fill_between(zbins,colorz[which][0,j],colorz[which][2,j],
			                 color=c,alpha=0.3)
			plt.plot(zbins,colorz[which][1,j],c=c,ls='-')
		plt.xlim(0.85,4.05)
		yr = [ (-0.7,4.2), (-0.3,1.7), (-0.15,0.5), (-0.2,0.5), (-0.15,0.75) ]
		plt.ylim(*yr[j])
		plt.ylabel(colornames[j])
	plt.subplot(3,2,6)
	plt.hist(simqsos['z'],zbins,log=True)
	plt.hist(simqsos['z'][simqsos['selected']],zbins,log=True)
	plt.hist(coreqsos['Z'],zbins,log=True,alpha=0.5)
	plt.xlim(0.85,4.05)
	plt.ylabel('n(z)')

def run_colorz_sim(model,nm=7,nz=500):
	mrange = (-27,-23)
	zrange = (0.9,4.0)
	mbins = np.linspace(*tuple(mrange+(nm,)))
	zbins = np.linspace(*tuple(zrange+(nz,)))
	M,z = np.meshgrid(mbins,zbins,indexing='ij')
	M = grids.AbsMagVar(grids.FixedSampler(M.flatten()),restWave=1450)
	z = grids.RedshiftVar(grids.FixedSampler(z.flatten()))
	qsos = grids.QsoSimPoints([M,z],cosmo=dr9cosmo,units='luminosity')
	qsos = runsim(model,None,'sdss_forest_grid.fits',qsos,
	              medianforest=True,const=True,nophot=True)
	synmags = np.array(qsos.data['synMag'].reshape(nm,nz,-1))
	synclrs = -np.diff(synmags,axis=-1)
	return dict(mbins=mbins,zbins=zbins,
	            synmags=synmags,syncolors=synclrs,
	            qsos=qsos)

if __name__=='__main__':
	import argparse
	parser = argparse.ArgumentParser(
	                          description='run eboss quasar simulations.')
	parser.add_argument('output',type=str,help='output file name')
	parser.add_argument('--forest',type=str,default='sdss_forest_grid.fits',
	    help='file containing forest grid (default:sdss_forest_grid.fits')
	parser.add_argument('-m','--model',type=str,default='bossdr9',
	    help='name of quasar model')
	parser.add_argument('-p','--processes',type=int,default=1,
	    help='number of processes to create')
	parser.add_argument('--qlf',type=str,default='bossdr9',
	    help='seed QLF model (default: BOSS DR9 PLE-LEDE)')
	parser.add_argument('--skyarea',type=float,default=3000,
	    help='area of sky in simulation (default: 3000 deg2)')

	import pickle
	cname = model['continuum']
	#model['continuum'] = ebossmodels.cont_models[cname]
	#model['continuum'][0][1] = (float(sys.argv[2]),0.3)
	model['emlines'] = {'scaleEWs': {'CIVb':float(sys.argv[2]),
	                                 'CIVn':float(sys.argv[2])}}
	print model
	cz = run_colorz_sim(model)
	del cz['qsos']
	sfx = "_civ_"+sys.argv[2]
	pickle.dump(cz,open(args.model+sfx+"_colorz.pkl","wb"))
