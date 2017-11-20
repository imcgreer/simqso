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

dr9cosmo = FlatLambdaCDM(70,1-0.7,name='BOSSDR9')

def def_kcorr(z):
	return continuum_kcorr('SDSS-i',1450,z)

def make_forest_grid(forestFile,nlos=1200,outputDir='.',nproc=6,**kwargs):
	photSys = [('SDSS','Legacy','ugri')]
	zbins = kwargs.pop('zBins',np.arange(0.9,4.6,0.01))
	waverange = kwargs.pop('waverange',(2900.,9000))
	R = kwargs.pop('R',300)
	forestModel = sqmodels.WP11_model
	photoMap = sqphoto.LazyPhotoMap(photSys)
	hiforest.generate_grid_forest(forestFile,forestModel,
	                              nlos,zbins,waverange,R,
	                              photoMap,outputDir=outputDir,
	                              nproc=nproc,**kwargs)

def sample_qlf(qlf,mrange=(17,22),zrange=(0.9,4.0),skyArea=3000):
	obsBand = 'SDSS-r'
	kcorr = lambda z: sqbase.continuum_kcorr(obsBand,1450,z)
	m2M = lambda z: sqbase.mag2lum(obsBand,1450,z,qlf.cosmo)
	m,z = grids.generateQlfPoints(qlf,mrange,zrange,obsBand,
	                              kcorr=kcorr,skyArea=skyArea,
	                              fast_sample=True)
	qlfGrid = grids.QsoSimPoints([m,z],cosmo=qlf.cosmo,units='flux')
	qlfGrid.addVar(grids.AbsMagVar(grids.FixedSampler(
	                               qlfGrid.appMag-m2M(qlfGrid.z))))
	return qlfGrid

photSys = [ ('SDSS','Legacy'), ('UKIRT','UKIDSS_LAS'), ('WISE','AllWISE') ]

def runsim(model,fileName,forestFile,qsoGrid,
           maxIter=2,nproc=1,medianforest=False,const=False,nophot=False):
	np.random.seed(12345)
	if nproc==1:
		procMap = map
	else:
		pool = multiprocessing.Pool(nproc)
		procMap = pool.map
	wave = fixed_R_dispersion(0.3e4,6e4,500)
	#
	qsoGrid = ebossmodels.add_continuum(qsoGrid,model['continuum'],
	                                    const=const)
	if 'dustem' in model:
		qsoGrid = ebossmodels.add_dust_emission(qsoGrid,model['dustem'],
		                                        const=const)
	if 'emlines' in model:
		qsoGrid = ebossmodels.add_emission_lines(qsoGrid,model['emlines'],
		                                         const=const)
	if 'iron' in model:
		qsoGrid = ebossmodels.add_iron(qsoGrid,wave,model['iron'],
		                               const=const)
	if 'dustext' in model:
		qsoGrid = ebossmodels.add_dust_extinction(qsoGrid,model['dustext'],
		                                          const=const)
	#
	qsoGrid.loadPhotoMap(photSys)
	#
	forest = hiforest.GridForest(forestFile,qsoGrid.photoBands,
	                             median=medianforest)
	forestVar = grids.SightlineVar(forest)
	qsoGrid.addVar(forestVar)
	#
	qsoGrid,spectra = sqrun.buildSpectraBulk(wave,qsoGrid,
	                                         procMap=procMap,
	                                         maxIter=maxIter,
	                                         verbose=5)
	#
	if not nophot:
		photoData = sqphoto.calcObsPhot(qsoGrid.synFlux,qsoGrid.photoMap)
		qsoGrid.addData(photoData)
	#
	if fileName is not None:
		qsoGrid.write(fileName)
	if nproc>1:
		pool.close()
	return qsoGrid

class BandIndexes(object):
	def __init__(self,simQsoTab):
		self.bands = simQsoTab.meta['OBSBANDS'].split(',')
		self.shortNames = [ b.split('-')[-1] for b in self.bands ]
	def __call__(self,b):
		return self.shortNames.index(b)

def apply_selection_fun(fileName,verbose=0,redo=False):
	xdFile = 'xdtmp.fits'
	qsos = Table.read(fileName)
	if 'PQSO' not in qsos.colnames or redo:
		if not os.path.exists(xdFile):
			cmd = ["idl","-e","xdprob,'{0}','{1}'".format(fileName,xdFile)]
			if verbose:
				print "executing "+" ".join(cmd)
			subprocess.call(cmd)
		xdqso = Table.read(xdFile)
		if redo and 'PQSO' in qsos.colnames:
			qsos.remove_columns(xdqso.colnames)
		qsos = hstack([qsos,xdqso])
	b = BandIndexes(qsos)
	# populate the fields needed for the mid-IR-optical color cuts
	qsos['f_opt'] = ( 1.0*qsos['obsFlux'][:,b('g')] +
	                  0.8*qsos['obsFlux'][:,b('r')] +
	                  0.6*qsos['obsFlux'][:,b('i')] ) / 2.4
	# need conversion to Vega for WISE
	f_W1_Vega = qsos['obsFlux'][:,b('W1')] * 10**(0.4*(2.699))
	f_W2_Vega = qsos['obsFlux'][:,b('W2')] * 10**(0.4*(3.339))
	qsos['f_WISE'] = ( 1.0*f_W1_Vega + 0.5*f_W2_Vega ) / 1.5
	#
	sel = True
	if verbose:
		print "{:7d} quasars at start".format(len(qsos))
	# roughly the FIBER2MAG i>17 cut
	sel &= qsos['obsMag'][:,b('i')] > 17
	if verbose:
		print "{:7d} after fiberMag_i>17 cut".format(sel.sum())
	# g < 22 OR r < 22
	sel &= ( (qsos['obsMag'][:,b('g')] < 22) | 
	         (qsos['obsMag'][:,b('r')] < 22) )
	if verbose:
		print "{:7d} after g<22 OR r<22 cut".format(sel.sum())
	# XDQSOz probability cut
	sel &= qsos['PQSO'] > 0.2
	if verbose:
		print "{:7d} after XDQSOz cut".format(sel.sum())
	# optical--mid-IR color cut
	sel &= ( qsos['f_WISE']*qsos['obsFlux'][:,b('g')] >
	          qsos['f_opt']*qsos['obsFlux'][:,b('i')]*10**(3.0/2.5) )
	if verbose:
		print "{:7d} after optical--mid-IR color cut".format(sel.sum())
	#
	qsos['selected'] = sel
	qsos.write(fileName,overwrite=True)
	os.remove(xdFile)

qso_models = {
  'bossdr9':{'continuum':'bossdr9','emlines':'bossdr9','iron':'def_iron'},
  'dr9expdust':{'continuum':'dr9expdust','emlines':'bossdr9',
                'iron':'def_iron','dustext':'dr9expdust'},
  'new':{'continuum':'def_plcontinuum','emlines':'bossdr9',
         'dustem':'LR17','iron':'def_iron'},
}

def qlf_ranges(model,simName,forestFile,qlf,skyArea,**kwargs):
	np.random.seed(12345)
	for dex in [0.2,0.5,1.0]:
		for sgn in [-1,1]:
			fn = simName+'_logphi_%+04.1f'%(sgn*dex)
			print 'running simulation {}'.format(fn)
			initpar = qlf.getpar()
			par = initpar.copy()
			par[1] += sgn*dex
			qsos = sample_qlf(qlf,skyArea=skyArea)
			runsim(model,fn,forestFile,qsos,**kwargs)
			apply_selection_fun(fn+'.fits',verbose=1,redo=True)

if __name__=='__main__':
	import argparse
	parser = argparse.ArgumentParser(
	                          description='run eboss quasar simulations.')
	parser.add_argument('output',type=str,help='output file name')
	parser.add_argument('--forest',type=str,default='sdss_forest_grid.fits',
	    help='file containing forest grid (default:sdss_forest_grid.fits')
	parser.add_argument('-m','--model',type=str,default='bossdr9',
	    help='name of quasar model')
	parser.add_argument('-p','--processes',type=int,default=7,
	    help='number of processes to create')
	parser.add_argument('--qlf',type=str,default='bossdr9',
	    help='seed QLF model (default: BOSS DR9 PLE-LEDE)')
	parser.add_argument('--skyarea',type=float,default=3000,
	    help='area of sky in simulation (default: 3000 deg2)')
	parser.add_argument('--noselection',action='store_true',
	    help='do not calculation selection function')
	parser.add_argument('--testranges',action='store_true',
	    help='test range of parameter values')
	args = parser.parse_args()
	if not os.path.exists(args.forest):
		print 'forest file {} does not exist, generating...'.format(
		                                                        args.forest)
		make_forest_grid(args.forest)
	model = qso_models[args.model]
	if args.qlf=='bossdr9':
		qlf = sqmodels.BOSS_DR9_PLEpivot(cosmo=dr9cosmo)
	else:
		raise ValueError
	if args.testranges:
		qlf_ranges(model,args.output,args.forest,qlf,args.skyarea,
		           nproc=args.processes)
	else:
		np.random.seed(12345)
		qsoGrid = sample_qlf(qlf,skyArea=args.skyarea)
		runsim(model,args.output,args.forest,qsoGrid,nproc=args.processes)
	if not args.noselection:
		apply_selection_fun(args.output+'.fits',verbose=1,redo=True)
