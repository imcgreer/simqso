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

def BOSS_DR9_PLE(which=1):
	if which==1:
		row = -1.16,-3.37,-22.85,1.241,-0.249,-5.96
	alpha,beta,MStar_i_z0,k1,k2,logPhiStar = row
	MStar1450_z0 = MStar_i_z0 + 0.890
	MStar = lumfun.PolyEvolParam([-2.5*k2,-2.5*k1,MStar1450_z0])
	return lumfun.DoublePowerLawLF(logPhiStar,MStar,alpha,beta,
	                               cosmo=dr9cosmo,kcorr=def_kcorr)

def BOSS_DR9_LEDE():
	c1,c2 = -0.689, -0.809
	logPhiStar_z2_2 = -5.83
	MStar_i_z2_2 = -26.49
	MStar1450_z0 = MStar_i_z2_2 + 1.486 # --> M1450
	MStar = lumfun.PolyEvolParam([c2,MStar1450_z0],z0=2.2)
	logPhiStar = lumfun.PolyEvolParam([c1,logPhiStar_z2_2],z0=2.2)
	alpha = -1.31
	beta = -3.45
	return lumfun.DoublePowerLawLF(logPhiStar,MStar,alpha,beta,
	                               cosmo=dr9cosmo,kcorr=def_kcorr)

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

def sample_qlf(qlf,mrange=(17,22),zrange=(0.9,4.0),skyArea=1000):
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

def boss_dr9_model_vars(qsoGrid,wave):
	fetempl = grids.VW01FeTemplateGrid(qsoGrid.z,wave,
	                                   scales=sqmodels.BossDr9_FeScalings)
	mvars = [ sqmodels.BossDr9_fiducial_continuum,
	          sqmodels.BossDr9_EmLineTemplate(qsoGrid.absMag),
	          grids.FeTemplateVar(fetempl) ]
	qsoGrid.addVars(mvars)
	return qsoGrid

photSys = [ ('SDSS','Legacy'), ('UKIRT','UKIDSS_LAS'), ('WISE','AllWISE') ]

def runsim(model,fileName,forestFile,qsoGrid=None,qlf=None,
           maxIter=2,nproc=1,medianforest=False,const=False,nophot=False):
	np.random.seed(12345)
	if nproc==1:
		procMap = map
	else:
		pool = multiprocessing.Pool(nproc)
		procMap = pool.map
	wave = fixed_R_dispersion(0.3e4,6e4,500)
	#
	if qsoGrid is None:
		if qlf is None:
			qlf = BOSS_DR9_LEDE()
		qsoGrid = sample_qlf(qlf)
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
			if which=='syn':
				plt.fill_between(zbins,colorz[which][0,j],colorz[which][2,j],
				                 color=c,alpha=0.3)
				plt.plot(zbins,colorz[which][1,j],c=c,ls='-')
			else:
				for i,pval in enumerate(pvals):
					ls = '-' if pval==50 else '--'
					plt.plot(zbins,colorz[which][i,j],c=c,ls=ls)
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

qso_models = {
  'bossdr9':{'continuum':'bossdr9','emlines':'bossdr9','iron':'def_iron'},
  'dr9expdust':{'continuum':'dr9expdust','emlines':'bossdr9',
                'iron':'def_iron','dustext':'dr9expdust'},
  'new':{'continuum':'def_plcontinuum','emlines':'bossdr9',
         'dustem':'LR17','iron':'def_iron'},
}

if __name__=='__main__':
	forestFile = 'sdss_forest_grid.fits'
	if not os.path.exists(forestFile):
		make_forest_grid()
	fileName = 'ebosscore'
	if len(sys.argv) > 1:
		modelname = sys.argv[1]
	else:
		modelname = 'bossdr9'
	model = qso_models[modelname]
	const = False
	nproc = 7
	qsos = None
	if False:
		runsim(model,fileName,forestFile,const=const,nproc=nproc,qsoGrid=qsos)
		apply_selection_fun(fileName+'.fits',verbose=1,redo=True)
	else:
		import pickle
		cz = run_colorz_sim(model)
		del cz['qsos']
		pickle.dump(cz,open(modelname+"_colorz.pkl","wb"))
