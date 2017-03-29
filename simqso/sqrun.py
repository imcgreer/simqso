#!/usr/bin/env python

import os
import time
import numpy as np
from astropy.io import fits
from astropy.table import Table,hstack
from astropy import cosmology

from . import sqbase
from . import sqgrids as grids
from .spectrum import Spectrum
from . import hiforest
from . import dustextinction
from . import sqphoto


def buildWaveGrid(simParams):
	dispersionScale = simParams.get('DispersionScale','logarithmic')
	if dispersionScale == 'logarithmic':
		lam1,lam2 = simParams['waveRange']
		R = simParams['SpecDispersion']
		wave = sqbase.fixed_R_dispersion(lam1,lam2,R)
	else:
		raise ValueError('Dispersion scale %s not supported' % dispersionScale)
	return wave


def reseed(par):
	try:
		np.random.seed(par['RandomSeed'])
	except KeyError:
		pass


def buildMzGrid(simParams):
	'''
	Create a grid of points in (M,z) space, each of these points are
	assigned to an individual simulated quasar.
	'''
	cosmodef = simParams.get('Cosmology')
	gridPars = simParams['GridParams']
	try:
		gridType = gridPars['GridType']
	except KeyError:
		raise ValueError('Must specify a GridType')
	m2M = lambda z: sqbase.mag2lum('SDSS-i',1450,z,cosmodef) # XXX
	reseed(gridPars)
	#
	if gridType.endswith('RedshiftGrid'):
		m1,m2,nm = gridPars['mRange']
		z1,z2,nz = gridPars['zRange']
		if type(nm) is int:
			mSampler = grids.GridSampler(m1,m2,nbins=nm)
		else:
			mSampler = grids.GridSampler(m1,m2,stepsize=nm)
		if type(nz) is int:
			zSampler = grids.GridSampler(z1,z2,nbins=nz)
		else:
			zSampler = grids.GridSampler(z1,z2,stepsize=nz)
		if gridType.startswith('Luminosity'):
			m = grids.AbsMagVar(mSampler,restWave=gridPars['LumUnits'])
			units = 'luminosity'
		elif gridType.startswith('Flux'):
			m = grids.AppMagVar(mSampler,band=gridPars['ObsBand'],
			                    )#restBand=gridPars['RestBand'])
			units = 'flux'
		z = grids.RedshiftVar(zSampler)
	elif gridType == 'FixedGrid':
		m = grids.FixedSampler(gridPars['fixed_M'])
		z = grids.FixedSampler(gridPars['fixed_z'])
		# XXX units
	elif gridType == 'LuminosityFunction':
		try:
			qlf = gridPars['QLFmodel']
		except KeyError:
			raise ValueError('Must specify a parameterization of the LF')
		qsoGrid = grids.generateQlfPoints(qlf,
		                                  gridPars['mRange'],
		                                  gridPars['zRange'],
		                                  m2M,cosmodef,
		                                  gridPars['ObsBand'],
		                                  **gridPars['QLFargs'])
	else:
		raise ValueError('GridType %s unknown' % gridType)
	if gridType != 'LuminosityFunction':
		qsoGrid = grids.QsoSimGrid([m,z],gridPars['nPerBin'],
		                           units=units,cosmo=cosmodef)
	try:
		_ = qsoGrid.absMag
	except:
		absMag = qsoGrid.appMag - m2M(qsoGrid.z)
		absMag = grids.AbsMagVar(grids.FixedSampler(absMag))
		qsoGrid.addVar(absMag)
	return qsoGrid




def buildForest(wave,z,simParams,outputDir):
	'''
    Create a set of absorbers for a given number of lines-of-sight, 
	sampled according to the input forest model. Then calculate the
	transmission along each line of sight. The input redshifts correspond
	to individual QSOs. The number of LOSs is generally smaller so that
	fewer forest computations are needed; individual LOSs are built up
	in redshift steps as each QSO redshift is iterated.
	'''
	forestParams = simParams['ForestParams']
	reseed(forestParams)
	forestType = forestParams.get('ForestType','Sightlines')
	nlos = forestParams.get('NumLinesOfSight',-1)
	forestFn = forestParams['FileName']
	if forestType == 'OneToOne':
		nlos = -1
	forestSpec = None
	try:
		print 'loading forest ',forestFn
		forestSpec = hiforest.load_spectra(forestFn,outputDir)
	except IOError:
		pass
	if forestType in ['Sightlines','OneToOne']:
		if forestSpec is None:
			print '... not found, generating forest'
			forestSpec = hiforest.generate_N_spectra(wave,z,nlos,
			                                         **forestParams)
			hiforest.save_spectra(forestSpec,forestFn,outputDir)
	elif forestType == 'Grid':
		if forestSpec is None:
			raise ValueError('Need to supply a forest grid')
		forestSpec = hiforest.generate_spectra_from_grid(wave,z,forestSpec,
		                                                 **forestParams)
	print 'done!'
	return forestSpec


def buildContinuumModels(qsoGrid,simParams):
	continuumParams = simParams['QuasarModelParams']['ContinuumParams']
	reseed(continuumParams)
	slopes = continuumParams['PowerLawSlopes'][::2]
	breakpts = continuumParams['PowerLawSlopes'][1::2]
	print '... building continuum grid'
	cmodel = continuumParams['ContinuumModel']
	if cmodel in ['GaussianPLawDistribution','FixedPLawDistribution',
	                                                'BrokenPowerLaw']:
		if cmodel in ['GaussianPLawDistribution','FixedPLawDistribution']:
			print 'WARNING: %s continuum is deprecated' % cmodel
		slopeVars = [ grids.GaussianSampler(*s) for s in slopes ]
		continuumVars = [ grids.BrokenPowerLawContinuumVar(slopeVars,
		                                                   breakpts) ]
	elif isinstance(cmodel,grids.QsoSimVar):
		continuumVars = [ cmodel ]
	else:
		raise ValueError
	qsoGrid.addVars(continuumVars)


def buildEmissionLineGrid(qsoGrid,simParams):
	emLineParams = simParams['QuasarModelParams']['EmissionLineParams']
	reseed(emLineParams)
	if emLineParams['EmissionLineModel'] == 'FixedVdBCompositeLines':
		emLineGrid = grids.generateVdBCompositeEmLines(
		                             minEW=emLineParams.get('minEW',1.0),
		                             noFe=emLineParams.get('VdB_noFe',False))
	elif emLineParams['EmissionLineModel'] == 'VariedEmissionLineGrid':
		emLineGrid = grids.generateBEffEmissionLines(qsoGrid.absMag,
		                                             **emLineParams)
	elif isinstance(emLineParams['EmissionLineModel'],grids.QsoSimVar):
		emLineGrid = emLineParams['EmissionLineModel']
	else:
		raise ValueError('invalid emission line model: ' +
		                    emLineParams['EmissionLineModel'])
	qsoGrid.addVar(emLineGrid)

def buildDustGrid(qsoGrid,simParams):
	print '... building dust extinction grid'
	dustParams = simParams['QuasarModelParams']['DustExtinctionParams']
	reseed(dustParams)
	if dustParams['DustExtinctionModel'] == 'Fixed E(B-V)':
		sampler = grids.ConstSampler(dustParams['E(B-V)'])
	elif dustParams['DustExtinctionModel']=='Exponential E(B-V) Distribution':
		sampler = grids.ExponentialSampler(dustParams['E(B-V)'])
	else:
		raise ValueError('invalid dust extinction model: '+
		                 dustParams['DustExtinctionModel'])
	if dustParams['DustModelName'] == 'SMC':
		dustVar = grids.SMCDustVar(sampler)
	elif dustParams['DustModelName'] == 'CalzettiSB':
		dustVar = grids.CalzettiDustVar(sampler)
	else:
		raise ValueError('invalid dust extinction model: '+
		                 dustParams['DustModelName'])
# XXX
#		                 fraction=dustParams.get('DustLOSfraction',1.0))
	qsoGrid.addVar(dustVar)


def buildFeatures(qsoGrid,wave,simParams):
	buildContinuumModels(qsoGrid,simParams)
	qsoParams = simParams['QuasarModelParams']
	if 'EmissionLineParams' in qsoParams:
		buildEmissionLineGrid(qsoGrid,simParams)
	if 'IronEmissionParams' in qsoParams:
		# only option for now is the VW01 template
		scalings = qsoParams['IronEmissionParams'].get('FeScalings')
		feGrid = grids.VW01FeTemplateGrid(qsoGrid.z,wave,scales=scalings)
		qsoGrid.addVar(grids.FeTemplateVar(feGrid))
	if 'DustExtinctionParams' in qsoParams:
		buildDustGrid(qsoGrid,simParams)


def buildQSOspectra(wave,qsoGrid,forest,photoMap,simParams,
                    maxIter,saveSpectra=False):
	'''
	Assemble the spectral components of each QSO from the input parameters.
	---
	Required:
	  'ContinuumModel'  :  'GaussianPLawDistribution','FixedPLawDistribution'
	---
	Optional:
	  'EmissionLineModel'   : 'None',
	                          'FixedVdBCompositeLines',
	                          'FixedLBQSEmissionLines'
	  'DustExtinctionModel' : 'None',
	                          'Fixed E(B-V)',
	                          'Exponential E(B-V) Distribution'
	'''
	if saveSpectra:
		spectra = np.zeros((qsoGrid.nObj,len(wave)))
	else:
		spectra = None
	nforest = len(forest['wave'])
	assert np.all(np.abs(forest['wave']-wave[:nforest]<1e-3))
	buildFeatures(qsoGrid,wave,simParams)
	spec = Spectrum(wave)
	synMag = np.zeros((qsoGrid.nObj,len(photoMap['bandpasses'])))
	synFlux = np.zeros_like(synMag)
	photoCache = sqphoto.getPhotoCache(wave,photoMap)
	print 'units are ',qsoGrid.units
	if qsoGrid.units == 'luminosity':
		nIter = 1
	else:
		nIter = maxIter
		# this should be pushed up to something like photoMap.getIndex(band)
		bands = photoMap['bandpasses'].keys()
		try:
			obsBand = qsoGrid.qsoVars[0].obsBand # XXX
			fluxBand = next(j for j in range(len(bands)) 
			                    if photoMap['filtName'][bands[j]]==obsBand)
		except:
			raise ValueError('band ',obsBand,' not found in ',bands)
		print 'fluxBand is ',fluxBand,bands
	#
	fluxNorm = {'wavelength':1450.,'M_AB':None,'DM':qsoGrid.distMod}
	def _getpar(feature,obj):
		if isinstance(feature.sampler,grids.NullSampler):
			return None
		else:
			return obj[feature.name]
	#
	for iterNum in range(nIter):
		print 'buildQSOspectra iteration ',iterNum+1,' out of ',nIter
		for i,obj in enumerate(qsoGrid):
			spec.setRedshift(obj['z'])
			# start with continuum
			fluxNorm['M_AB'] = obj['absMag']
			for feature in qsoGrid.getVars(grids.ContinuumVar):
				spec = feature.add_to_spec(spec,_getpar(feature,obj),
				                           fluxNorm=fluxNorm)
			# add emission (multiplicative) features
			emspec = Spectrum(wave,z=obj['z'])
			for feature in qsoGrid.getVars(grids.EmissionFeatureVar):
				emspec = feature.add_to_spec(emspec,_getpar(feature,obj))
			spec *= emspec + 1
			# add any remaining features
			for feature in qsoGrid.getVars(grids.SpectralFeatureVar):
				if isinstance(feature,grids.ContinuumVar) or \
				   isinstance(feature,grids.EmissionFeatureVar):
					continue
				spec = feature.add_to_spec(spec,_getpar(feature,obj))
			# apply HI forest blanketing # XXX move up obj.index
			spec.f_lambda[:nforest] *= forest['T'][i]
			# calculate synthetic magnitudes from the spectra through the
			# specified bandpasses
			synMag[i],synFlux[i] = sqphoto.calcSynPhot(spec,photoMap,
			                                           photoCache,
			                                           synMag[i],
			                                           synFlux[i])
			if saveSpectra:
				spectra[i] = spec.f_lambda
			spec.clear()
		if nIter > 1:
			# find the largest mag offset
			dm = synMag[...,fluxBand] - qsoGrid.appMag
			print '--> delta mag mean = %.7f, rms = %.7f, |max| = %.7f' % \
			              (dm.mean(),dm.std(),np.abs(dm).max())
			qsoGrid.absMag[:] -= dm
			dmagMax = np.abs(dm).max()
			# resample features with updated absolute mags
			qsoGrid.resample()
			if dmagMax < 0.01:
				break
	qsoGrid.addVar(grids.SynMagVar(grids.FixedSampler(synMag)))
	qsoGrid.addVar(grids.SynFluxVar(grids.FixedSampler(synFlux)))
	return qsoGrid,spectra


class TimerLog():
	def __init__(self):
		self.stages = ['StartSimulation']
		self.times = [time.time()]
	def __call__(self,stage):
		self.stages.append(stage)
		self.times.append(time.time())
	def dump(self):
		self.__call__('Finish')
		stages = self.stages[1:]
		times = np.array(self.times[1:]) - self.times[0]
		#itimes = np.concatenate([[0,],np.diff(times)]) 
		itimes = np.diff(self.times)
		ftimes = itimes / times[-1]
		print '%20s %8s %8s %8s' % ('stage','time','elapsed','frac')
		for t in zip(stages,itimes,times,ftimes):
			print '%20s %8.3f %8.3f %8.3f' % t
		print

def writeGridData(simParams,Mz,outputDir):
	simPar = copy(simParams)
	# XXX need to write parameters out or something...
	simPar['Cosmology'] = simPar['Cosmology'].name
	try:
		del simPar['GridParams']['QLFmodel']
	except:
		pass
	# XXX
	Mz.data.meta['GRIDPARS'] = str(simPar['GridParams'])
	Mz.data.write(os.path.join(outputDir,simPar['GridFileName']+'.fits'),
	              overwrite=True)

def readSimulationData(fileName,outputDir,retParams=False):
	qsoGrid = grids.QsoSimObjects()
	qsoGrid.read(os.path.join(outputDir,fileName+'.fits'))
	simPars = qsoGrid.simPars
	gridPars = simPars['GridParams']
	if True:
		mSampler = grids.FixedSampler(qsoGrid.appMag)
		m = grids.AppMagVar(mSampler,band=gridPars['ObsBand'])
	try:
		mSampler = grids.FixedSampler(qsoGrid.appMag)
		m = grids.AppMagVar(mSampler,band=gridPars['ObsBand'])
	except:
		mSampler = grids.FixedSampler(qsoGrid.absMag)
		m = grids.AbsMagVar(mSampler,restWave=gridPars['LumUnits'])
	z = grids.RedshiftVar(grids.FixedSampler(qsoGrid.z))
	qsoGrid.addVars([m,z])
	if retParams:
		return qsoGrid,simPars
	return qsoGrid


def qsoSimulation(simParams,**kwargs):
	'''
	Run a complete simulation.
	1) Construct (M,z) grid of QSOs.
	2) Generate Lyman forest transmission spectra from a subsample of 
	   random LOSs.
	3) Sample QSO spectral features (continuum, emission lines, dust).
	4) Build simulated spectra and derive photometry.
	5) Transfer the simulated photometry to observed photometry by 
	   calculating errors and folding them in.
	...
	Keyword arguments:
	  saveSpectra: save the simulated spectra, not just the photometry.
	        Beware! filt could be quite large (Nqso x Npixels) [default:False]
	  forestOnly: only generate the forest transmission spectra [default:False]
      onlyMap: only do the simulation of observed photometry, assuming 
	           synthetic photometry has already been generated [default:False]
	  noPhotoMap: skip the simulation of observed photometry [default:False]
	  outputDir: write files to this directory [default:'./']
	'''
	saveSpectra = kwargs.get('saveSpectra',False)
	forestOnly = kwargs.get('forestOnly',False)
	onlyMap = kwargs.get('onlyMap',False)
	noPhotoMap = kwargs.get('noPhotoMap',False)
	noWriteOutput = kwargs.get('noWriteOutput',False)
	outputDir = kwargs.get('outputDir','./')
	#
	# build or restore the grid of (M,z) for each QSO
	#
	wave = buildWaveGrid(simParams)
	reseed(simParams)
	timerLog = TimerLog()
	try:
		qsoGrid,simParams = readSimulationData(simParams['FileName'],
		                                       outputDir,retParams=True)
	except IOError:
		print simParams['FileName']+' output not found'
		if 'GridFileName' in simParams:
			print 'restoring grid from ',simParams['GridFileName']
			try:
				qsoGrid = readSimulationData(simParams['GridFileName'],
				                             outputDir)
			except IOError:
				print simParams['GridFileName'],' not found, generating'
				qsoGrid = buildMzGrid(simParams)
				writeGridData(simParams,qsoGrid,outputDir)
		else:
			print 'generating Mz grid'
			qsoGrid = buildMzGrid(simParams)
		if not forestOnly:
			if not noWriteOutput and 'GridFileName' in simParams:
				writeGridData(simParams,qsoGrid,outputDir)
	qsoGrid.setCosmology(simParams.get('Cosmology'))
	timerLog('Initialize Grid')
	#
	# get the forest transmission spectra, or build if needed
	#
	if not onlyMap:
		if 'ForestParams' not in simParams:
			# no forest applied, overrides T to always return one
			class NullForest(object):
				def __getitem__(self,i):
					return 1
			forest = dict(wave=wave[:2],T=NullForest())
		else:
			forest = buildForest(wave,qsoGrid.z,simParams,outputDir)
		# make sure that the forest redshifts actually match the grid
		assert np.allclose(forest['z'],qsoGrid.z)
	if forestOnly:
		timerLog.dump()
		return
	timerLog('Generate Forest')
	#
	# Use continuum and emission line distributions to build the components
	# of the intrinsic QSO spectrum, then calculate photometry
	#
	photoMap = sqphoto.load_photo_map(simParams['PhotoMapParams'])
	if not onlyMap:
		_,spectra = buildQSOspectra(wave,qsoGrid,forest,photoMap,simParams,
		                            maxIter=simParams.get('maxFeatureIter',5),
		                            saveSpectra=saveSpectra)
	timerLog('Build Quasar Spectra')
	#
	# map the simulated photometry to observed values with uncertainties
	#
	if not noPhotoMap:
		print 'mapping photometry'
		reseed(simParams['PhotoMapParams'])
		photoData = sqphoto.calcObsPhot(qsoGrid.synFlux,photoMap)
		qsoGrid.addData(photoData)
		timerLog('PhotoMap')
	timerLog.dump()
	if not noWriteOutput:
		qsoGrid.write(simParams,outputDir=outputDir)
	if saveSpectra:
		fits.writeto(os.path.join(outputDir,
		                          simParams['FileName']+'_spectra.fits'),
		             spectra,overwrite=True)

def load_spectra(simFileName,outputDir='.'):
	simdat,par = readSimulationData(simFileName,outputDir,retParams=True)
	sp = fits.getdata(os.path.join(outputDir,simFileName+'_spectra.fits'))
	wave = buildWaveGrid(par)
	qsos = hstack([simdat.data,Table(dict(spec=sp))])
	return wave,qsos

def generateForestGrid(simParams,**kwargs):
	forestParams = simParams['ForestParams']
	outputDir = kwargs.get('outputDir','./')
	try:
		tgrid = hiforest.load_spectra(forestParams['FileName'],outputDir)
		print 'grid already exists, exiting'
		return
	except IOError:
		pass
	wave = buildWaveGrid(simParams)
	zbins = np.arange(*forestParams['GridzBins'])
	nlos = forestParams['NumLinesOfSight']
	timerLog = TimerLog()
	tgrid = hiforest.generate_grid_spectra(wave,zbins,nlos,**forestParams)
	timerLog('BuildForest')
	hiforest.save_spectra(tgrid,forestParams['FileName'],outputDir)
	timerLog.dump()

