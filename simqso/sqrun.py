#!/usr/bin/env python

import numpy as np
from astropy.io import fits
from astropy.table import Table

from . import sqbase
from . import sqgrids as grids
from . import hiforest
from . import sqphoto
from .spectrum import QSOSpectrum


def buildWaveGrid(simParams):
	dispersionScale = simParams.get('DispersionScale','logarithmic')
	if dispersionScale == 'logarithmic':
		lam1,lam2 = simParams['waveRange']
		R = simParams['SpecResolution']
		wave = sqbase.fixed_R_dispersion(lam1,lam2,R)
	else:
		raise ValueError('Dispersion scale %s not supported' % dispersionScale)
	return wave


def buildMzGrid(gridPars):
	'''
	Create a grid of points in (M,z) space, each of these points are
	assigned to an individual simulated quasar.
	'''
	try:
		gridType = gridPars['GridType']
	except KeyError:
		raise ValueError('Must specify a GridType')
	np.random.seed(gridPars.get('MzSeed'))
	if gridType == 'LFSampledGrid':
		try:
			lfType = gridPars['LuminosityFunction']
		except KeyError:
			raise ValueError('Must specify a parameterization of the LF')
		Mz = grids.LFsampledPoints(gridPars['zRange'],gridPars['zRange'],
		                           lfType,**gridPars.get('LFparams',{}))
		print 'got ',Mz.zgrid.shape,' (M,z) points'
		return Mz
	elif gridType == 'LuminosityRedshiftGrid':
		Mz = grids.LuminosityRedshiftGrid(np.arange(*gridPars['mRange']),
		                                  np.arange(*gridPars['zRange']),
		                                  gridPars['nPerBin'],
		                                  lumUnits=gridPars.get('LumUnits',
		                                                        'M1450'))
	elif gridType == 'FluxRedshiftGrid':
		Mz = grids.FluxRedshiftGrid(np.arange(*gridPars['mRange']),
		                            np.arange(*gridPars['zRange']),
		                            gridPars['nPerBin'],
		                            obsBand=gridPars.get('ObsBand','i'),
		                            restBand=gridPars.get('RestBand','1450'))
	elif gridType == 'FixedGrid':
		Mz = grids.FixedMzGrid(gridPars['fixed_M'],gridPars['fixed_z'])
	else:
		raise ValueError('GridType %s unknown' % gridType)
#	if gridPars.get('Mz_LFtransfer',False):
#		print 'transferring uniform grid to LF-sampled grid...'
#		qlf = gridPars.get('Mz_QLF','Richards06')
#		Mz = qsogrid.MzGrid_QLFresample(Mz,gridPars['Mz_i_min'],
#		                                gridPars['Mz_i_max'],qlf)
#		print 'done!'
	return Mz




def buildForest(wave,z,forestParams):
	'''
    Create a set of absorbers for a given number of lines-of-sight, 
	sampled according to the input forest model. Then calculate the
	transmission along each line of sight. The input redshifts correspond
	to individual QSOs. The number of LOSs is generally smaller so that
	fewer forest computations are needed; individual LOSs are built up
	in redshift steps as each QSO redshift is iterated.
	'''
#	if kwargs.get('verbose',False):
#		print 'seeding forest generator to ',kwargs['forestSeed']
	np.random.seed(forestParams.get('ForestSeed'))
	forestType = forestParams.get('ForestType','Sightlines')
	nlos = forestParams.get('NumLinesOfSight',-1)
	if forestType == 'OneToOne':
		nlos = -1
	if forestType in ['Sightlines','OneToOne']:
		forestSpec = hiforest.generate_N_spectra(wave,z,nlos,**forestParams)
	elif forestType == 'Grid':
		zbins = np.arange(*forestParams['GridzBins'])
		tgrid = hiforest.generate_grid_spectra(wave,zbins,nlos,**forestParams)
		forestSpec = hiforest.generate_spectra_from_grid(wave,z,tgrid,**kwargs)
	return forestSpec


def buildContinuumModels(Mz,continuumParams):
	slopes = continuumParams['PowerLawSlopes'][::2]
	breakpts = continuumParams['PowerLawSlopes'][1::2]
	print '... building continuum grid'
	np.random.seed(continuumParams.get('RandomSeed'))
	if continuumParams['ContinuumModel'] == 'GaussianPLawDistribution':
		meanSlopes = [s[0] for s in slopes]
		stdSlopes = [s[1] for s in slopes]
		continuumGrid = grids.GaussianPLContinuumGrid(Mz.Mgrid,Mz.zgrid,
		                                     meanSlopes,stdSlopes,breakpts)
	elif continuumParams['ContinuumModel'] == 'FixedPLawDistribution':
		continuumGrid = grids.FixedPLContinuumGrid(Mz.Mgrid,Mz.zgrid,
		                                           slopes,breakpts)
	else:
		raise ValueError
	return continuumGrid



def buildQSOspectra(wave,Mz,forest,photoMap,qsoParams,
                    featuresIn=None,saveSpectra=False):
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
		spectra = []
	else:
		spectra = None
	nforest = len(forest['wave'])
	assert np.all(np.abs(forest['wave']-wave[:nforest]<1e-3))
	continua = buildContinuumModels(Mz,qsoParams['ContinuumParams'])
	features = None #buildFeatures(qsoParams,continua)
	spec = QSOSpectrum(wave)
	gridShape = Mz.Mgrid.shape
	synMags = np.zeros(gridShape+(len(photoMap['bandpasses']),))
	synFluxes = np.zeros_like(synMags)
	photoCache = sqphoto.getPhotoCache(wave,photoMap)
	for M,z,idx in Mz:
		spec.setRedshift(z)
		spec.setPowerLawContinuum(continua.get(idx),
		                          fluxNorm={'wavelength':1450.,'M_AB':M,
		                                    'DM':Mz.distMod})
		T = forest['T'][np.ravel_multi_index(idx,gridShape)]
		spec.f_lambda[:nforest] *= T
		synMags[idx],synFluxes[idx] = sqphoto.calcSynPhot(spec,photoMap,
		                                                  photoCache,
		                                                  synMags[idx],
		                                                  synFluxes[idx])
	return dict(synMags=synMags,synFluxes=synFluxes,
	            continua=continua,features=features,spectra=spectra)


def timerLog(action):
	pass

def readSimulationData(fileName):
	qsoData = fits.getdata(fileName+'.fits',1)
	return Table(qsoData)

def initGridData(simParams,Mz):
	return Table({'M':Mz.Mgrid,'z':Mz.zgrid})

def initSimulationData(simParams,gridData,features):
	featureData = [feature.getTable() for feature in features]
	return Table(gridData + featureData)

def writeSimulationData(simParams,dataTab):
	# XXX write params to header
	# XXX write feature data to extensions
	fileName = simParams['FileName']
	dataTab.write(fileName+'.fits')


def qsoSimulation(simParams,saveSpectra=False,
                  forestOnly=False,onlyMap=False,noPhotoMap=False):
	'''
	Run a complete simulation.
	1) Construct (M,z) grid of QSOs.
	2) Generate Lyman forest transmission spectra from a subsample of 
	   random LOSs.
	3) Sample QSO spectral features (continuum, emission lines, dust).
	4) Build simulated spectra and derive photometry.
	5) Transfer the simulated photometry to observed photometry by 
	   calculating errors and folding them in.
	'''
	#
	# build or restore the grid of (M,z) for each QSO
	#
	wave = buildWaveGrid(simParams)
	timerLog('StartSimulation')
	try:
		# if simulation data already exists, load it and skip to photomap
		qsoData = readSimulationData(simParams['FileName'])
		Mz = grids.MzGridFromData(qsoData,simParams['GridParams'])
	except IOError:
		print simParams['FileName']+' output not found'
		if 'GridFileName' in simParams:
			print 'restoring MzGrid from ',simParams['GridFileName']
			try:
				gridData = fits.getdata(simParams['GridFileName']+'.fits')
				Mz = qsogrid.MzGridFromData(gridData,simParams['GridParams'])
			except IOError:
				print simParams['GridFileName'],' not found, generating'
				Mz = buildMzGrid(simParams['GridParams'])
				gridData = initGridData(simParams,Mz)
				writeGridData(simParams['GridFileName'],gridData)
		else:
			print 'generating Mz grid'
			Mz = buildMzGrid(simParams['GridParams'])
		if not forestOnly:
			gridData = initGridData(simParams,Mz)
			writeSimulationData(simParams,gridData)
	Mz.setCosmology(simParams.get('Cosmology'))
	#
	# get the forest transmission spectra, or build if needed
	#
	if not onlyMap:
		timerLog('BuildForest')
		forestFn = simParams['ForestParams']['FileName']
		try:
			print 'loading forest ',
			forest = hiforest.load_spectra(forestFn)
		except IOError:
			print '... not found, generating forest'
			forest = buildForest(wave,Mz.getRedshifts(),
			                     simParams['ForestParams'])
			hiforest.save_spectra(forest,forestFn)
			print 'done!'
		if '_scaleTransmission' in simParams:
			# a debugging hack to rescale the forest transmission levels
			forest['T'] *= simParams['scaleTransmission']
			np.clip(forest['T'],0,1,out=forest['T'])
	if forestOnly:
		timerLog('Finish')
		return
	#
	# Use continuum and emission line distributions to build the components
	# of the intrinsic QSO spectrum, then calculate photometry
	#
	photoMap = sqphoto.load_photo_map(simParams['PhotoMapParams'])
	if not onlyMap:
		timerLog('BuildQuasarSpectra')
		simQSOs = buildQSOspectra(wave,Mz,forest,photoMap,
		                          simParams['QuasarModelParams'],
		                          saveSpectra=saveSpectra)
		qsoData = initSimulationData(simParams,gridData,simQSOs)
	#
	# map the simulated photometry to observed values with uncertainties
	#
	if not noPhotoMap:
		print 'mapping photometry'
		timerLog('PhotoMap')
		qsoData = photoMap.mapObserved(qsoData)
	timerLog('Finish')
	qsotab.writeto(simParams['FileName']+'.fits',clobber=True)

