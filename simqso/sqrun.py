#!/usr/bin/env python

import os
import ast
from copy import copy
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
	np.random.seed(gridPars.get('RandomSeed',simParams.get('RandomSeed')))
	#
	if gridType.endswith('RedshiftGrid'):
		m1,m2,nm = gridPars['mRange']
		z1,z2,nz = gridPars['zRange']
		mSampler = grids.GridSampler(m1,m2,nbins=nm)
		zSampler = grids.GridSampler(z1,z2,nbins=nz)
		if gridType.startswith('Luminosity'):
			m = grids.AbsMagVar(mSampler,restWave=gridPars['LumUnits'])
			units = 'luminosity'
		elif gridType.startswith('Flux'):
			m = grids.AppMagVar(mSampler,band=gridPars['ObsBand'],
			                    restBand=gridPars['RestBand'])
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
		                           units=units,cosmo=cosmo)
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
	np.random.seed(forestParams.get('RandomSeed',
	               simParams.get('RandomSeed')))
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
	np.random.seed(continuumParams.get('RandomSeed',
	               simParams.get('RandomSeed')))
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
	else:
		raise ValueError
	qsoGrid.addVars(continuumVars)



def buildEmissionLineGrid(qsoGrid,simParams):
	emLineParams = simParams['QuasarModelParams']['EmissionLineParams']
	np.random.seed(emLineParams.get('RandomSeed',simParams.get('RandomSeed')))
#	try:
#		# if the user has passed in a model, instantiate it
#		emLineGrid = emLineParams['EmissionLineModel'](Mz.mGrid,Mz.zGrid,
#		                                               **emLineParams)
#		return emLineGrid
#	except TypeError:
#		pass
	# otherwise construct a model from the existing set
	if emLineParams['EmissionLineModel'] == 'FixedVdBCompositeLines':
		emLineGrid = grids.generateVdBCompositeEmLines(
		                             minEW=emLineParams.get('minEW',1.0),
		                             noFe=emLineParams.get('VdB_noFe',False))
#		# XXX hacky
#		if emLineParams.get('addSBB',False):
#			emLineGrid.addSBB()
#	elif emLineParams['EmissionLineModel'] == 'FixedLBQSEmissionLines':
#		emLineGrid = qsotemplates.FixedLBQSemLineGrid(
#		                                noFe=emLineParams.get('LBQSnoFe',False))
	elif emLineParams['EmissionLineModel'] == 'VariedEmissionLineGrid':
		emLineGrid = grids.generateBEffEmissionLines(qsoGrid.absMag,
		                                             **emLineParams)
	else:
		raise ValueError('invalid emission line model: ' +
		                    emLineParams['EmissionLineModel'])
	qsoGrid.addVar(emLineGrid)

def buildDustGrid(qsoGrid,simParams):
	print '... building dust extinction grid'
	dustParams = simParams['QuasarModelParams']['DustExtinctionParams']
	np.random.seed(dustParams.get('RandomSeed',simParams.get('RandomSeed')))
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


def buildFeatures(Mz,wave,simParams):
	buildContinuumModels(Mz,simParams)
	qsoParams = simParams['QuasarModelParams']
	if 'EmissionLineParams' in qsoParams:
		buildEmissionLineGrid(Mz,simParams)
	if 'IronEmissionParams' in qsoParams:
		# only option for now is the VW01 template
		scalings = qsoParams['IronEmissionParams'].get('FeScalings')
		feGrid = grids.VW01FeTemplateGrid(Mz.z,wave,scales=scalings)
		Mz.addVar(grids.FeTemplateVar(feGrid))
	if 'DustExtinctionParams' in qsoParams:
		buildDustGrid(Mz,simParams)


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
	fluxNorm = {'wavelength':1450.,'M_AB':None,'DM':qsoGrid.distMod}
	for iterNum in range(nIter):
		print 'buildQSOspectra iteration ',iterNum+1,' out of ',nIter
		for i,obj in enumerate(qsoGrid):
			spec.setRedshift(obj['z'])
			# start with continuum
			fluxNorm['M_AB'] = obj['absMag']
			for feature in qsoGrid.getVars(grids.ContinuumVar):
				par = obj[feature.name]
				spec = feature.add_to_spec(spec,par,fluxNorm=fluxNorm)
			# add emission (multiplicative) features
			emspec = Spectrum(wave,z=obj['z'])
			for feature in qsoGrid.getVars(grids.EmissionFeatureVar):
				if isinstance(feature.sampler,grids.NullSampler):
					par = None
				else:
					par = obj[feature.name]
				emspec += feature.add_to_spec(emspec,par)
			spec *= emspec + 1
			# add any remaining features
			for feature in qsoGrid.getVars(grids.SpectralFeatureVar):
				if isinstance(feature,grids.ContinuumVar) or \
				   isinstance(feature,grids.EmissionFeatureVar):
					continue
				if isinstance(feature.sampler,grids.NullSampler):
					par = None
				else:
					par = obj[feature.name]
				feature.add_to_spec(spec,par)
			# apply HI forest blanketing
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
	if retParams:
		hdr = fits.getheader(os.path.join(outputDir,fileName+'.fits'),1)
		simPars = ast.literal_eval(hdr['SQPARAMS'])
		# XXX get it from parameters...
###		simPars['Cosmology'] = {
###		  'WMAP9':cosmology.WMAP9,
###		}[simPars['Cosmology']]
		return qsoGrid,simPars
	return qsoGrid

def writeSimulationData(simParams,Mz,outputDir):
	# Primary extension just contains model parameters in header
	simPar = copy(simParams)
	# XXX need to write parameters out or something...
	simPar['Cosmology'] = simPar['Cosmology'].name
	try:
		del simPar['GridParams']['QLFmodel']
	except:
		pass
	Mz.data.meta['SQPARAMS'] = str(simPar)
	Mz.data.write(os.path.join(outputDir,simPar['FileName']+'.fits'),
	         overwrite=True)


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
	timerLog = TimerLog()
	try:
		# simulation data already exists, load the Mz grid
		cosmo = simParams['Cosmology'] # XXX
		try:
			# XXX a hack until figuring out how to save this in header
			qlf = simParams['GridParams']['QLFmodel']
		except:
			qlf = None
		Mz,simParams = readSimulationData(simParams['FileName'],
		                                  outputDir,retParams=True)
		# XXX hack copy back in
		simParams['GridParams']['QLFmodel'] = qlf
		simParams['Cosmology'] = cosmo
	except IOError:
		print simParams['FileName']+' output not found'
		if 'GridFileName' in simParams:
			print 'restoring MzGrid from ',simParams['GridFileName']
			try:
				Mz,simParams = readSimulationData(simParams['GridFileName'],
				                                  outputDir,retParams=True)
			except IOError:
				print simParams['GridFileName'],' not found, generating'
				Mz = buildMzGrid(simParams)
				writeGridData(simParams,Mz,outputDir)
		else:
			print 'generating Mz grid'
			Mz = buildMzGrid(simParams)
		if not forestOnly:
			if not noWriteOutput and 'GridFileName' in simParams:
				writeGridData(simParams,Mz,outputDir)
	Mz.setCosmology(simParams.get('Cosmology'))
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
			forest = buildForest(wave,Mz.z,simParams,outputDir)
		# make sure that the forest redshifts actually match the grid
		assert np.allclose(forest['z'],Mz.z)
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
		_,spectra = buildQSOspectra(wave,Mz,forest,photoMap,simParams,
		                          maxIter=simParams.get('maxFeatureIter',5),
		                          saveSpectra=saveSpectra)
	timerLog('Build Quasar Spectra')
	#
	# map the simulated photometry to observed values with uncertainties
	#
	if not noPhotoMap:
		print 'mapping photometry'
		np.random.seed(simParams['PhotoMapParams'].get('RandomSeed',
		               simParams.get('RandomSeed')))
		photoData = sqphoto.calcObsPhot(Mz.synFlux,photoMap)
		timerLog('PhotoMap')
	else:
		photoData = None
	timerLog.dump()
	if not noWriteOutput:
		writeSimulationData(simParams,Mz,outputDir)
	if saveSpectra:
		fits.writeto(os.path.join(outputDir,
		                          simParams['FileName']+'_spectra.fits'),
		             spectra,overwrite=True)

def load_spectra(simFileName,outputDir='.'):
	simdat,par = readSimulationData(simFileName,outputDir,retParams=True)
	sp = fits.getdata(os.path.join(outputDir,simFileName+'_spectra.fits'))
	wave = buildWaveGrid(par)
	return wave,simdat,sp

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

