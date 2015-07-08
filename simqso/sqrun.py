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
from .spectrum import QSOSpectrum
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
	np.random.seed(gridPars.get('RandomSeed',simParams.get('RandomSeed')))
	if gridType == 'LuminosityFunction':
		try:
			qlf = gridPars['QLFmodel']
		except KeyError:
			raise ValueError('Must specify a parameterization of the LF')
		Mz = grids.LuminosityFunctionFluxGrid(gridPars,qlf,cosmodef,
		                                      **gridPars['QLFargs'])
	elif gridType == 'LuminosityRedshiftGrid':
		Mz = grids.LuminosityRedshiftGrid(gridPars,cosmodef)
	elif gridType == 'FluxRedshiftGrid':
		Mz = grids.FluxRedshiftGrid(gridPars,cosmodef)
	elif gridType == 'FixedGrid':
		Mz = grids.FixedMzGrid(gridPars['fixed_M'],gridPars['fixed_z'])
	else:
		raise ValueError('GridType %s unknown' % gridType)
	if gridPars.get('LFSampledGrid',False):
		print 'transferring uniform grid to LF-sampled grid...'
		try:
			qlf = gridPars['QLFmodel']
		except KeyError:
			raise ValueError('Must specify a parameterization of the LF')
		Mz = grids.MzGrid_QLFresample(Mz,qlf)
		print 'done!'
	return Mz




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


def buildContinuumModels(Mz,simParams):
	continuumParams = simParams['QuasarModelParams']['ContinuumParams']
	np.random.seed(continuumParams.get('RandomSeed',
	               simParams.get('RandomSeed')))
	slopes = continuumParams['PowerLawSlopes'][::2]
	breakpts = continuumParams['PowerLawSlopes'][1::2]
	print '... building continuum grid'
	if continuumParams['ContinuumModel'] == 'GaussianPLawDistribution':
		meanSlopes = [s[0] for s in slopes]
		stdSlopes = [s[1] for s in slopes]
		continuumGrid = grids.GaussianPLContinuumGrid(Mz.mGrid,Mz.zGrid,
		                                     meanSlopes,stdSlopes,breakpts)
	elif continuumParams['ContinuumModel'] == 'FixedPLawDistribution':
		continuumGrid = grids.FixedPLContinuumGrid(Mz.mGrid,Mz.zGrid,
		                                           slopes,breakpts)
	else:
		raise ValueError
	return continuumGrid



def buildEmissionLineGrid(Mz,simParams):
	emLineParams = simParams['QuasarModelParams']['EmissionLineParams']
	np.random.seed(emLineParams.get('RandomSeed',simParams.get('RandomSeed')))
	if emLineParams['EmissionLineModel'] == 'FixedVdBCompositeLines':
		emLineGrid = grids.FixedVdBcompositeEMLineGrid(Mz.mGrid,Mz.zGrid,
		                             minEW=emLineParams.get('minEW',1.0),
		                             noFe=emLineParams.get('VdB_noFe',False))
		# XXX hacky
		if emLineParams.get('addSBB',False):
			emLineGrid.addSBB()
#	elif emLineParams['EmissionLineModel'] == 'FixedLBQSEmissionLines':
#		emLineGrid = qsotemplates.FixedLBQSemLineGrid(
#		                                noFe=emLineParams.get('LBQSnoFe',False))
	elif emLineParams['EmissionLineModel'] == 'VariedEmissionLineGrid':
		emLineGrid = grids.VariedEmissionLineGrid(Mz.mGrid,Mz.zGrid,
		                                          **emLineParams)
	else:
		raise ValueError('invalid emission line model: ' +
		                    emLineParams['EmissionLineModel'])
	if 'addLines' in emLineParams:
		for l in emLineParams['addLines']:
			print 'adding line ',l
			emLineGrid.addLine(*l)
	return emLineGrid

def buildDustGrid(Mz,simParams):
	print '... building dust extinction grid'
	dustParams = simParams['QuasarModelParams']['DustExtinctionParams']
	np.random.seed(dustParams.get('RandomSeed',simParams.get('RandomSeed')))
	if dustParams['DustExtinctionModel'] == 'Fixed E(B-V)':
		dustGrid = grids.FixedDustGrid(Mz.mGrid,Mz.zGrid,
		                 dustParams['DustModelName'],dustParams['E(B-V)'])
	elif dustParams['DustExtinctionModel']=='Exponential E(B-V) Distribution':
		dustGrid = grids.ExponentialDustGrid(Mz.mGrid,Mz.zGrid,
		                 dustParams['DustModelName'],dustParams['E(B-V)'],
		                 fraction=dustParams.get('DustLOSfraction',1.0))
	else:
		raise ValueError('invalid dust extinction model: '+
		                 dustParams['DustExtinctionModel'])
	return dustGrid


class SpectralFeature(object):
	def __init__(self,grid):
		self.grid = grid
	def update(self,*args):
		self.grid.update(*args)
	def getTable(self,hdr):
		return self.grid.getTable(hdr)

class EmissionLineFeature(SpectralFeature):
	def apply_to_spec(self,spec,idx):
		emlines = self.grid.get(idx)
		spec.addEmissionLines(emlines)

class IronEmissionFeature(SpectralFeature):
	def apply_to_spec(self,spec,idx):
		feTemplate = self.grid.get(idx)
		spec.addTemplate('Fe',feTemplate)

class DustExtinctionFeature(SpectralFeature):
	def apply_to_spec(self,spec,idx):
		dustfn = self.grid.get(idx)
		spec.convolve_restframe(*dustfn)

def buildFeatures(Mz,wave,simParams):
	features = []
	qsoParams = simParams['QuasarModelParams']
	if 'EmissionLineParams' in qsoParams:
		emLineGrid = buildEmissionLineGrid(Mz,simParams)
		emLineFeature = EmissionLineFeature(emLineGrid)
		features.append(emLineFeature)
	if 'IronEmissionParams' in qsoParams:
		# only option for now is the VW01 template
		scalings = qsoParams['IronEmissionParams'].get('FeScalings')
		feGrid = grids.VW01FeTemplateGrid(Mz.mGrid,Mz.zGrid,wave,
		                                  scales=scalings)
		feFeature = IronEmissionFeature(feGrid)
		features.append(feFeature)
	if 'DustExtinctionParams' in qsoParams:
		dustGrid = buildDustGrid(Mz,simParams)
		dustFeature = DustExtinctionFeature(dustGrid)
		features.append(dustFeature)
		pass
	return features


def buildQSOspectra(wave,Mz,forest,photoMap,simParams,
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
		spectra = np.zeros((Mz.mGrid.size,len(wave)))
	else:
		spectra = None
	nforest = len(forest['wave'])
	assert np.all(np.abs(forest['wave']-wave[:nforest]<1e-3))
	continua = buildContinuumModels(Mz,simParams)
	features = buildFeatures(Mz,wave,simParams)
	spec = QSOSpectrum(wave)
	gridShape = Mz.mGrid.shape
	synMag = np.zeros(gridShape+(len(photoMap['bandpasses']),))
	synFlux = np.zeros_like(synMag)
	photoCache = sqphoto.getPhotoCache(wave,photoMap)
	print 'units are ',Mz.units
	if Mz.units == 'luminosity':
		nIter = 1
	else:
		nIter = maxIter
		# this should be pushed up to something like photoMap.getIndex(band)
		bands = photoMap['bandpasses'].keys()
		try:
			fluxBand = next(j for j in range(len(bands)) 
			                    if photoMap['filtName'][bands[j]]==Mz.obsBand)
		except:
			raise ValueError('band ',Mz.obsBand,' not found in ',bands)
		print 'fluxBand is ',fluxBand,bands
	for iterNum in range(nIter):
		print 'buildQSOspectra iteration ',iterNum+1,' out of ',nIter
		for M,z,idx in Mz:
			i = np.ravel_multi_index(idx,gridShape)
			spec.setRedshift(z)
			# start with continuum
			spec.setPowerLawContinuum(continua.get(idx),
			                          fluxNorm={'wavelength':1450.,'M_AB':M,
			                                    'DM':Mz.distMod})
			# add additional emission/absorption features
			for feature in features:
				feature.apply_to_spec(spec,idx)
			# apply HI forest blanketing
			spec.f_lambda[:nforest] *= forest['T'][i]
			# calculate synthetic magnitudes from the spectra through the
			# specified bandpasses
			synMag[idx],synFlux[idx] = sqphoto.calcSynPhot(spec,photoMap,
			                                               photoCache,
			                                               synMag[idx],
			                                               synFlux[idx])
			if saveSpectra and iterNum==nIter-1:
				spectra[i] = spec.f_lambda
###		print 'before: ',Mz.mGrid,synMag[...,-1]
		if nIter > 1:
			dmagMax = Mz.updateMags(synMag[...,fluxBand])
			continua.update(Mz.mGrid,Mz.zGrid)
			for feature in features:
				feature.update(Mz.mGrid,Mz.zGrid)
###			print 'after: ',Mz.mGrid,synMag[...,-1]
			if dmagMax < 0.01:
				break
	return dict(synMag=synMag,synFlux=synFlux,
	            continua=continua,features=features,spectra=spectra)


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

def initGridData(simParams,Mz):
	if Mz.units == 'flux':
		return Table({'M':Mz.mGrid.flatten(),'z':Mz.zGrid.flatten(),
		              'appMag':Mz.appMagGrid.flatten()})
	else:
		return Table({'M':Mz.mGrid.flatten(),'z':Mz.zGrid.flatten()})

def writeGridData(simParams,Mz,gridData,outputDir):
	simPar = copy(simParams)
	# XXX need to write parameters out or something...
	simPar['Cosmology'] = simPar['Cosmology'].name
	try:
		del simPar['GridParams']['QLFmodel']
	except:
		pass
	hdr0 = fits.Header()
	hdr0['GRIDPARS'] = str(simPar['GridParams'])
	hdr0['GRIDUNIT'] = Mz.units
	hdulist = [fits.PrimaryHDU(header=hdr0),]
	hdu1 = fits.BinTableHDU.from_columns(np.array(gridData))
	hdulist.append(hdu1)
	hdulist = fits.HDUList(hdulist)
	hdulist.writeto(os.path.join(outputDir,simPar['GridFileName']+'.fits'),
	                clobber=True)

def readSimulationData(fileName,outputDir,retParams=False):
	qsoData = fits.getdata(os.path.join(outputDir,fileName+'.fits'),1)
	qsoData = Table(qsoData)
	if retParams:
		hdr = fits.getheader(os.path.join(outputDir,fileName+'.fits'),0)
		simPars = ast.literal_eval(hdr['SQPARAMS'])
		# XXX get it from parameters...
###		simPars['Cosmology'] = {
###		  'WMAP9':cosmology.WMAP9,
###		}[simPars['Cosmology']]
		return qsoData,simPars
	return qsoData

def writeSimulationData(simParams,Mz,gridData,simQSOs,photoData,outputDir,
                        writeFeatures):
	outShape = gridData['z'].shape
	fShape = outShape + (-1,) # shape for a "feature", vector at each point
	# Primary extension just contains model parameters in header
	simPar = copy(simParams)
	# XXX need to write parameters out or something...
	simPar['Cosmology'] = simPar['Cosmology'].name
	try:
		del simPar['GridParams']['QLFmodel']
	except:
		pass
	hdr0 = fits.Header()
	hdr0['SQPARAMS'] = str(simPar)
	hdr0['GRIDUNIT'] = Mz.units
	hdr0['GRIDDIM'] = str(outShape)
	hdulist = [fits.PrimaryHDU(header=hdr0),]
	# extension 1 contains the M,z grid and synthetic and observed fluxes
	if simQSOs is None:
		dataTab = gridData
	else:
		fluxData = Table({'synMag':simQSOs['synMag'].reshape(fShape),
		                  'synFlux':simQSOs['synFlux'].reshape(fShape)})
		if photoData is not None:
			_photoData = Table({ field:arr.reshape(fShape) 
			                        for field,arr in photoData.items() })
			dataTab = hstack([gridData,fluxData,_photoData])
		else:
			dataTab = hstack([gridData,fluxData])
	hdu1 = fits.BinTableHDU.from_columns(np.array(dataTab))
	hdulist.append(hdu1)
	# extension 2 contains feature information (slopes, line widths, etc.)
	if writeFeatures and simQSOs is not None:
		hdr2 = fits.Header()
		contData = simQSOs['continua'].getTable(hdr2)
		featureData = [feature.getTable(hdr2) 
		                  for feature in simQSOs['features']]
		featureData = [t for t in featureData if t is not None]
		featureTab = hstack([contData,]+featureData)
		hdu2 = fits.BinTableHDU.from_columns(np.array(featureTab))
		hdu2.header += hdr2
		hdulist.append(hdu2)
	hdulist = fits.HDUList(hdulist)
	hdulist.writeto(os.path.join(outputDir,simParams['FileName']+'.fits'),
	                clobber=True)


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
	'''
	saveSpectra = kwargs.get('saveSpectra',False)
	forestOnly = kwargs.get('forestOnly',False)
	onlyMap = kwargs.get('onlyMap',False)
	noPhotoMap = kwargs.get('noPhotoMap',False)
	noWriteOutput = kwargs.get('noWriteOutput',False)
	writeFeatures = kwargs.get('writeFeatures',False)
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
		qsoData,simParams = readSimulationData(simParams['FileName'],
		                                       outputDir,retParams=True)
		# XXX hack copy back in
		simParams['GridParams']['QLFmodel'] = qlf
		simParams['Cosmology'] = cosmo
		if simParams['GridParams']['GridType'].startswith('Flux'):
			Mz = grids.FluxGridFromData(qsoData,simParams['GridParams'],
			                            simParams.get('Cosmology'))
			gridData = qsoData['M','z','appMag']
			#gridData = qsoData['appMag','z']
		else:
			Mz = grids.LuminosityGridFromData(qsoData,simParams['GridParams'],
			                                  simParams.get('Cosmology'))
			gridData = qsoData['M','z']
	except IOError:
		print simParams['FileName']+' output not found'
		if 'GridFileName' in simParams:
			print 'restoring MzGrid from ',simParams['GridFileName']
			try:
				gridData = fits.getdata(os.path.join(outputDir,
				                        simParams['GridFileName']+'.fits'))
				if simParams['GridParams']['GridType'].startswith('Flux'):
					Mz = grids.FluxGridFromData(gridData,
					                            simParams['GridParams'],
					                            simParams.get('Cosmology'))
				else:
					Mz = grids.LuminosityGridFromData(gridData,
					                                  simParams['GridParams'],
					                                simParams.get('Cosmology'))
			except IOError:
				print simParams['GridFileName'],' not found, generating'
				Mz = buildMzGrid(simParams)
				gridData = initGridData(simParams,Mz)
				writeGridData(simParams,Mz,gridData,outputDir)
		else:
			print 'generating Mz grid'
			Mz = buildMzGrid(simParams)
		if not forestOnly:
			gridData = initGridData(simParams,Mz)
			if not noWriteOutput:
				writeSimulationData(simParams,Mz,gridData,None,None,
				                    outputDir,writeFeatures)
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
			forest = buildForest(wave,Mz.getRedshifts(),simParams,outputDir)
		# make sure that the forest redshifts actually match the grid
		assert np.allclose(forest['z'],Mz.zGrid.flatten())
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
		simQSOs = buildQSOspectra(wave,Mz,forest,photoMap,simParams,
		                          maxIter=simParams.get('maxFeatureIter',3),
		                          saveSpectra=saveSpectra)
	timerLog('Build Quasar Spectra')
	#
	# map the simulated photometry to observed values with uncertainties
	#
	if not noPhotoMap:
		print 'mapping photometry'
		np.random.seed(simParams['PhotoMapParams'].get('RandomSeed',
		               simParams.get('RandomSeed')))
		photoData = sqphoto.calcObsPhot(simQSOs['synFlux'],photoMap)
		timerLog('PhotoMap')
	else:
		photoData = None
	timerLog.dump()
	if not noWriteOutput:
		gridData['M'] = Mz.mGrid.flatten()
		writeSimulationData(simParams,Mz,gridData,simQSOs,photoData,
		                    outputDir,writeFeatures)
	if saveSpectra:
		fits.writeto(os.path.join(outputDir,
		                          simParams['FileName']+'_spectra.fits.gz'),
		             simQSOs['spectra'],clobber=True)

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

