#!/usr/bin/env python

import numpy as np
from astropy.io import fits
from astropy.table import Table,hstack

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
		R = simParams['SpecResolution']
		wave = sqbase.fixed_R_dispersion(lam1,lam2,R)
	else:
		raise ValueError('Dispersion scale %s not supported' % dispersionScale)
	return wave


def buildMzGrid(gridPars,cosmodef):
	'''
	Create a grid of points in (M,z) space, each of these points are
	assigned to an individual simulated quasar.
	'''
	try:
		gridType = gridPars['GridType']
	except KeyError:
		raise ValueError('Must specify a GridType')
	np.random.seed(gridPars.get('MzSeed'))
	if gridType == 'LuminosityFunction':
		try:
			qlf = gridPars['QLFmodel']
		except KeyError:
			raise ValueError('Must specify a parameterization of the LF')
		Mz = grids.LuminosityFunctionFluxGrid(gridPars['mRange'],gridPars['zRange'],
		                                      qlf,cosmodef,**gridPars['QLFargs'])
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
		                            cosmodef,
		                            obsBand=gridPars.get('ObsBand','SDSS-i'),
		                            restBand=gridPars.get('RestBand','1450'))
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
		forestSpec = hiforest.generate_spectra_from_grid(wave,z,tgrid,
		                                                 **forestParams)
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



def buildEmissionLineGrid(Mz,emLineParams):
	np.random.seed(emLineParams.get('RandomSeed'))
	if emLineParams['EmissionLineModel'] == 'FixedVdBCompositeLines':
		emLineGrid = grids.FixedVdBcompositeEMLineGrid(Mz.Mgrid,Mz.zgrid,
		                             minEW=emLineParams.get('minEW',1.0),
		                             noFe=emLineParams.get('VdB_noFe',False))
		# XXX hacky
		if emLineParams.get('addSBB',False):
			emLineGrid.addSBB()
#	elif emLineParams['EmissionLineModel'] == 'FixedLBQSEmissionLines':
#		emLineGrid = qsotemplates.FixedLBQSemLineGrid(
#		                                noFe=emLineParams.get('LBQSnoFe',False))
	elif emLineParams['EmissionLineModel'] == 'VariedEmissionLineGrid':
		emLineGrid = grids.VariedEmissionLineGrid(Mz.Mgrid,Mz.zgrid,
		                                          **emLineParams)
	else:
		raise ValueError('invalid emission line model: ' +
		                    emLineParams['EmissionLineModel'])
	if 'addLines' in emLineParams:
		for l in emLineParams['addLines']:
			print 'adding line ',l
			emLineGrid.addLine(*l)
	return emLineGrid

def buildDustGrid(Mz,dustParams):
	print '... building dust extinction grid'
	np.random.seed(dustParams.get('DustSeed'))
	if dustParams['DustExtinctionModel'] == 'Fixed E(B-V)':
		dustGrid = grids.FixedDustGrid(Mz.Mgrid,Mz.zgrid,
		                 dustParams['DustModelName'],dustParams['E(B-V)'])
	elif dustParams['DustExtinctionModel']=='Exponential E(B-V) Distribution':
		dustGrid = grids.ExponentialDustGrid(Mz.Mgrid,Mz.zgrid,
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

def buildFeatures(Mz,wave,qsoParams):
	features = []
	if 'EmissionLineParams' in qsoParams:
		emLineGrid = buildEmissionLineGrid(Mz,qsoParams['EmissionLineParams'])
		emLineFeature = EmissionLineFeature(emLineGrid)
		features.append(emLineFeature)
	if 'IronEmissionParams' in qsoParams:
		# only option for now is the VW01 template
		feGrid = grids.VW01FeTemplateGrid(Mz.Mgrid,Mz.zgrid,wave,
		           scales=qsoParams['IronEmissionParams'].get('FeScalings'))
		feFeature = IronEmissionFeature(feGrid)
		features.append(feFeature)
	if 'DustExtinctionParams' in qsoParams:
		dustGrid = buildDustGrid(Mz,qsoParams['DustExtinctionParams'])
		dustFeature = DustExtinctionFeature(dustGrid)
		features.append(dustFeature)
		pass
	return features


def buildQSOspectra(wave,Mz,forest,photoMap,qsoParams,
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
		spectra = np.zeros((Mz.Mgrid.size,len(wave)))
	else:
		spectra = None
	nforest = len(forest['wave'])
	assert np.all(np.abs(forest['wave']-wave[:nforest]<1e-3))
	continua = buildContinuumModels(Mz,qsoParams['ContinuumParams'])
	features = buildFeatures(Mz,wave,qsoParams)
	spec = QSOSpectrum(wave)
	gridShape = Mz.Mgrid.shape
	synMag = np.zeros(gridShape+(len(photoMap['bandpasses']),))
	synFlux = np.zeros_like(synMag)
	photoCache = sqphoto.getPhotoCache(wave,photoMap)
	if Mz.units == 'luminosity':
		nIter = 1
	else:
		nIter = maxIter
		bands = photoMap['bandpasses'].keys()
		fluxBand = next(j for j in range(len(bands)) if bands[j]==Mz.obsBand)
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
		if nIter > 1:
			Mz.updateMags(synMag[...,fluxBand])
			continua.update(Mz.Mgrid,Mz.zgrid)
			for feature in features:
				feature.update(Mz.Mgrid,Mz.zgrid)
	return dict(synMag=synMag,synFlux=synFlux,
	            continua=continua,features=features,spectra=spectra)


def timerLog(action):
	pass

def initGridData(simParams,Mz):
	return Table({'M':Mz.Mgrid.flatten(),'z':Mz.zgrid.flatten()})

def readSimulationData(fileName):
	qsoData = fits.getdata(fileName+'.fits',1)
	return Table(qsoData)

def writeSimulationData(simParams,Mz,gridData,simQSOs,photoData,**kwargs):
	writeFeatures = kwargs.get('writeFeatures',False)
	outShape = gridData['M'].shape
	fShape = outShape + (-1,) # shape for a "feature", vector at each point
	# Primary extension just contains model parameters in header
	hdr0 = fits.Header()
	hdr0['SQPARAMS'] = str(simParams)
	hdr0['GRIDUNIT'] = Mz.units
	# can be read back as ast.literal_eval(hdr['SQPARAMS'])
	hdulist = [fits.PrimaryHDU(header=hdr0),]
	# extension 1 contains the M,z grid and synthetic and observed fluxes
	if simQSOs is None:
		dataTab = gridData
	else:
		fluxData = Table({'synMag':simQSOs['synMag'].reshape(fShape),
		                  'synFlux':simQSOs['synFlux'].reshape(fShape)})
		# XXX temporary
		zarr = np.zeros_like(simQSOs['synMag']).reshape(outShape+(-1,))
		obsFluxData = Table({'obsMag':zarr,
		                     'obsFlux':zarr})
		dataTab = hstack([gridData,fluxData,obsFluxData])
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
	hdulist.writeto(simParams['FileName']+'.fits',clobber=True)


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
	#
	# build or restore the grid of (M,z) for each QSO
	#
	wave = buildWaveGrid(simParams)
	timerLog('StartSimulation')
	try:
		# simulation data already exists, load the Mz grid
		qsoData = readSimulationData(simParams['FileName'])
		hdr = fits.getheader(simParams['FileName']) # a bit kludgy
		Mz = grids.MzGridFromData(qsoData,simParams['GridParams'],hdr)
		gridData = qsoData['M','z']
	except IOError:
		print simParams['FileName']+' output not found'
		if 'GridFileName' in simParams:
			print 'restoring MzGrid from ',simParams['GridFileName']
			try:
				gridData = fits.getdata(simParams['GridFileName']+'.fits')
				hdr = fits.getheader(simParams['GridFileName']) # again kludgy
				Mz = qsogrid.MzGridFromData(gridData,simParams['GridParams'],hdr)
			except IOError:
				print simParams['GridFileName'],' not found, generating'
				Mz = buildMzGrid(simParams['GridParams'],
				                 simParams.get('Cosmology'))
				gridData = initGridData(simParams,Mz)
				gridData.write(simParams['GridFileName']+'.fits')
		else:
			print 'generating Mz grid'
			Mz = buildMzGrid(simParams['GridParams'],
			                 simParams.get('Cosmology'))
		if not forestOnly:
			gridData = initGridData(simParams,Mz)
			writeSimulationData(simParams,Mz,gridData,None,None)
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
		                          maxIter=simParams.get('maxFeatureIter',3),
		                          saveSpectra=saveSpectra)
	#
	# map the simulated photometry to observed values with uncertainties
	#
	if not noPhotoMap:
		print 'mapping photometry'
		timerLog('PhotoMap')
		photoData = photoMap.mapObserved(simQSOs)
	else:
		photoData = None
	timerLog('Finish')
	writeSimulationData(simParams,Mz,gridData,simQSOs,photoData,**kwargs)
	if saveSpectra:
		fits.writeto(simParams['FileName']+'_spectra.fits.gz',
		             simQSOs['spectra'],clobber=True)

