#!/usr/bin/env python

import os
import numpy as np
from functools import partial
from astropy.io import fits
from astropy.table import Table,hstack
from astropy import cosmology

from . import sqbase
from . import sqgrids as grids
from . import hiforest
from . import dustextinction
from . import sqphoto

import multiprocessing


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


def buildQsoGrid(simParams):
	'''
	Create a grid of simulated quasar "points". This function parses the 
	'GridParams' section of simParams, and intreprets the following options:

	- FluxRedshiftGrid : points are defined by (appMag,z)
	- LuminosityRedshiftGrid : points are defined by (absMag,z)
	- LuminosityFunction : points are defined by (appMag,z) and sampled from
	  a luminosity function.
	'''
	cosmodef = simParams.get('Cosmology')
	gridPars = simParams['GridParams']
	try:
		gridType = gridPars['GridType']
	except KeyError:
		raise ValueError('Must specify a GridType')
	m2M = lambda z: sqbase.mag2lum(gridPars['ObsBand'],gridPars['RestBand'],
	                               z,cosmodef)
	reseed(gridPars)
	#
	def get_nbins(low,high,n):
		if type(n) is int:
			return n
		else:
			return int( np.floor((high - low) / n) )
	if gridType.endswith('RedshiftGrid'):
		m1,m2,nm = gridPars['mRange']
		z1,z2,nz = gridPars['zRange']
		nBins = ( get_nbins(m1,m2,nm), get_nbins(z1,z2,nz) )
		mSampler = grids.UniformSampler(m1,m2)
		zSampler = grids.UniformSampler(z1,z2)
		if gridType.startswith('Luminosity'):
			m = grids.AbsMagVar(mSampler,restWave=gridPars['LumUnits'])
			units = 'luminosity'
		elif gridType.startswith('Flux'):
			m = grids.AppMagVar(mSampler,band=gridPars['ObsBand'],
			                    )#restBand=gridPars['RestBand'])
			units = 'flux'
		z = grids.RedshiftVar(zSampler)
	elif gridType == 'FixedGrid':
		raise NotImplementedError
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
		qsoGrid = grids.QsoSimGrid([m,z],nBins,gridPars['nPerBin'],
		                           units=units,cosmo=cosmodef)
	try:
		_ = qsoGrid.absMag
	except:
		absMag = qsoGrid.appMag - m2M(qsoGrid.z)
		absMag = grids.AbsMagVar(grids.FixedSampler(absMag))
		qsoGrid.addVar(absMag)
	return qsoGrid




def buildForest(wave,z,simParams,outputDir):
	'''Create a set of absorbers for a given number of lines-of-sight, 
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
	tgrid = hiforest.IGMTransmissionGrid(wave,nlos,zmax=z.max(),
	                                     **forestParams)
	return tgrid
	####
	waveMax = (1+z.max()+0.2)*1217
	wave = wave[:np.searchsorted(wave,waveMax)]
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


def buildFeatures(qsoGrid,wave,simParams,forest=None):
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
	if forest is not None:
		forestVar = grids.HIAbsorptionVar(forest)
		qsoGrid.addVar(forestVar)

def _getpar(feature,obj):
	if isinstance(feature.sampler,grids.NullSampler):
		return None
	elif isinstance(feature.sampler,grids.IndexSampler):
		return obj.index
	else:
		return obj[feature.name]

def buildQsoSpectrum(wave,cosmo,specFeatures,photoCache,saveSpectra,obj):
	spec = sqbase.Spectrum(wave,z=obj['z'])
	# start with continuum
	distmod = lambda z: cosmo.distmod(z).value
	fluxNorm = {'wavelength':1450.,'M_AB':obj['absMag'],'DM':distmod}
	for feature in specFeatures: #qsoGrid.getVars(grids.ContinuumVar):
		if isinstance(feature,grids.ContinuumVar):
			spec = feature.add_to_spec(spec,_getpar(feature,obj),
			                           fluxNorm=fluxNorm)
	# add emission (multiplicative) features
	emspec = sqbase.Spectrum(wave,z=obj['z'])
	for feature in specFeatures: #qsoGrid.getVars(grids.EmissionFeatureVar):
		if isinstance(feature,grids.EmissionFeatureVar):
			emspec = feature.add_to_spec(emspec,_getpar(feature,obj))
	spec *= emspec + 1
	# add any remaining features
	for feature in specFeatures: #qsoGrid.getVars(grids.SpectralFeatureVar):
		if isinstance(feature,grids.ContinuumVar) or \
		   isinstance(feature,grids.EmissionFeatureVar):
			continue
		spec = feature.add_to_spec(spec,_getpar(feature,obj))
	if photoCache is not None:
		# calculate synthetic magnitudes from the spectra through the
		# specified bandpasses
		rv = sqphoto.calcSynPhot(spec,photoCache=photoCache)
	else:
		rv = ()
	if saveSpectra:
		rv += (spec.f_lambda,)
	return rv

def buildGrpSpectra(wave,cosmo,specFeatures,photoCache,saveSpectra,
                    nIter,fluxBand,objGroup):
	rv = []
	zi = objGroup['z'].argsort()
	for i in zi:
		for iterNum in range(nIter):
			_rv = buildQsoSpectrum(wave,cosmo,specFeatures,photoCache,
			                       saveSpectra,objGroup[i])
			if nIter > 1:
				synMag = _rv[0]
				dm = synMag[fluxBand] - objGroup['appMag'][i]
				objGroup['absMag'][i] -= dm
				# resample features with updated absolute mags
				#qsoGrid.resample() 
				for var in specFeatures:
					if var.update:
# XXX argh, latest problem is that this is a *subset* (in fact, single)
# point in the variable, but from resample on down expect the full array...
						import pdb; pdb.set_trace()
						var.resample(objGroup[var.dependentVars][i])
# XXX this can be passed up
#						objGroup[var.name][i] = var(None)
				if dm < 0.005:
					break
			rv.append(_rv)
	return rv

def _regroup(spOut):
	# XXX tell me there's a better way to do this
	n = len(spOut[0])
	rv = [ [] for i in range(n) ]
	for sp in spOut:
		for j in range(n):
			rv[j].append(sp[j])
	return [ np.array(v) for v in rv ]

def buildQsoSpectra2(wave,qsoGrid,photoMap=None,
                     maxIter=1,saveSpectra=False):
	'''Assemble the spectral components of QSOs from the input parameters.

	Parameters
	----------
	wave : `~numpy.ndarray`
	    Input wavelength grid.
	'''
	if photoMap is not None:
		photoCache = sqphoto.getPhotoCache(wave,photoMap)
	else:
		photoCache = None
	print 'simulating ',qsoGrid.nObj,' quasar spectra'
	print 'units are ',qsoGrid.units
	if qsoGrid.units == 'luminosity' or photoMap is None:
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
	pool = multiprocessing.Pool(7)
	if True:
		specFeatures = qsoGrid.getVars(grids.SpectralFeatureVar)
		build_grp_spec = partial(buildGrpSpectra,wave,qsoGrid.cosmo,
		                         specFeatures,photoCache,saveSpectra,
		                         fluxBand,nIter)
		specOut = map(build_grp_spec,qsoGrid.iter_sightlines())
		#specOut = map(build_one_spec,qsoGrid.iter_reorder(zi))
		#specOut = pool.map(build_one_spec,qsoGrid.iter_reorder(zi))
		specOut = _regroup(specOut)
		synMag,synFlux = specOut[:2]
		synMag = synMag[zi.argsort()]
		synFlux = synFlux[zi.argsort()]
	if saveSpectra:
		spectra = specOut[2][zi.argsort()]
	else:
		spectra = None
	if photoMap is not None:
		qsoGrid.addVar(grids.SynMagVar(grids.FixedSampler(synMag)))
		qsoGrid.addVar(grids.SynFluxVar(grids.FixedSampler(synFlux)))
	return qsoGrid,spectra

def buildQsoSpectra(wave,qsoGrid,photoMap=None,
                    maxIter=1,saveSpectra=False):
	'''Assemble the spectral components of QSOs from the input parameters.

	Parameters
	----------
	wave : `~numpy.ndarray`
	    Input wavelength grid.
	'''
	if photoMap is not None:
		photoCache = sqphoto.getPhotoCache(wave,photoMap)
	else:
		photoCache = None
	print 'simulating ',qsoGrid.nObj,' quasar spectra'
	print 'units are ',qsoGrid.units
	if qsoGrid.units == 'luminosity' or photoMap is None:
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
	pool = multiprocessing.Pool(7)
	for iterNum in range(nIter):
		specFeatures = qsoGrid.getVars(grids.SpectralFeatureVar)
		samplers = []
		for f in specFeatures:
			samplers.append(f.sampler)
			if not ( isinstance(f.sampler,grids.NullSampler) or 
			         isinstance(f.sampler,grids.IndexSampler) ):
				f.sampler = None
		build_one_spec = partial(buildQsoSpectrum,wave,qsoGrid.cosmo,
		                         specFeatures,photoCache,saveSpectra)
		print 'buildQsoSpectra iteration ',iterNum+1,' out of ',nIter
		specOut = map(build_one_spec,qsoGrid)
		#specOut = pool.map(build_one_spec,qsoGrid)
		specOut = _regroup(specOut)
		synMag,synFlux = specOut[:2]
		for f,s in zip(specFeatures,samplers):
			f.sampler = s
		if nIter > 1:
			# find the largest mag offset
			dm = synMag[:,fluxBand] - qsoGrid.appMag
			print '--> delta mag mean = %.7f, rms = %.7f, |max| = %.7f' % \
			              (dm.mean(),dm.std(),np.abs(dm).max())
			qsoGrid.absMag[:] -= dm
			dmagMax = np.abs(dm).max()
			# resample features with updated absolute mags
			qsoGrid.resample()
			if dmagMax < 0.01:
				break
	if saveSpectra:
		spectra = specOut[2]
	else:
		spectra = None
	if photoMap is not None:
		qsoGrid.addVar(grids.SynMagVar(grids.FixedSampler(synMag)))
		qsoGrid.addVar(grids.SynFluxVar(grids.FixedSampler(synFlux)))
	return qsoGrid,spectra


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

	1. Construct grid of QSOs.
	2. Generate Lyman forest transmission spectra from a subsample of 
	   random LOSs (optional).
	3. Sample QSO spectral features (continuum, emission lines, dust).
	4. Build simulated spectra and derive photometry (photometry is optional).
	5. Transfer the simulated photometry to observed photometry by 
	   calculating errors and folding them in (optional).

	Parameters
	----------
	saveSpectra : bool 
	    save the simulated spectra, not just the photometry.
	    Beware! result may be quite large (Nqso x Npixels). [default:False]
	forestOnly : bool
	    Only generate the forest transmission spectra. [default:False]
	onlyMap : bool
	    Only do the simulation of observed photometry, assuming 
	    synthetic photometry has already been generated [default:False]
	noPhotoMap : bool
	    skip the simulation of observed photometry [default:False]
	outputDir : str
	    write files to this directory [default:'./']
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
	timerLog = sqbase.TimerLog()
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
				qsoGrid = buildQsoGrid(simParams)
				qsoGrid.write(simParams,outputDir,
				              simParams['GridFileName']+'.fits')
		else:
			print 'generating QSO grid'
			qsoGrid = buildQsoGrid(simParams)
		if not forestOnly:
			if not noWriteOutput and 'GridFileName' in simParams:
				qsoGrid.write(simParams,outputDir,
				              simParams['GridFileName']+'.fits')
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
###		# make sure that the forest redshifts actually match the grid
###		assert np.allclose(forest['z'],qsoGrid.z)
	if forestOnly:
		timerLog.dump()
		return
###	assert np.allclose(forest['wave'],wave[:len(forest['wave'])])
	timerLog('Generate Forest')
	#
	# Use continuum and emission line distributions to build the components
	# of the intrinsic QSO spectrum, then calculate photometry
	#
	photoMap = sqphoto.load_photo_map(simParams['PhotoMapParams'])
	if not onlyMap:
		buildFeatures(qsoGrid,wave,simParams,forest)
		#_,spectra = buildQsoSpectra(wave,qsoGrid,photoMap=photoMap,
		_,spectra = buildQsoSpectra2(wave,qsoGrid,photoMap=photoMap,
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
	reseed(forestParams)
	timerLog = sqbase.TimerLog()
	tgrid = hiforest.generate_grid_spectra(wave,zbins,nlos,**forestParams)
	timerLog('BuildForest')
	hiforest.save_spectra(tgrid,forestParams['FileName'],outputDir,
	                      saveAbs=True)
	timerLog.dump()

