#!/usr/bin/env python

import os
import warnings

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
from . import sqmodels

import multiprocessing


def _reseed(par):
    try:
        np.random.seed(par['RandomSeed'])
    except KeyError:
        pass

def _build_wave_grid(simParams):
    dispersionScale = simParams.get('DispersionScale','logarithmic')
    if dispersionScale == 'logarithmic':
        lam1,lam2 = simParams['waveRange']
        R = simParams['SpecDispersion']
        wave = sqbase.fixed_R_dispersion(lam1,lam2,R)
    else:
        raise ValueError('Dispersion scale %s not supported' % dispersionScale)
    return wave

def _build_qso_grid(simParams):
    '''Create a grid of simulated quasar "points". This function parses the 
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
    kcorrType = gridPars.get('InitialKCorrection','Continuum')
    if kcorrType == 'Continuum':
        kcorr = sqbase.ContinuumKCorr(gridPars['ObsBand'],
                                      gridPars['RestBand'])
    elif kcorrType == 'DefaultEmissionLine':
        kcorr = sqbase.EmissionLineKCorr(gridPars['ObsBand'],
                                         gridPars['RestBand'])
    else:
        raise ValueError
    _reseed(gridPars)
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
            m = grids.AppMagVar(mSampler,gridPars['ObsBand'])
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
            qlf.set_cosmology(cosmodef)
        except KeyError:
            raise ValueError('Must specify a parameterization of the LF')
        qsoGrid = grids.generateQlfPoints(qlf,
                                          gridPars['mRange'],
                                          gridPars['zRange'],
                                          kcorr,
                                          **gridPars['QLFargs'])
        units = 'flux'
    else:
        raise ValueError('GridType %s unknown' % gridType)
    if gridType != 'LuminosityFunction':
        qsoGrid = grids.QsoSimGrid([m,z],nBins,gridPars['nPerBin'],
                                   units=units,cosmo=cosmodef)
    try:
        _ = qsoGrid.absMag
    except:
        absMag = grids.AbsMagFromAppMagVar(qsoGrid.appMag,z,kcorr,cosmo,
                                           gridPars['RestBand'])
        qsoGrid.addVar(absMag)
    return qsoGrid

def _build_forest(wave,z,simParams,outputDir):
    '''Create a set of absorbers for a given number of lines-of-sight, 
    sampled according to the input forest model. Then calculate the
    transmission along each line of sight. The input redshifts correspond
    to individual QSOs. The number of LOSs is generally smaller so that
    fewer forest computations are needed; individual LOSs are built up
    in redshift steps as each QSO redshift is iterated.
    '''
    forestParams = simParams['ForestParams']
    _reseed(forestParams)
    forestFn = forestParams.get('FileName')
    tgrid = None
    if forestFn:
        try:
            tgrid = hiforest.CachedIGMTransmissionGrid(forestFn,outputDir)
            if not np.allclose(wave[:len(tgrid.specWave)],tgrid.specWave):
                raise ValueError("Input wavegrid doesn't match stored wave")
        except IOError:
            pass
    if tgrid is None:
        nlos = forestParams['NumLinesOfSight']
        forestModel = forestParams['ForestModel']
        if isinstance(forestModel,str):
            forestModel = sqmodels.forestModels[forestModel]
        tgrid = hiforest.IGMTransmissionGrid(wave,forestModel,nlos,
                                             zmax=z.max(),**forestParams)
    return tgrid

def _build_continua(qsoGrid,simParams,verbose=0):
    continuumParams = simParams['QuasarModelParams']['ContinuumParams']
    _reseed(continuumParams)
    slopes = continuumParams['PowerLawSlopes'][::2]
    breakpts = continuumParams['PowerLawSlopes'][1::2]
    if verbose > 0:
        print('... building continuum grid')
    cmodel = continuumParams['ContinuumModel']
    if cmodel == 'BrokenPowerLaw':
        slopeVars = [ grids.GaussianSampler(*s) for s in slopes ]
        continuumVars = [ grids.BrokenPowerLawContinuumVar(slopeVars,
                                                           breakpts) ]
    elif isinstance(cmodel,grids.QsoSimVar):
        continuumVars = [ cmodel ]
    else:
        raise ValueError
    qsoGrid.addVars(continuumVars)


def _build_emission_line_grid(qsoGrid,simParams):
    emLineParams = simParams['QuasarModelParams']['EmissionLineParams']
    _reseed(emLineParams)
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

def _build_dust_grid(qsoGrid,simParams,verbose=0):
    if verbose > 0:
        print('... building dust extinction grid')
    dustParams = simParams['QuasarModelParams']['DustExtinctionParams']
    _reseed(dustParams)
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
    qsoGrid.addVar(dustVar)

def _build_features(qsoGrid,wave,simParams,forest=None,verbose=0):
    _build_continua(qsoGrid,simParams,verbose=verbose)
    qsoParams = simParams['QuasarModelParams']
    if 'EmissionLineParams' in qsoParams:
        _build_emission_line_grid(qsoGrid,simParams)
    if 'IronEmissionParams' in qsoParams:
        # only option for now is the VW01 template
        scalings = qsoParams['IronEmissionParams'].get('FeScalings')
        feGrid = grids.VW01FeTemplateGrid(qsoGrid.z,wave,scales=scalings)
        qsoGrid.addVar(grids.FeTemplateVar(feGrid))
    if 'DustExtinctionParams' in qsoParams:
        _build_dust_grid(qsoGrid,simParams,verbose=verbose)
    if forest is not None:
        if isinstance(forest,hiforest.CachedIGMTransmissionGrid):
            losMap = forest.losMap
        else:
            losMap = None
        if isinstance(forest,hiforest.GridForest):
            forestVar = grids.SightlineVar(forest,losMap=losMap)
        else:
            forestVar = grids.HIAbsorptionVar(forest,losMap=losMap)
        qsoGrid.addVar(forestVar)

def _getpar(feature,obj):
    if feature is None:
        return None
    elif isinstance(feature.sampler,grids.NullSampler):
        return None
    elif isinstance(feature.sampler,grids.IndexSampler):
        return obj.index
    else:
        return obj[feature.name]

def build_qso_spectrum(wave,cosmo,specFeatures,obj,**kwargs):
    '''Generate a single mock quasar spectrum given the input list of
    spectral features and their associated values.

    Parameters
    ----------
    wave : `~numpy.ndarray`
        Input wavelength grid.
    cosmo : astropy.cosmo.Cosmology object
        Cosmology.
    specFeatures : sequence of sqbase.SpectralFeature instances
        List of spectral features defining the quasar model.
    obj : dict, namedtuple, or astropy.Table.Row object
        Values associated with each specFeature, indexed by 
        SpectralFeature.name

    Returns
    -------
    spec : `~sqbase.Spectrum`
        Spectrum object containing the generated spectrum.

    Notes
    -----
    While building a single spectrum through this interface is supported,
    using the higher level build_spectra_from_grid() is likely a better 
    option.
    '''
    iterNum = kwargs.pop('iterNum',1)
    saveComponents = kwargs.pop('save_components',False)
    spec = sqbase.Spectrum(wave,z=obj['z'])
    if saveComponents:
        base = sqbase.Spectrum(spec.wave,spec.f_lambda.copy(),spec.z)
        components = {}
    # start with continuum
    if cosmo is None:
        fluxNorm = None
    else:
        distmod = lambda z: cosmo.distmod(z).value
        fluxNorm = {'wavelength':1450.,'M_AB':obj['absMag'],'DM':distmod}
    for feature in specFeatures:
        if isinstance(feature,grids.ContinuumVar):
            assocvals = _getpar(feature.get_associated_var(),obj)
            spec = feature.add_to_spec(spec,_getpar(feature,obj),
                                       assocvals=assocvals,
                                       fluxNorm=fluxNorm)
            if saveComponents:
                components[feature.name] = spec - base
                base.f_lambda[:] = spec.f_lambda
    # add emission (multiplicative) features
    emspec = sqbase.Spectrum(wave,z=obj['z'])
    if saveComponents:
        base = sqbase.Spectrum(emspec.wave,emspec.f_lambda.copy(),emspec.z)
    for feature in specFeatures:
        if isinstance(feature,grids.EmissionFeatureVar):
            assocvals = _getpar(feature.get_associated_var(),obj)
            emspec = feature.add_to_spec(emspec,_getpar(feature,obj),
                                         assocvals=assocvals)
            if saveComponents:
                components[feature.name] = emspec - base
                base.f_lambda[:] = emspec.f_lambda
    spec *= emspec + 1
    # add any remaining features
    for feature in specFeatures:
        if isinstance(feature,grids.ContinuumVar) or \
           isinstance(feature,grids.EmissionFeatureVar):
            continue
        assocvals = _getpar(feature.get_associated_var(),obj)
        spec = feature.add_to_spec(spec,_getpar(feature,obj),
                                   assocvals=assocvals,
                                   advance=(iterNum==1))
        if saveComponents:
            components[feature.name] = spec - base
            base.f_lambda[:] = spec.f_lambda
    if saveComponents:
        return spec,components
    else:
        return spec

def _build_group_spectra(wave,cosmo,specFeatures,photoCache,saveSpectra,
                         fluxBand,nIter,verbose,objGroup):
    n = len(objGroup)
    if verbose and verbose > 0:
        losGrp = objGroup['igmlos'][0]
        if losGrp % verbose == 0:
            print('processing ',n,' obj in group ',losGrp)
    rv = dict()
    if photoCache:
        nb = len(photoCache)
        rv['synMag'] = np.zeros((n,nb),dtype=np.float32)
        rv['synFlux'] = np.zeros((n,nb),dtype=np.float32)
    if saveSpectra:
        nw = len(wave)
        rv['spectra'] = np.zeros((n,nw),dtype=np.float32)
    zi = objGroup['z'].argsort()
    for i in zi:
        for iterNum in range(1,nIter+1):
            sp = build_qso_spectrum(wave,cosmo,specFeatures,objGroup[i],
                                    iterNum=iterNum)
            if photoCache is not None:
                synMag,synFlux = sqphoto.calcSynPhot(sp,photoCache=photoCache)
            if nIter > 1:
                dm = synMag[fluxBand] - objGroup['appMag'][i]
                objGroup['absMag'][i] -= dm
                # resample features with updated absolute mags
                for var in specFeatures:
                    if var.dependentVars is not None:
                        var.resample(objGroup[var.dependentVars][i],ii=i)
                        # pass index as 1d-array to preserve correct shape
                        objGroup[var.name][i] = var(None,ii=np.array([i]))
                if np.abs(dm) < 0.005:
                    break
        if photoCache is not None:
            rv['synMag'][i] = synMag
            rv['synFlux'][i] = synFlux
        if saveSpectra:
            rv['spectra'][i] = sp.f_lambda
    rv['absMag'] = objGroup['absMag'].copy()
    return rv

def _regroup(spOut):
    # XXX tell me there's a better way to do this
    n = len(spOut[0])
    rv = [ [] for i in range(n) ]
    for sp in spOut:
        for j in range(n):
            rv[j].append(sp[j])
    return [ np.array(v) for v in rv ]

def _build_spectra_with_phot(wave,cosmo,specFeatures,photoCache,
                             objData,iterNum=None,saveSpectra=False):
    sp = build_qso_spectrum(wave,cosmo,specFeatures,objData,
                            iterNum=iterNum)
    if photoCache is None:
        rv = (None,None)
    else:
        rv = sqphoto.calcSynPhot(sp,photoCache=photoCache)
    if saveSpectra:
        rv = rv + (sp.f_lambda,)
    else:
        rv = rv + (None,)
    return rv

def build_spectra_from_grid(wave,qsoGrid,procMap=map,
                            maxIter=1,verbose=0,saveSpectra=False):
    '''Assemble the spectral components of QSOs from the input parameters.

    Parameters
    ----------
    wave : `~numpy.ndarray`
        Input wavelength grid.
    qsoGrid : `~sqgrids.QsoSimObjects`
        Quasar grid describing spectral model and providing parameters for
        individual objects.
    procMap : callable
        `map`-like function, can be multiprocessing.Pool.map
    maxIter : int
        maximum number of iterations used to converge on per-object 
        k-corrections if the grid is defined with respect to apparent
        magnitudes. Default is 1.
    verbose : int
        verbosity level
    saveSpectra : bool
        Keep the generated spectra in addition to the synthetic photometry.
        Uses a lot of memory if the grid is large.
    '''
    photoCache = qsoGrid.getPhotoCache(wave)
    if verbose > 0:
        print('simulating ',qsoGrid.nObj,' quasar spectra')
        print('units are ',qsoGrid.units)
    if qsoGrid.units == 'luminosity' or photoCache is None:
        nIter = 1
        fluxBand = None
    else:
        nIter = maxIter
        fluxBand = qsoGrid.getObsBandIndex()
    #
    for iterNum in range(1,nIter+1):
        specFeatures = qsoGrid.getVars(grids.SpectralFeatureVar)
        samplers = []
        for f in specFeatures:
            samplers.append(f.sampler)
            if not ( isinstance(f.sampler,grids.NullSampler) or 
                     isinstance(f.sampler,grids.IndexSampler) ):
                f.sampler = None
        build_one_spec = partial(_build_spectra_with_phot,wave,qsoGrid.cosmo,
                                 specFeatures,photoCache,iterNum=iterNum,
                                 saveSpectra=saveSpectra)
        if verbose > 1:
            print('buildSpectra iteration ',iterNum,' out of ',nIter)
        specOut = list(procMap(build_one_spec,qsoGrid))
        specOut = _regroup(specOut)
        synMag,synFlux,spectra = specOut
        v = qsoGrid.getVars(grids.SightlineVar)
        if len(v) > 0 and isinstance(v[0].forest,hiforest.GridForest):
            jj,dm,df = v[0].forest.get(qsoGrid.data['igmlos'],
                                       qsoGrid.data['z'])
            synMag[:,jj] += dm
            synFlux[:,jj] *= df
        for f,s in zip(specFeatures,samplers):
            f.sampler = s
        if nIter > 1:
            # find the largest mag offset
            dm = synMag[:,fluxBand] - qsoGrid.appMag
            if verbose > 1:
                print('--> delta mag mean = %.7f, rms = %.7f, |max| = %.7f' % \
                          (dm.mean(),dm.std(),np.abs(dm).max()))
            qsoGrid.absMag[:] -= dm
            dmagMax = np.abs(dm).max()
            # resample features with updated absolute mags
            for var in specFeatures:
                if var.dependentVars is not None:
                    var.resample(qsoGrid.data[var.dependentVars])
                    qsoGrid.data[var.name][:] = var(None)
            if dmagMax < 0.01:
                break
    if qsoGrid.photoMap is not None:
        qsoGrid.addVar(grids.SynMagVar(grids.FixedSampler(synMag)))
        qsoGrid.addVar(grids.SynFluxVar(grids.FixedSampler(synFlux)))
    return qsoGrid,spectra

def build_spectra_by_sightline(wave,qsoGrid,procMap=map,
                               maxIter=1,verbose=0,saveSpectra=False):
    '''See build_spectra_from_grid(). This method sorts the quasars by
    IGM sightline and redshift, so that the forest transmission spectra
    can be generated one-at-a-time by adding absorbers up to each quasar
    redshift associated with a single sightline. This method is necessary
    if the forest component is an ~sqbase.IGMTransmissionGrid with a number
    of sightlines smaller than the number of quasars.
    '''
    photoCache = qsoGrid.getPhotoCache(wave)
    if verbose > 0:
        print('simulating ',qsoGrid.nObj,' quasar spectra')
        print('units are ',qsoGrid.units)
        print('max number iterations: ',maxIter)
    verby = 0 if not verbose else qsoGrid.nObj//(5*verbose)
    if qsoGrid.units == 'luminosity' or photoCache is None:
        nIter = 1
        fluxBand = None
    else:
        nIter = maxIter
        fluxBand = qsoGrid.getObsBandIndex()
    #
    # extract the feature lists, group by sightline, and run
    specFeatures = qsoGrid.getVars(grids.SpectralFeatureVar)
    build_grp_spec = partial(_build_group_spectra,wave,qsoGrid.cosmo,
                             specFeatures,photoCache,saveSpectra,
                             fluxBand,nIter,verby)
    qsoGroups = qsoGrid.group_by('igmlos',with_index=True)
    # pool.map() doesn't like the iterable produced by table.group_by(), so
    # forcing resolution of the elements here with list() -- not that much
    # memory anyway
    specOut = list(procMap(build_grp_spec,list(qsoGroups)))
    if qsoGrid.photoMap:
        bands = qsoGrid.photoBands
        def newarr():
            return np.zeros((qsoGrid.nObj,len(bands)),dtype=np.float32)
        qsoGrid.addVar(grids.SynMagVar(grids.FixedSampler(newarr())))
        qsoGrid.addVar(grids.SynFluxVar(grids.FixedSampler(newarr())))
    # the output needs to be remapped to the input locations
    for objgrp,out in zip(qsoGroups,specOut):
        for k in ['absMag','synMag','synFlux']:
            qsoGrid.data[k][objgrp['_ii']] = out[k]
    if saveSpectra:
        spectra = np.vstack([s['spectra'] for s in specOut])
        spectra = spectra[qsoGroups.parent['_ii'].argsort()]
    else:
        spectra = None
    return qsoGrid,spectra


def qso_simulation(simParams,**kwargs):
    '''Run a complete simulation using a dictionary of parameters to define
    the quasar model.

    1. Construct grid of QSOs.
    2. Generate Lyman forest transmission spectra from a subsample of 
       random LOSs (optional).
    3. Sample QSO spectral features (continuum, emission lines, dust).
    4. Build simulated spectra and derive photometry (photometry is optional).
    5. Transfer the simulated photometry to observed photometry by 
       calculating errors and folding them in (optional).

    Parameters
    ----------
    simParams : dict
        Quasar model parameters.
    saveSpectra : bool 
        save the simulated spectra, not just the photometry.
        Beware! result may be quite large (Nqso x Npixels). [default:False]
    forestOnly : bool
        Only generate the forest transmission spectra. [default:False]
    noPhotoMap : bool
        skip the simulation of observed photometry [default:False]
    outputDir : str
        write files to this directory [default:'./']
    nproc : int
        number of processes to use [default: 1]

    Returns
    -------
    qsoGrid : sqbase.QsoSimObjectsinstance
        simulation grid
    spectra : np.ndarray
        [optional if saveSpectra=True] 2D array of simulated spectra
    '''
    saveSpectra = kwargs.get('saveSpectra',False)
    forestOnly = kwargs.get('forestOnly',False)
    noPhotoMap = kwargs.get('noPhotoMap',False)
    noWriteOutput = kwargs.get('noWriteOutput',False)
    outputDir = kwargs.get('outputDir','./')
    nproc = kwargs.get('nproc',1)
    verbose = kwargs.get('verbose',0)
    #
    # build or restore the grid of (M,z) for each QSO
    #
    wave = _build_wave_grid(simParams)
    _reseed(simParams)
    if nproc > 1:
        pool = multiprocessing.Pool(nproc)
        procMap = pool.map
    else:
        procMap = map
    timerLog = sqbase.TimerLog()
    try:
        # only keep the grid parameters and rerun everything else
        # XXX this assumes the grid is magnitude/redshift. Need to keep track
        #     of the grid variables for other types of grids.
        qsoGrid,simParams = restore_qso_grid(simParams['FileName'],None,
                                             outputDir=outputDir,
                                             retParams=True,
                                             keep=['absMag','appMag','z'])
    except IOError:
        if verbose > 0:
            print(simParams['FileName']+' output not found')
        if 'GridFileName' in simParams:
            if verbose > 0:
                print('restoring grid from ',simParams['GridFileName'])
            try:
                qsoGrid = restore_qso_grid(simParams['GridFileName'],None,
                                           outputDir=outputDir)
            except IOError:
                if verbose > 0:
                    print(simParams['GridFileName'],' not found, generating')
                qsoGrid = _build_qso_grid(simParams)
                qsoGrid.write(simParams,outputDir,
                              simParams['GridFileName']+'.fits')
        else:
            if verbose > 0:
                print('generating QSO grid')
            qsoGrid = _build_qso_grid(simParams)
        if not forestOnly:
            if not noWriteOutput and 'GridFileName' in simParams:
                qsoGrid.write(simParams,outputDir,
                              simParams['GridFileName']+'.fits')
    qsoGrid.setCosmology(simParams.get('Cosmology'))
    timerLog('Initialize Grid')
    #
    # configure the IGM transmission spectra grid (load if cached)
    #
    if 'ForestParams' in simParams:
        forest = _build_forest(wave,qsoGrid.z,simParams,outputDir)
    else:
        forest = None
    if forestOnly:
        timerLog.dump()
        return
    #
    if isinstance(forest,hiforest.IGMTransmissionGrid):
        # build sightlines on-the-fly
        buildSpec = build_spectra_by_sightline
        # if the user specified a file name, save the forest spectra in it
        fpar = simParams.get('ForestParams',{})
        forestFn = fpar.get('FileName')
        if forestFn:
            # map the objects to sightlines and save the forest spectra grid
            losSampler = grids.RandomSubSampler(forest.numSightLines)
            losMap = losSampler.sample(qsoGrid.nObj)
            forest.write(forestFn,outputDir,losMap=losMap,
                         z_em=qsoGrid.z,**fpar)
            # now use the cached forest
            forest = hiforest.CachedIGMTransmissionGrid(forestFn,outputDir)
            if not np.allclose(wave[:len(tgrid.specWave)],tgrid.specWave):
                raise ValueError("Input wavegrid doesn't match stored wave")
            timerLog('Generate Forest')
    else:
        # else no forest or cached forest
        buildSpec = build_spectra_from_grid
    #
    qsoGrid.loadPhotoMap(simParams['PhotoMapParams']['PhotoSystems'])
    if 'GridForestFile' in simParams:
        forest = hiforest.GridForest(simParams['GridForestFile'],
                                     qsoGrid.photoBands)
    #
    # add the quasar model variables to the grid (does the random sampling)
    #
    _build_features(qsoGrid,wave,simParams,forest,verbose=verbose)
    timerLog('Generate Features')
    #
    # Use continuum and emission line distributions to build the components
    # of the intrinsic QSO spectrum, then calculate photometry
    #
    _,spectra = buildSpec(wave,qsoGrid,procMap,
                          maxIter=simParams.get('maxFeatureIter',5),
                          verbose=verbose,saveSpectra=saveSpectra)
    timerLog('Build Quasar Spectra')
    #
    # map the simulated photometry to observed values with uncertainties
    #
    if not noPhotoMap:
        if verbose > 0:
            print('mapping photometry')
        _reseed(simParams['PhotoMapParams'])
        photoData = sqphoto.calcObsPhot(qsoGrid.synFlux,qsoGrid.photoMap)
        qsoGrid.addData(photoData)
        timerLog('PhotoMap')
    timerLog.dump()
    if nproc > 1:
        pool.close()
    if not noWriteOutput:
        qsoGrid.write(simPars=simParams,outputDir=outputDir)
    if saveSpectra:
        spfn = os.path.join(outputDir,simParams['FileName']+'_spectra.fits')
        save_spectra(wave,spectra,spfn,outputDir)
        return qsoGrid,spectra
    else:
        return qsoGrid

def save_spectra(wave,spectra,fileName,outputDir='.',overwrite=True):
    '''Save the spectra generated from the simulation to a FITS table.
    Wavelength information is stored in the header using standard IRAF
    keywords.
    '''
    logwave = np.log(wave[:2])
    dloglam = np.diff(logwave)
    hdr = fits.Header()
    hdr['CD1_1'] = float(dloglam)
    hdr['CRPIX1'] = 1
    hdr['CRVAL1'] = logwave[0]
    hdr['CRTYPE1'] = 'LOGWAVE'
    hdr['SPECSCAL'] = (1e-17,'erg/s/cm^2/A')
    spectra = (spectra*1e17).astype(np.float32)
    if not fileName.endswith('.fits'):
        fileName += '.fits'
    fits.writeto(os.path.join(outputDir,fileName),spectra,header=hdr,
                 overwrite=overwrite)

def load_spectra(fileName,outputDir='.'):
    '''Load simlated spectra from a FITS file.

    Returns
    -------
    wave : np.ndarray
        wavelength vector (N points).
    spec : np.ndarray
        MxN array containing the full set of spectra.
    '''
    if not fileName.endswith('.fits'):
        fileName += '.fits'
    spec,hdr = fits.getdata(fileName,header=True)
    wi = np.arange(spec.shape[-1])
    logwave = hdr['CRVAL1'] + hdr['CD1_1']*(wi-(hdr['CRPIX1']-1))
    wave = np.exp(logwave)
    return wave,spec

def restore_qso_grid(fileName,wave,outputDir='.',
                     retParams=False,keep='all',**kwargs):
    '''Restore a saved quasar grid into an sqbase.QsoSimObjects instance.

    Parameters
    ----------
    fileName : string
        Name of file containing output simulation grid.
    wave : `numpy.ndarray`
        Wavelength grid. Not required to be the same as the grid used for
        the initial run; i.e., one can run a simulation and generate
        spectra with one grid, save the grid output, and then restore
        with a new wavelength grid.
        If wave=None, attempts to restore wavelength grid from saved
        parameter dict.
    retParams : bool
        if true, return the dict of simulation parameters (if present)
    keep : str or sequence
        if 'all', restore all variables from the simulation output.
        if a list of strings, return only the named variables.
        E.g., if keep=['absMag','z'], retain only the absolute mag and 
        redshift variables. This allows simulations to be rerun while
        maintaining some of the same inputs.
    kwargs : 
        Additional kwargs passed to QsoSimObjects.read().

    Returns
    -------
    qsoGrid : sqbase.QsoSimObjects
        A new simulation grid based on the output of a previous simulation.
    '''
    qsoGrid = grids.QsoSimObjects()
    if not fileName.endswith('.fits'):
        fileName += '.fits'
    qsoGrid.read(os.path.join(outputDir,fileName),**kwargs)
    if wave is None:
        try:
            wave = _build_wave_grid(qsoGrid.simPars)
        except:
            raise ValueError("Must specify wavelength grid.")
    #
    if keep != 'all':
        for var in qsoGrid.getVars():
            if var.name not in keep:
                qsoGrid.removeVar(var)
        # XXX should have a better way to track these derived data columns
        for name in ['obsFlux','obsFluxErr','obsMag','obsMagErr']:
            try:
                qsoGrid.data.remove_column(name)
            except:
                pass
    # IGM transmission spectra depend on a (possibly) pre-computed grid,
    # which must be regenerated
    try:
        hiVar = qsoGrid.getVars(grids.HIAbsorptionVar)[0]
        fmodel,nlos,kwargs = hiVar.varmeta
        igmGrid = hiforest.IGMTransmissionGrid(wave,fmodel,nlos,**kwargs)
        hiVar.set_forest_grid(igmGrid)
    except IndexError:
        # no forest
        pass
    # Fe template spectra depend on a (possibly) pre-computed grid,
    # which must be regenerated
    try:
        feVar = qsoGrid.getVars(grids.FeTemplateVar)[0]
        kwargs = feVar.varmeta
        fetempl = grids.VW01FeTemplateGrid(qsoGrid.z,wave,**kwargs)
        feVar.set_template_grid(fetempl)
    except IndexError:
        # no forest
        pass
    if retParams:
        return qsoGrid,qsoGrid.simPars
    #
    return qsoGrid

def load_sim_output(simFileName,outputDir='.',with_spec=True):
    '''Load the output from a previous simulation. Returns only the
    values associated with model variables, unless with_spec=True, in which
    case the spectra are returned also (assuming they have been saved).
    '''
    simdat = restore_qso_grid(simFileName,None,outputDir=outputDir)
    if with_spec:
        if simFileName.endswith('.fits'):
            fn = simFileName
        else:
            fn = simFileName+'_spectra.fits'
        wave,sp = load_spectra(fn,outputDir)
        qsos = hstack([simdat.data,Table(dict(spec=sp))])
        return wave,qsos
    else:
        return simdat.data



def buildWaveGrid(*args,**kwargs):
    '''DEPRECATED since v1.3'''
    warnings.warn("access to buildWaveGrid is deprecated",
                  DeprecationWarning)
    return _build_wave_grid(*args,**kwargs)

def buildQsoGrid(*args,**kwargs):
    '''DEPRECATED since v1.3'''
    warnings.warn("access to buildQsoGrid is deprecated",
                  DeprecationWarning)
    return _build_qso_grid(*args,**kwargs)

def buildForest(*args,**kwargs):
    '''DEPRECATED since v1.3'''
    warnings.warn("access to buildForest is deprecated",
                  DeprecationWarning)
    return _build_forest(*args,**kwargs)

def buildContinuumModels(*args,**kwargs):
    '''DEPRECATED since v1.3'''
    warnings.warn("access to buildContinuumModels is deprecated",
                  DeprecationWarning)
    return _build_continua(*args,**kwargs)

def buildEmissionLineGrid(*args,**kwargs):
    '''DEPRECATED since v1.3'''
    warnings.warn("access to buildEmissionLineGrid is deprecated",
                  DeprecationWarning)
    return _build_emission_line_grid(*args,**kwargs)

def buildDustGrid(*args,**kwargs):
    '''DEPRECATED since v1.3'''
    warnings.warn("access to buildDustGrid is deprecated",
                  DeprecationWarning)
    return _build_dust_grid(*args,**kwargs)

def buildFeatures(*args,**kwargs):
    '''DEPRECATED since v1.3'''
    warnings.warn("access to buildFeatures is deprecated",
                  DeprecationWarning)
    return _build_features(*args,**kwargs)

def buildQsoSpectrum(*args,**kwargs):
    '''DEPRECATED since v1.3'''
    warnings.warn("buildQsoSpectrum is deprecated, use build_qso_spectrum",
                  DeprecationWarning)
    return build_qso_spectrum(*args,**kwargs)

def buildGrpSpectra(*args,**kwargs):
    '''DEPRECATED since v1.3'''
    warnings.warn("access to buildGrpSpectra is deprecated",
                  DeprecationWarning)
    return _build_group_spectra(*args,**kwargs)

def buildSpectraBySightLine(*args,**kwargs):
    '''DEPRECATED since v1.3, use build_spectra_by_sightline'''
    warnings.warn("buildSpectraBySightLine is deprecated, "
                  "use build_spectra_by_sightline instead",
                  DeprecationWarning)
    return build_spectra_by_sightline(*args,**kwargs)

def buildSpecWithPhot(*args,**kwargs):
    '''DEPRECATED since v1.3'''
    warnings.warn("access to buildSpecWithPhot is deprecated",
                  DeprecationWarning)
    return _build_spectra_with_phot(*args,**kwargs)

def buildSpectraBulk(*args,**kwargs):
    '''DEPRECATED since v1.3, use build_spectra_from_grid'''
    warnings.warn("buildSpectraBulk is deprecated, "
                  "use build_spectra_from_grid instead",
                  DeprecationWarning)
    return build_spectra_from_grid(*args,**kwargs)

def qsoSimulation(*args,**kwargs):
    '''DEPRECATED since v1.3, use qso_simulation'''
    warnings.warn("qsoSimulation is deprecated, "
                  "use qso_simulation instead",
                  DeprecationWarning)
    return qso_simulation(*args,**kwargs)

def readSimulationData(*args,**kwargs):
    '''DEPRECATED since v1.3, use restore_qso_grid'''
    warnings.warn("readSimulationData is deprecated ('clean' is ignored), "
                  "use restore_qso_grid instead",
                  DeprecationWarning)
    kwargs.pop('clean')
    return restore_qso_grid(*args,**kwargs)

