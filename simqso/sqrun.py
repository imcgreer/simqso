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
from . import sqmodels

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
    kcorrType = gridPars.get('InitialKCorrection','Continuum')
    if kcorrType == 'Continuum':
        kcorr = sqbase.ContinuumKCorr(gridPars['ObsBand'],
                                      gridPars['RestBand'])
    elif kcorrType == 'DefaultEmissionLine':
        kcorr = sqbase.EmissionLineKCorr(gridPars['ObsBand'],
                                         gridPars['RestBand'])
    else:
        raise ValueError
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


def buildContinuumModels(qsoGrid,simParams,verbose=0):
    continuumParams = simParams['QuasarModelParams']['ContinuumParams']
    reseed(continuumParams)
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

def buildDustGrid(qsoGrid,simParams,verbose=0):
    if verbose > 0:
        print('... building dust extinction grid')
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
#                         fraction=dustParams.get('DustLOSfraction',1.0))
    qsoGrid.addVar(dustVar)


def buildFeatures(qsoGrid,wave,simParams,forest=None,verbose=0):
    buildContinuumModels(qsoGrid,simParams,verbose=verbose)
    qsoParams = simParams['QuasarModelParams']
    if 'EmissionLineParams' in qsoParams:
        buildEmissionLineGrid(qsoGrid,simParams)
    if 'IronEmissionParams' in qsoParams:
        # only option for now is the VW01 template
        scalings = qsoParams['IronEmissionParams'].get('FeScalings')
        feGrid = grids.VW01FeTemplateGrid(qsoGrid.z,wave,scales=scalings)
        qsoGrid.addVar(grids.FeTemplateVar(feGrid))
    if 'DustExtinctionParams' in qsoParams:
        buildDustGrid(qsoGrid,simParams,verbose=verbose)
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

def buildQsoSpectrum(wave,cosmo,specFeatures,obj,iterNum=1,
                     save_components=False):
    spec = sqbase.Spectrum(wave,z=obj['z'])
    if save_components:
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
            if save_components:
                components[feature.name] = spec - base
                base.f_lambda[:] = spec.f_lambda
    # add emission (multiplicative) features
    emspec = sqbase.Spectrum(wave,z=obj['z'])
    if save_components:
        base = sqbase.Spectrum(emspec.wave,emspec.f_lambda.copy(),emspec.z)
    for feature in specFeatures:
        if isinstance(feature,grids.EmissionFeatureVar):
            assocvals = _getpar(feature.get_associated_var(),obj)
            emspec = feature.add_to_spec(emspec,_getpar(feature,obj),
                                         assocvals=assocvals)
            if save_components:
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
        if save_components:
            components[feature.name] = spec - base
            base.f_lambda[:] = spec.f_lambda
    if save_components:
        return spec,components
    else:
        return spec

def buildGrpSpectra(wave,cosmo,specFeatures,photoCache,saveSpectra,
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
            sp = buildQsoSpectrum(wave,cosmo,specFeatures,objGroup[i],iterNum)
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

def buildSpectraBySightLine(wave,qsoGrid,procMap=map,
                            maxIter=1,verbose=0,saveSpectra=False):
    '''Assemble the spectral components of QSOs from the input parameters.

    Parameters
    ----------
    wave : `~numpy.ndarray`
        Input wavelength grid.
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
    build_grp_spec = partial(buildGrpSpectra,wave,qsoGrid.cosmo,
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

def buildSpecWithPhot(wave,cosmo,specFeatures,photoCache,
                      objData,iterNum=None,saveSpectra=False):
    sp = buildQsoSpectrum(wave,cosmo,specFeatures,objData,
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

def buildSpectraBulk(wave,qsoGrid,procMap=map,
                     maxIter=1,verbose=0,saveSpectra=False):
    '''Assemble the spectral components of QSOs from the input parameters.

    Parameters
    ----------
    wave : `~numpy.ndarray`
        Input wavelength grid.
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
        build_one_spec = partial(buildSpecWithPhot,wave,qsoGrid.cosmo,
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


def readSimulationData(fileName,outputDir,retParams=False,clean=False):
    qsoGrid = grids.QsoSimObjects()
    qsoGrid.read(os.path.join(outputDir,fileName+'.fits'),clean=clean)
    simPars = qsoGrid.simPars
    gridPars = simPars['GridParams']
    if True:
        mSampler = grids.FixedSampler(qsoGrid.appMag)
        m = grids.AppMagVar(mSampler,gridPars['ObsBand'])
    try:
        mSampler = grids.FixedSampler(qsoGrid.appMag)
        m = grids.AppMagVar(mSampler,gridPars['ObsBand'])
    except:
        mSampler = grids.FixedSampler(qsoGrid.absMag)
        m = grids.AbsMagVar(mSampler,restWave=gridPars['LumUnits'])
    z = grids.RedshiftVar(grids.FixedSampler(qsoGrid.z))
    qsoGrid.addVars([m,z])
    if retParams:
        return qsoGrid,simPars
    return qsoGrid


def restore_qso_grid(fileName,wave,outputDir='.',**kwargs):
    qsoGrid = grids.QsoSimObjects()
    if not fileName.endswith('.fits'):
        fileName += '.fits'
    qsoGrid.read(os.path.join(outputDir,fileName),**kwargs)
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
    #
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
    noPhotoMap : bool
        skip the simulation of observed photometry [default:False]
    outputDir : str
        write files to this directory [default:'./']
    nproc : int
        number of processes to use [default: 1]
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
    wave = buildWaveGrid(simParams)
    reseed(simParams)
    if nproc > 1:
        pool = multiprocessing.Pool(nproc)
        procMap = pool.map
    else:
        procMap = map
    timerLog = sqbase.TimerLog()
    try:
        qsoGrid,simParams = readSimulationData(simParams['FileName'],
                                               outputDir,retParams=True,
                                               clean=True)
    except IOError:
        if verbose > 0:
            print(simParams['FileName']+' output not found')
        if 'GridFileName' in simParams:
            if verbose > 0:
                print('restoring grid from ',simParams['GridFileName'])
            try:
                qsoGrid = readSimulationData(simParams['GridFileName'],
                                             outputDir)
            except IOError:
                if verbose > 0:
                    print(simParams['GridFileName'],' not found, generating')
                qsoGrid = buildQsoGrid(simParams)
                qsoGrid.write(simParams,outputDir,
                              simParams['GridFileName']+'.fits')
        else:
            if verbose > 0:
                print('generating QSO grid')
            qsoGrid = buildQsoGrid(simParams)
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
        forest = buildForest(wave,qsoGrid.z,simParams,outputDir)
    else:
        forest = None
    if forestOnly:
        timerLog.dump()
        return
    #
    if isinstance(forest,hiforest.IGMTransmissionGrid):
        # build sightlines on-the-fly
        buildSpec = buildSpectraBySightLine
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
        buildSpec = buildSpectraBulk
    #
    qsoGrid.loadPhotoMap(simParams['PhotoMapParams']['PhotoSystems'])
    if 'GridForestFile' in simParams:
        forest = hiforest.GridForest(simParams['GridForestFile'],
                                     qsoGrid.photoBands)
    #
    # add the quasar model variables to the grid (does the random sampling)
    #
    buildFeatures(qsoGrid,wave,simParams,forest,verbose=verbose)
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
        reseed(simParams['PhotoMapParams'])
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

def load_sim_output(simFileName,outputDir='.',with_spec=True):
    simdat,par = readSimulationData(simFileName,outputDir,retParams=True)
    if with_spec:
        sp = fits.getdata(os.path.join(outputDir,simFileName+'_spectra.fits'))
        wave = buildWaveGrid(par)
        qsos = hstack([simdat.data,Table(dict(spec=sp))])
        return wave,qsos
    else:
        return simdat.data

def save_spectra(wave,spectra,fileName,outputDir='.',overwrite=True):
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
    if not fileName.endswith('.fits'):
        fileName += '.fits'
    spec,hdr = fits.getdata(fileName,header=True)
    wi = np.arange(spec.shape[-1])
    logwave = hdr['CRVAL1'] + hdr['CD1_1']*(wi-(hdr['CRPIX1']-1))
    wave = np.exp(logwave)
    return wave,spec

def generate_default_binned_forest(fileName,outputDir='.',**kwargs):
    nlos = kwargs.pop('numSightlines',1000)
    zbins = kwargs.pop('zBins',np.arange(0.1,4.6,0.025))
    waverange = kwargs.pop('waverange',(1300.,7000))
    R = kwargs.pop('R',300)
    hiforest.generate_binned_forest(fileName,sqmodels.WP11_model,
                                    nlos,zbins,waverange,R,
                                    outputDir=outputDir,**kwargs)

