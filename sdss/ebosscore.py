#!/usr/bin/env python

import os,sys
import itertools
import subprocess
import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
from astropy.cosmology import FlatLambdaCDM
from astropy.table import Table,hstack,vstack
from simqso import sqbase,sqrun,sqmodels,sqphoto,lumfun,hiforest
from simqso import sqgrids as grids
from simqso.sqbase import continuum_kcorr,fixed_R_dispersion

import ebossmodels

dr9cosmo = FlatLambdaCDM(70,1-0.7,name='BOSSDR9')

def make_forest_grid(forestFile,forestType,wave,z,
                     nlos=5000,outputDir='.',nproc=6,**kwargs):
    zbins = kwargs.pop('zBins',np.arange(0.9,4.6,0.01))
    waverange = kwargs.pop('waverange',(2900.,9000))
    R = kwargs.pop('R',300)
    forestModel = sqmodels.WP11_model
    if forestType == 'meanmag':
        photSys = [('SDSS','Legacy','ugri')]
        photoMap = sqphoto.LazyPhotoMap(photSys)
        hiforest.generate_grid_forest(forestFile,forestModel,
                                      nlos,zbins,waverange,R,
                                      photoMap,outputDir=outputDir,
                                      nproc=nproc,**kwargs)
    else:
        forest = hiforest.IGMTransmissionGrid(wave,forestModel,nlos,
                                              zmax=z.max(),**kwargs)
        losSampler = grids.RandomSubSampler(forest.numSightLines)
        losMap = losSampler.sample(qsoGrid.nObj)
        forest.write(forestFile,outputDir,losMap=losMap,z_em=z)

def sample_qlf(qlf,mrange=(17,22),zrange=(0.9,4.0),
               skyArea=3000,emlinekcorr=True):
    obsBand = 'SDSS-r'
    restBand = 1450
    if emlinekcorr:
        kcorr = sqbase.EmissionLineKCorr(obsBand,restBand)
    else:
        kcorr = sqbase.ContinuumKCorr(obsBand,restBand)
    qlfGrid = grids.generateQlfPoints(qlf,mrange,zrange,kcorr,
                                      skyArea=skyArea,fast_sample=True)
    return qlfGrid

photSys = [ ('SDSS','Legacy'), ('UKIRT','UKIDSS_LAS'), ('WISE','AllWISE') ]

def runsim(model,fileName,forest,qsoGrid,
           foresttype='meanmag',nlos=5000,
           maxIter=3,procMap=map,wave=None,
           medianforest=False,const=False,
           nophot=False,withspec=False,outputDir='.'):
    np.random.seed(12345)
    if wave is None:
        wave = fixed_R_dispersion(0.3e4,6e4,500)
    #
    qsoGrid = ebossmodels.add_continuum(qsoGrid,model['continuum'],
                                        const=const)
    if model.get('dustem'):
        qsoGrid = ebossmodels.add_dust_emission(qsoGrid,model['dustem'],
                                                const=const)
    if model.get('emlines'):
        qsoGrid = ebossmodels.add_emission_lines(qsoGrid,model['emlines'],
                                                 const=const)
    if model.get('iron'):
        qsoGrid = ebossmodels.add_iron(qsoGrid,wave,model['iron'],
                                       const=const)
    if model.get('dustext'):
        qsoGrid = ebossmodels.add_dust_extinction(qsoGrid,model['dustext'],
                                                  const=const)
    #
    qsoGrid.loadPhotoMap(photSys)
    #
    if not forest is None:
        for extn in ['','.fits','.fits.gz']:
            forestFile = forest+extn
            if os.path.exists(forestFile):
                break
        if not os.path.exists(forestFile):
            print('forest file {} does not exist, generating...'.format(forest))
            make_forest_grid(forest,foresttype,wave,qsoGrid.z,
                             nlos=nlos,outputDir=outputDir)
        if foresttype == 'meanmag':
            forest = hiforest.GridForest(forestFile,
                                         qsoGrid.photoBands,
                                         median=medianforest)
            forestVar = grids.SightlineVar(forest)
        else:
            forest = hiforest.CachedIGMTransmissionGrid(forestFile)
            forestVar = grids.HIAbsorptionVar(forest,losMap=forest.losMap)
        qsoGrid.addVar(forestVar)
    #
    qsoGrid,spectra = sqrun.buildSpectraBulk(wave,qsoGrid,
                                             procMap=procMap,
                                             maxIter=maxIter,
                                             saveSpectra=withspec,
                                             verbose=0)
    #
    if not nophot:
        photoData = sqphoto.calcObsPhot(qsoGrid.synFlux,qsoGrid.photoMap)
        qsoGrid.addData(photoData)
    #
    if fileName is not None:
        qsoGrid.write(fileName)
    if withspec:
        return qsoGrid,wave,spectra
    else:
        return qsoGrid

class BandIndexes(object):
    def __init__(self,simQsoTab):
        self.bands = simQsoTab.meta['OBSBANDS'].split(',')
        self.shortNames = [ b.split('-')[-1] for b in self.bands ]
    def __call__(self,b):
        return self.shortNames.index(b)

def get_sim_optwise_mags(simqsos):
    b = BandIndexes(simqsos)
    # populate the fields needed for the mid-IR-optical color cuts
    simqsos['f_opt'] = ( 1.0*simqsos['obsFlux'][:,b('g')] +
                         0.8*simqsos['obsFlux'][:,b('r')] +
                         0.6*simqsos['obsFlux'][:,b('i')] ) / 2.4
    # need conversion to Vega for WISE
    f_W1_Vega = simqsos['obsFlux'][:,b('W1')] * 10**(0.4*(2.699))
    f_W2_Vega = simqsos['obsFlux'][:,b('W2')] * 10**(0.4*(3.339))
    simqsos['f_WISE'] = ( 1.0*f_W1_Vega + 0.5*f_W2_Vega ) / 1.5
    return simqsos

def run_xdqso(args):
    fileName,xdFile,verbose = args
    if not os.path.exists(xdFile):
        FNULL = open(os.devnull,'w')
        cmd = ["idl","-e","xdprob,'{0}','{1}'".format(fileName,xdFile)]
        if verbose:
            print("executing "+" ".join(cmd))
        subprocess.call(cmd,stdout=FNULL,stderr=FNULL)

def apply_selection_fun(fileName,verbose=0,redo=False,procMap=None,nsplit=1):
    xdFile = fileName.replace('.fits','__xdtmp.fits')
    qsos = Table.read(fileName)
    if 'PQSO' not in qsos.colnames or redo:
        if nsplit == 1:
            run_xdqso( (fileName,xdFile,verbose) )
            xdqso = Table.read(xdFile)
            os.remove(xdFile)
        else:
            splits = np.array_split(np.arange(len(qsos)),nsplit)
            procArgs = []
            for i,split in enumerate(splits):
                tmpFn = fileName.replace('.fits','__%02d.fits'%i)
                tmpXdFn = xdFile.replace('.fits','__%02d.fits'%i)
                qsos[split].write(tmpFn,overwrite=True)
                procArgs.append( (tmpFn,tmpXdFn,verbose) )
            procMap(run_xdqso,procArgs)
            xdqso = []
            for i,(tmpFn,tmpXdFn,_) in enumerate(procArgs):
                xdqso.append( Table.read(tmpXdFn) )
                os.remove(tmpFn)
                os.remove(tmpXdFn)
            xdqso = vstack(xdqso)
        if redo and 'PQSO' in qsos.colnames:
            qsos.remove_columns(xdqso.colnames)
        for col in xdqso.colnames:
            xdqso[col] = xdqso[col].astype(np.float32)
        qsos = hstack([qsos,xdqso])
    b = BandIndexes(qsos)
    # populate the fields needed for the mid-IR-optical color cuts
    qsos = get_sim_optwise_mags(qsos)
    #
    sel = True
    if verbose:
        print("{:7d} quasars at start".format(len(qsos)))
    # roughly the FIBER2MAG i>17 cut
    sel &= qsos['obsMag'][:,b('i')] > 17
    if verbose:
        print("{:7d} after fiberMag_i>17 cut".format(sel.sum()))
    # g < 22 OR r < 22
    sel &= ( (qsos['obsMag'][:,b('g')] < 22) | 
             (qsos['obsMag'][:,b('r')] < 22) )
    if verbose:
        print("{:7d} after g<22 OR r<22 cut".format(sel.sum()))
    # XDQSOz probability cut
    sel &= qsos['PQSO'] > 0.2
    if verbose:
        print("{:7d} after XDQSOz cut".format(sel.sum()))
    # optical--mid-IR color cut
    sel &= ( qsos['f_WISE']*qsos['obsFlux'][:,b('g')] >
              qsos['f_opt']*qsos['obsFlux'][:,b('i')]*10**(3.0/2.5) )
    if verbose:
        print("{:7d} after optical--mid-IR color cut".format(sel.sum()))
    #
    qsos['selected'] = sel
    qsos.write(fileName,overwrite=True)

def qlf_ranges(model,simName,forestFile,qlf,skyArea,**kwargs):
    np.random.seed(12345)
    for dex in [0.2,0.5,1.0]:
        for sgn in [-1,1]:
            fn = simName+'_logphi_%+04.1f'%(sgn*dex)
            print('running simulation {}'.format(fn))
            initpar = qlf.getpar()
            par = initpar.copy()
            par[1] += sgn*dex
            qsos = sample_qlf(qlf,skyArea=skyArea)
            runsim(model,fn,forestFile,qsos,**kwargs)
            apply_selection_fun(fn+'.fits',verbose=1,redo=True)

def emline_ranges(model,modelName,line,forestFile,qlf,skyArea,**kwargs):
    model['emlines'] = ebossmodels.emline_models[model['emlines']]
    scls = np.linspace(0.5,1.5,6)
    #for scl in [0.5,0.75,1.0,1.25,1.5]:
    for scl1,scl2 in itertools.product(scls,scls):
        #fn = '_'.join([modelName,line,'%.2f'%scl])
        fn = '_'.join([modelName,line,'%.2f'%scl1,'%.2f'%scl2])
        fn = os.path.join(kwargs.get('outputDir','.'),fn+'.fits')
        print('running simulation {}'.format(fn))
        if not os.path.exists(fn) or args.redo:
            np.random.seed(12345)
            qsos = sample_qlf(qlf,skyArea=skyArea)
            #np.random.seed(int(pow(2,10**scl)))
            np.random.seed(int(pow(2,scl1+2*scl2)))
            #model['emlines']['scaleEWs'][line] = scl
            model['emlines']['scaleEWs']['LyAb'] = scl1
            model['emlines']['scaleEWs']['CIVb'] = scl2
            print(model)
            runsim(model,fn,forestFile,qsos,**kwargs)
            apply_selection_fun(fn,verbose=1,redo=True)

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser(
                              description='run eboss quasar simulations.')
    parser.add_argument('model',type=str,
        help='name of quasar model')
    parser.add_argument('--forest',type=str,default='sdss_forest_grid',
        help='file containing forest spectra (default:sdss_forest_grid)')
    parser.add_argument('--foresttype',type=str,default='meanmag',
        help="type of forest spectra (default: 'meanmag', also 'fullres')")
    parser.add_argument('-o','--outputdir',type=str,default='.',
        help='output directory (default:.)')
    parser.add_argument('--suffix',type=str,
        help='suffix to add to filename ( --> <MODEL>_<SUFFIX>.fits )')
    parser.add_argument('-p','--processes',type=int,default=7,
        help='number of processes to create')
    parser.add_argument('-R','--redo',action='store_true',
        help='force redo if output exists')
    parser.add_argument('-s','--seed',type=int,default=12345,
        help='random seed')
    parser.add_argument('--qlf',type=str,default='bossdr9',
        help='seed QLF model (default: BOSS DR9 PLE-LEDE)')
    parser.add_argument('--skyarea',type=float,default=3000,
        help='area of sky in simulation (default: 3000 deg2)')
    parser.add_argument('--nlos',type=int,default=5000,
        help='number of sightlines for forest spectra (default: 5000)')
    parser.add_argument('--noselection',action='store_true',
        help='do not calculate selection function')
    parser.add_argument('--testranges',action='store_true',
        help='test range of parameter values')
    parser.add_argument('--continuum',type=str,
        help='specify continuum model')
    parser.add_argument('--emlines',type=str,
        help='specify emission line model')
    parser.add_argument('--iron',type=str,
        help='specify iron template')
    parser.add_argument('--dustem',type=str,
        help='specify dust emission model')
    parser.add_argument('--dustext',type=str,
        help='specify dust extinction model')
    args = parser.parse_args()
    assert args.foresttype in ['meanmag','fullres']
    if not os.path.exists(args.outputdir):
        os.makedirs(args.outputdir)
    #
    modelName = args.model
    model = ebossmodels.qso_models.get(args.model,{})
    def eval_arg(arg):
        return None if arg == 'none' else arg
    if args.continuum:
        model['continuum'] = eval_arg(args.continuum)
    if args.emlines:
        model['emlines'] = eval_arg(args.emlines)
    if args.dustem:
        model['dustem'] = eval_arg(args.dustem)
    if args.iron:
        model['continuum'] = eval_arg(args.iron)
    if args.dustext:
        model['continuum'] = eval_arg(args.dustext)
    #
    if args.qlf=='bossdr9':
        qlf = sqmodels.BOSS_DR9_PLEpivot(cosmo=dr9cosmo)
    else:
        raise ValueError
    if args.processes == 1:
        procMap = map
    else:
        pool = multiprocessing.Pool(args.processes)
        procMap = pool.map
    #
    if args.testranges:
        #qlf_ranges(model,simName,args.forest,qlf,args.skyarea,
        #           nproc=args.processes)
        emline_ranges(model,modelName,'LyAb',args.forest,qlf,args.skyarea,
                      foresttype=args.foresttype,
                      procMap=procMap,outputDir=args.outputdir)
    else:
        np.random.seed(args.seed)
        fn = modelName
        if args.suffix:
            fn += '_'+args.suffix
        fn = os.path.join(args.outputdir,fn+'.fits')
        if os.path.exists(fn) and not args.redo:
            sys.exit(0)
        qsoGrid = sample_qlf(qlf,skyArea=args.skyarea)
        runsim(model,fn,args.forest,qsoGrid,
               foresttype=args.foresttype,nlos=args.nlos,
               procMap=procMap,outputDir=args.outputdir)
        if not args.noselection:
            apply_selection_fun(fn,verbose=1,procMap=procMap,redo=True,
                                nsplit=args.processes)
    if args.processes > 1:
        pool.close()

