#!/usr/bin/env python

import os,sys
from copy import deepcopy
from collections import defaultdict
import numpy as np
from scipy.stats import binned_statistic
import matplotlib.pyplot as plt
from matplotlib import ticker
from astropy.table import Table,hstack
from simqso import sqgrids as grids

import ebosscore
import ebossfit
import ebossmodels

class percfun(object):
    def __init__(self,pval):
        self.pval = pval
    def __call__(self,x):
        x = x[x<1e20]
        if len(x) < 10:
            return np.nan
        return np.percentile(x,self.pval)

def calc_colorz(z,clrs,pvals,zedges):
    zbins = zedges[:-1] + np.diff(zedges)/2
    colorz = np.zeros((len(pvals),clrs.shape[-1],len(zbins)),dtype=np.float32)
    for i,pval in enumerate(pvals):
        clrz,_,_ = binned_statistic(z,clrs.T,percfun(pval),zedges)
        colorz[i] = clrz
    return colorz

def sim_colorz(simqsos,pvals,zedges,ratios=True,refBand=None,maglim=None):
    b = ebosscore.BandIndexes(simqsos)
    if ratios:
        fluxk = 'Flux'
        d = '/'
    else:
        fluxk = 'Mag'
        d = '-'
    if refBand is None:
        bnames = [ d.join(bb)
                     for bb in zip(b.shortNames[:-1],b.shortNames[1:]) ]
    else:
        bnames = [ b_+d+refBand for b_ in b.shortNames if b_ != refBand ]
    colorz = {}
    for which in ['syn','obs']:
        z = simqsos['z']
        flux = simqsos[which+fluxk]
        sel = np.ones(len(simqsos),dtype=bool)
        if maglim is not None:
            sel &= simqsos['obsMag'][:,b(refBand)] < maglim
        if which=='obs':
            sel &= simqsos['selected']
        flux = flux[sel]
        z = z[sel]
        if ratios:
            if refBand is None:
                clrs = flux[:,:-1] / flux[:,1:]
            else:
                clrs = flux / flux[:,[b(refBand)]]
                clrs = np.delete(clrs,b(refBand),1)
        else:
            clrs = -np.diff(flux,axis=1)
        # needs to be masked to match observations table
        clrs = np.ma.array(clrs)
        colorz[which] = calc_colorz(z,clrs,pvals,zedges)
    tab = Table(colorz)
    return tab,bnames

def ebosscore_colorz(coreqsos,pvals,zedges,maglim=None,
                     refBand=None,ratios='neighboring'):
    photsets = ['sdss','ukidss','wise']
    features,names,refFlux = coreqsos.extract_features(featureset=photsets,
                                                       refband=refBand,
                                                       ratios=ratios)
    zqso = coreqsos.specz
    if maglim:
        refMag = 22.5 - 2.5*np.log10(refFlux.clip(1e-5,np.inf))
        ii = np.where(refMag < maglim)[0]
        features = features[ii]
        zqso = zqso[ii]
    clrs = features.filled(1e20)
    colorz = calc_colorz(zqso,clrs,pvals,zedges)
    return Table(dict(ebosscore=colorz)),names

# mags
#yr = [ (-0.7,4.2), (-0.3,1.7), (-0.15,0.5), (-0.2,0.5), (-0.15,0.75) ]

fratio_yrange = {
  'u/g':(-0.3,1.3), 'g/r':(0.1,1.3), 'r/i':(0.5,1.2), 'i/z':(0.5,1.3),
  'W1/W2':(0.4,1.3),
}

def colorz_compare(simqsos,coreqsos,maglim=None,refBand=None):
    zedges = np.linspace(0.9,4.0,32)
    zbins = zedges[:-1] + np.diff(zedges)/2
    pvals = [25,50,75]
    ratios = 'neighboring' if refBand is None else 'byref'
    colorz,simClrNames = sim_colorz(simqsos,pvals,zedges,
                                    refBand=refBand,maglim=maglim)
    refBand_ = 'i' if refBand is None else refBand
    colorz2,ebossClrNames = ebosscore_colorz(coreqsos,pvals,zedges,
                                             ratios=ratios,refBand=refBand_,
                                             maglim=maglim)
    assert np.all(np.array(simClrNames)==np.array(ebossClrNames))
    colorz = hstack([colorz,colorz2])
    fig = plt.figure(figsize=(9.5,7))
    plt.subplots_adjust(0.055,0.05,0.99,0.99,0.23,0.15)
    for j,clrName in enumerate(simClrNames):
        plt.subplot(4,3,j+1)
        for which in ['syn','obs','ebosscore']:
            c = {'syn':'C0','obs':'C1','ebosscore':'C2'}[which]
            plt.fill_between(zbins,colorz[which][0,j],colorz[which][2,j],
                             color=c,alpha=0.3)
            plt.plot(zbins,colorz[which][1,j],c=c,ls='-')
        plt.xlim(0.85,4.05)
        yr = fratio_yrange.get(clrName)
        if yr:
            plt.ylim(*yr)
        plt.ylabel(clrName,size=9)
    plt.subplot(4,3,12)
    plt.hist(simqsos['z'],zbins,log=True)
    plt.hist(simqsos['z'][simqsos['selected']],zbins,log=True)
    plt.hist(coreqsos.specz,zbins,log=True,alpha=0.5)
    plt.xlim(0.85,4.05)
    plt.ylabel('n(z)',size=9)
    for ax in fig.get_axes():
        plt.setp(ax.get_xticklabels()+ax.get_yticklabels(),fontsize=8)
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.2))

def get_colorz_bins(mrange=(-27,-23),zrange=(0.9,4.0),nm=7,nz=500):
    mbins = np.linspace(*tuple(mrange+(nm,)))
    zbins = np.linspace(*tuple(zrange+(nz,)))
    return mbins,zbins

def model_colorz_tracks(model,forestFile,**kwargs):
    mbins,zbins = get_colorz_bins(**kwargs)
    nm,nz = len(mbins),len(zbins)
    M,z = np.meshgrid(mbins,zbins,indexing='ij')
    M = grids.AbsMagVar(grids.FixedSampler(M.flatten()),restWave=1450)
    z = grids.RedshiftVar(grids.FixedSampler(z.flatten()))
    qsos = grids.QsoSimPoints([M,z],
                              cosmo=ebosscore.dr9cosmo,
                              units='luminosity')
    qsos = ebosscore.runsim(model,None,forestFile,qsos,
                            medianforest=True,const=True,nophot=True)
    synmags = np.array(qsos.data['synMag'].reshape(nm,nz,-1))
    synclrs = -np.diff(synmags,axis=-1)
    synfluxes = np.array(qsos.data['synFlux'].reshape(nm,nz,-1))
    synfratios = synfluxes[...,:-1] / synfluxes[...,1:]
    return dict(mbins=mbins,zbins=zbins,
                synmags=synmags,syncolors=synclrs,
                synfluxes=synfluxes,synfratios=synfratios,
                qsos=qsos)

def colorz_param_trends(modelName,forestFile):
    tab = Table()
    #
    def add_entry(tab,nm,s,cz):
        for k1,k2 in [('mags','synmags'),('clrs','syncolors'),
                      ('fluxes','synfluxes'),('fratio','synfratios')]:
            k = '_'.join([nm,s,k1])
            tab[k] = cz[k2]
    #
    model = deepcopy(ebossmodels.qso_models[modelName])
    cname = model['continuum']
    for j in range(len(ebossmodels.cont_models[cname][0])):
        for ds in [-0.3,0.0,0.3]:
            model['continuum'] = deepcopy(ebossmodels.cont_models[cname])
            model['continuum'][0][j] = (model['continuum'][0][j][0]+ds,None)
            print(ds,model)
            cz = model_colorz_tracks(model,forestFile)
            add_entry(tab,'slope%d'%j,'%+4.1f'%ds,cz)
            print()
    #
    model = deepcopy(ebossmodels.qso_models[modelName])
    emtemplate = ebossmodels.emline_models[
                               model['emlines']]['EmissionLineTrendFilename']
    if emtemplate.endswith('v6'):
        Halpha = 'HA'
    else:
        Halpha = 'Halpha'
    if emtemplate[-2:] in ['v5','v6']:
        LyB = 'LyB'
    else:
        LyB = 'LyB+OVI'
    for l in [LyB,'LyA','CIV','MgII','Hbeta',Halpha]:
        for scl in [0.5,1.0,2.0]:
            model['emlines'] = {'scaleEWs':{},
                                'EmissionLineTrendFilename':emtemplate}
            if l in ['LyA','CIV','MgII','HA']:
                for c in 'bn':
                    model['emlines']['scaleEWs'][l+c] = scl
            else:
                model['emlines']['scaleEWs'][l] = scl
            print(l,model)
            cz = model_colorz_tracks(model,forestFile)
            add_entry(tab,l,'%3.1f'%scl,cz)
        print()
    #
    if 'dustem' in ebossmodels.qso_models[modelName]:
        model = deepcopy(ebossmodels.qso_models[modelName])
        dustnm = model['dustem']
        comps = ebossmodels.dustem_models[dustnm]
        for i,c in enumerate(comps):
            for f in [0.5,1.0,2.0]:
                model['dustem'] = deepcopy(ebossmodels.dustem_models[dustnm])
                pars = model['dustem'][c]
                model['dustem'][c] = [(pars[0][0]*f,None)] + pars[1:]
                print(f,model)
                cz = model_colorz_tracks(model,forestFile)
                add_entry(tab,'%sfrac'%c,'%3.1f'%f,cz)
                print()
        for i,c in enumerate(comps):
            for f in [0.7,1.0,1.3]:
                model['dustem'] = deepcopy(ebossmodels.dustem_models[dustnm])
                pars = model['dustem'][c]
                model['dustem'][c] = [pars[0]] + [(pars[1][0]*f,None)]
                print(f,model)
                cz = model_colorz_tracks(model,forestFile)
                add_entry(tab,'%sT'%c,'%3.1f'%f,cz)
                print()
    return cz['mbins'],cz['zbins'],tab

def plot_trends(modelName,trendFile,coreqsos):
    mbins,zbins = get_colorz_bins()
    eboss_zedges = np.linspace(0.9,4.0,32)
    eboss_zbins = eboss_zedges[:-1] + np.diff(eboss_zedges)/2
    pvals = [25,50,75]
    obs_colorz = ebosscore_colorz(coreqsos,pvals,eboss_zedges)[0]['ebosscore']
    trends = Table.read(trendFile)
    pars = defaultdict(list)
    for c in trends.colnames:
        name,val,which = c.split('_')
        if which=='clrs':
            pars[name].append(val)
    #nrow,ncol = 5,2
    nrow,ncol = 3,2
    m_i = 3
    asmags = False
    f = '-' if asmags else '/'
    clrnames = [b1+f+b2 for b1,b2 in zip('ugri','griz')] + ['W1'+f+'W2']
    pclrs = [ 'C%d'%i for i in range(1,10,2) ]
    for p,vals in pars.items():
        #plt.figure(figsize=(7.5,8))
        plt.figure(figsize=(7.5,5.5))
        plt.subplots_adjust(0.10,0.07,0.98,0.94,0.28)
        plt.figtext(0.5,0.99,p,ha='center',va='top',size=15)
        for pnum,j in enumerate([0,1,2,3,-1],start=1):
            ax = plt.subplot(nrow,ncol,pnum)
            for _i,v in enumerate(vals):
                if asmags:
                    k = '_'.join([p,v,'clrs'])
                else:
                    k = '_'.join([p,v,'fratio'])
                ax.plot(zbins,trends[k][m_i,:,j],label=v,c=pclrs[_i])
            c = 'C0'
            obs_cz = obs_colorz[:,j]
            if asmags:
                obs_cz = -2.5*np.log10(obs_cz.clip(1e-2,np.inf))
            ax.fill_between(eboss_zbins,obs_cz[0],obs_cz[2],
                            color=c,alpha=0.3)
            ax.plot(eboss_zbins,obs_cz[1],c=c,ls='-')
            ax.set_ylabel(clrnames[pnum-1])
            ax.set_ylim(fratio_yrange[clrnames[pnum-1]])
            plt.setp(ax.get_xticklabels()+ax.get_yticklabels(),fontsize=9)
            ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.2))
            if j==0: plt.legend()
        plt.savefig(modelName+'_'+p+'.pdf')
        plt.close()

def plot_model_trends(model='all',forestFile=None):
    coreqsos = ebossfit.eBossQsos() 
    if model=='all':
        models = list(ebossmodels.qso_models.keys())
    else:
        models = [model]
    for modelName in models:
        trendfn = modelName+'_trends.fits'
        if not os.path.exists(trendfn):
            m,z,trends = colorz_param_trends(modelName,forestFile)
            trends.write(trendfn,overwrite=True)
        else:
            trends = Table.read(trendfn)
        plot_trends(modelName,trendfn,coreqsos)

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser(
                              description='run eboss color-z simulations.')
    parser.add_argument('fitsfile',nargs='*',type=str,
        help='input file name(s)')
    parser.add_argument('--forest',type=str,default='sdss_forest_grid',
        help='file containing forest grid (default:sdss_forest_grid)')
    parser.add_argument('-m','--model',type=str,default='bossdr9',
        help='name of quasar model')
    parser.add_argument('--trends',action="store_true",
        help='show parameter color-z mean trends instead of running sim')
    parser.add_argument('--tracks',action="store_true",
        help='show color-z mean trends instead of running sim')
    parser.add_argument('--maglim',type=float,
        help='limiting magnitude')
    parser.add_argument('--refband',type=str,
        help='colors are relative to reference band instead of '+
             'adjoining filters')
    args = parser.parse_args()
    if args.trends:
        plot_model_trends(model=args.model,forestFile=args.forest)
    elif args.tracks:
        model = ebossmodels.qso_models[args.model]
        cz = model_colorz_tracks(model,args.forest)
    else:
        coreqsos = ebossfit.eBossQsos()
        for ff in args.fitsfile:
            print(ff)
            simqsos = Table.read(ff)
            colorz_compare(simqsos,coreqsos,maglim=args.maglim,
                           refBand=args.refband)
            plt.figtext(0.5,0.05,os.path.basename(ff).replace('.fits',''),
                        ha='center',size=15)
            sfx = ''
            if args.maglim:
                sfx += '_ilt%.1f' % args.maglim
            if args.refband:
                sfx += '_ref-%s' % args.refband
            plt.savefig(ff.replace('.fits','_colorz'+sfx+'.pdf'))

