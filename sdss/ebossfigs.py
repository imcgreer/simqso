#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from astropy.stats import sigma_clip

from simqso import linetemplates
import ebosscore
import ebossmodels

def get_mag_z_bins(ebossqsos,simqsos,nz=6,nm=20,bandNum=2):
    zedges = np.linspace(0.9,4.0,nz+1)
    zbins = zedges[:-1] + np.diff(zedges)/2
    medges = np.linspace(17,22.2,nm+1)
    mbins = medges[:-1] + np.diff(medges)/2
    flux = ebossqsos.sdssFluxes[:,bandNum].filled(0).clip(1e-10,np.inf)
    e_mag = 22.5-2.5*np.log10(flux)
    s_mag = simqsos['obsMag'][:,bandNum]
    e_zi = np.digitize(ebossqsos.specz,zedges)
    s_zi = np.digitize(simqsos['z'],zedges)
    return dict(mbins=mbins,zbins=zbins,medges=medges,zedges=zedges,
                ebossMag=e_mag,simMag=s_mag,eboss_zi=e_zi,sim_zi=s_zi)

def _init_fig(nz=6):
    if nz==6:
        plt.figure(figsize=(7.5,7))
        plt.subplots_adjust(0.08,0.08,0.97,0.95,0.23,0.44)
    else:
        raise NotImplementedError

def binnedlf(ebossqsos,simqsos,**kwargs):
    _init_fig()
    binDat = get_mag_z_bins(ebossqsos,simqsos,**kwargs)
    for i,z in enumerate(binDat['zbins']):
        ax = plt.subplot(3,2,i+1)
        ii = np.where(binDat['eboss_zi']==i+1)[0]
        n_e,_ = np.histogram(binDat['ebossMag'][ii],binDat['medges'])
        ii = np.where((binDat['sim_zi']==i+1) & simqsos['selected'])[0]
        n_s,_ = np.histogram(binDat['simMag'][ii],binDat['medges'])
        ii = np.where(binDat['sim_zi']==i+1)[0]
        n_t,_ = np.histogram(binDat['simMag'][ii],binDat['medges'])
        #
        ii = np.where(n_t>0)[0]
        plt.plot(binDat['mbins'][ii],n_t[ii],label='sim-int')
        ii = np.where(n_s>0)[0]
        plt.plot(binDat['mbins'][ii],n_s[ii],
                 drawstyle='steps-mid',label='sim-obs')
        ii = np.where(n_e>0)[0]
        plt.errorbar(binDat['mbins'][ii],n_e[ii],np.sqrt(n_e[ii]),
                     fmt='s',mfc='none',mec='0.2',ms=5,
                     ecolor='0.2',label='DR14qso')
        if i==0:
            plt.legend()
        plt.yscale('log')
        plt.xlim(22.5,16.9)
        ax.set_title(r'$%.1f<z<%.1f$' %
                     (binDat['zedges'][i],binDat['zedges'][i+1]))
        if i%2==0:
            ax.set_ylabel('N')
        if i>=4:
            ax.set_xlabel('r mag')

def colorcolor(ebossqsos,simqsos,which='optwise',**kwargs):
    _init_fig()
    b = ebosscore.BandIndexes(simqsos)
    if which=='optwise':
        simqsos = ebosscore.get_sim_optwise_mags(simqsos)
        f_opt,f_WISE = ebossqsos.get_optwise()
        sim_fr1 = simqsos['obsFlux'][:,b('g')] / simqsos['obsFlux'][:,b('i')]
        sim_fr2 = simqsos['f_opt'] / simqsos['f_WISE']
        eboss_fr1 = ebossqsos.sdssFluxes[:,1] / ebossqsos.sdssFluxes[:,3]
        eboss_fr2 = f_opt / f_WISE
    else:
        fr1,fr2 = which.split(',')
        b11,b12 = fr1.split('/')
        b21,b22 = fr2.split('/')
        sim_fr1 = simqsos['obsFlux'][:,b(b11)] / simqsos['obsFlux'][:,b(b12)]
        sim_fr2 = simqsos['obsFlux'][:,b(b21)] / simqsos['obsFlux'][:,b(b22)]
        frat,names = ebossqsos.extract_features(['sdss','wise'],
                                                ratios='neighboring')
        eboss_fr1 = frat[:,names.index(fr1.lower())]
        eboss_fr2 = frat[:,names.index(fr2.lower())]
    binDat = get_mag_z_bins(ebossqsos,simqsos,**kwargs)
    def plotit(x,y,which,nbins=15,**kwargs):
        x = x.clip(0,10)
        y = y.clip(0,10)
        x = sigma_clip(x,iters=2,sigma=5)
        y = sigma_clip(y,iters=2,sigma=5)
        ii = np.where(~x.mask & ~y.mask)
        x = x[ii].filled()
        y = y[ii].filled()
#        xr,yr = {'optwise':[(0.0,1.3),(-0.02,0.08)]}[which]
        n,xx,yy = np.histogram2d(x,y,nbins)#,[xr,yr])
        plt.contour(n.T,extent=[xx[0],xx[-1],yy[0],yy[-1]],**kwargs)
    for i,z in enumerate(binDat['zbins']):
        ax = plt.subplot(3,2,i+1)
        ii = np.where(binDat['eboss_zi']==i+1)[0]
        plotit(eboss_fr1[ii],eboss_fr2[ii],which,colors='k')
        ii = np.where((binDat['sim_zi']==i+1) & simqsos['selected'])[0]
        plotit(sim_fr1[ii],sim_fr2[ii],which,colors='C1')
        if which=='optwise':
            xx = np.linspace(*tuple(ax.get_xlim()+(25,)))
            plt.plot(xx,xx*10**(-3/2.5),c='m')
        if i==0:
            plt.legend()
        ax.set_title(r'$%.1f<z<%.1f$' %
                     (binDat['zedges'][i],binDat['zedges'][i+1]))
#        if i%2==0:
#            ax.set_ylabel('N')
#        if i>=4:
#            ax.set_xlabel('r mag')

def intrinsic_lf(simqsos,qlf,nz=6,nm=20,skyArea=3000.):
    _init_fig()
    plt.subplots_adjust(left=0.11)
    #
    zedges = np.linspace(0.9,4.0,nz+1)
    zbins = zedges[:-1] + np.diff(zedges)/2
    medges = np.linspace(-29,-21.5,nm+1)
    mbins = medges[:-1] + np.diff(medges)/2
    m1450 = simqsos['absMag']
    zi = np.digitize(simqsos['z'],zedges)
    dm = np.diff(medges)[0]
    dz = np.diff(zedges)[0]
    dVdz = qlf.cosmo.differential_comoving_volume(zbins).value
    dO = (skyArea/41253)*4*np.pi
    #
    for i,z in enumerate(zbins):
        ax = plt.subplot(3,2,i+1)
        ii = np.where((zi==i+1) & simqsos['selected'])[0]
        n_s,_ = np.histogram(m1450[ii],medges)
        ii = np.where(zi==i+1)[0]
        n_t,_ = np.histogram(m1450[ii],medges)
        #
        dVdzdmdzdO = dVdz[i] * dm * dz * dO
        e_s = np.sqrt(n_s.astype(float)) / dVdzdmdzdO
        e_t = np.sqrt(n_t.astype(float)) / dVdzdmdzdO
        n_s = n_s.astype(float) / dVdzdmdzdO
        n_t = n_t.astype(float) / dVdzdmdzdO
        #
        ii = np.where(n_t>0)[0]
        plt.errorbar(mbins[ii],n_t[ii],e_t[ii],
                     fmt='s',mfc='none',mec='0.2',ms=5,
                     ecolor='0.2',label='sim-int')
        plt.plot(medges,10**qlf(medges,z),label='qlf')
        ii = np.where(n_s>0)[0]
        plt.plot(mbins[ii],n_s[ii])
        if i==0:
            plt.legend()
        plt.yscale('log')
        plt.xlim(-21.2,-30)
        ax.set_title(r'$%.1f<z<%.1f$' % (zedges[i],zedges[i+1]))
        if i%2==0:
            ax.set_ylabel(r'$\Phi(M)$')
        if i>=4:
            ax.set_xlabel(r'$M_{1450}$')

def compare_qsfit_lines(simqsos,qsfit,line):
    #
    j = simqsos.meta['LINENAME'].split(',').index(line)
    simEw = simqsos['emLines'][:,j,1]
    simM1450 = simqsos['absMag']
    #
    qsfitEw = qsfit[linetemplates.trlines_qsfit[line]]
    qsfitM1450 = linetemplates.get_qsfit_M1450(qsfit)
    ii = np.where(~np.isnan(qsfitEw) & ~np.isnan(qsfitM1450))[0]
    qsfitEw = np.array(qsfitEw[ii])
    qsfitM1450 = np.array(qsfitM1450[ii])
    #
    def plotit(m1450,ew,nbins=15,**kwargs):
        ii = np.where((ew>0) & (m1450>-30) & (m1450<-19))[0]
        logew = np.log10(ew[ii])
        n,xx,yy = np.histogram2d(m1450[ii],logew,nbins)
        plt.contour(n.T,extent=[xx[0],xx[-1],yy[0],yy[-1]],**kwargs)
    #
    plt.figure()
    plotit(qsfitM1450,qsfitEw,colors='k')
    plotit(simM1450,simEw,colors='C1')

def model_spectrum(model,**kwargs):
    from simqso.sqbase import fixed_R_dispersion
    from simqso.sqrun import buildQsoSpectrum
    from simqso import sqgrids as grids
    wave = fixed_R_dispersion(0.1e4,10e4,500)
    m,z = np.array([-25]),np.array([1.0])
    M = grids.AbsMagVar(grids.FixedSampler(m),restWave=1450)
    z = grids.RedshiftVar(grids.FixedSampler(z))
    qsos = grids.QsoSimPoints([M,z],
                              cosmo=ebosscore.dr9cosmo,
                              units='luminosity')
    qsos = ebosscore.runsim(model,None,None,qsos,wave=wave,
                            const=True,nophot=True)
    specFeatures = qsos.getVars(grids.SpectralFeatureVar)
    spec,comp = buildQsoSpectrum(wave,qsos.cosmo,specFeatures,
                                 qsos.data[0],save_components=True)
    spec.wave /= 2.0
    return spec,comp,qsos

def model_spectrum_z(model,**kwargs):
    from simqso.sqbase import datadir
    from astropy.io import fits
    #
    filterdata = fits.open(datadir+'filtercurves.fits')
    spec,comp,qsos = model_spectrum(model,**kwargs)
    #
    wscl = 1e-4
    fscl = 1e12
    rflam = spec.wave
    #
    filts = ['SDSS-'+b for b in 'UGRIZ'] + \
            ['UKIDSS-'+b for b in 'JHK'] + \
            ['WISE-'+b for b in ['W1','W2']]
    contFeatures = qsos.getVars(grids.ContinuumVar)
    plt.ioff()
    for z in np.linspace(1.0,4.0,11):
        plt.figure()
        plt.subplots_adjust(0.1,0.1,0.98,0.94)
        ax = plt.subplot(111)
        for cf in contFeatures:
            k = cf.name
            plt.plot(rflam*wscl,rflam*comp[k].f_lambda*fscl,label=k)
            if isinstance(cf,grids.BrokenPowerLawContinuumVar):
                plcont = comp[k].f_lambda
        for k in ['emLines','fetempl']:
            plt.plot(rflam*wscl,
                     0.5*rflam*(1+comp[k].f_lambda)*fscl*plcont,label=k)
        plt.plot(rflam*wscl,rflam*spec.f_lambda*fscl,label='total')
        for filt in filts:
            fdat = filterdata[filt].data
            ffscl = 1/fdat['Rlam'].max()
            plt.fill_between(fdat['lam']/(1+z)*wscl,
                             fdat['Rlam']*ffscl,1e-1,alpha=0.2,label=None)
        plt.xscale('log')
        plt.yscale('log')
        plt.ylim(1e-1,2.5)
        plt.title(r'$z=%.2f$'%z)
        ax.set_xticks([0.09,0.12,0.15,0.19,0.28,0.50,0.65,1.0,2.0])
        plt.xlim(0.3/(1+z),6.0/(1+z))
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
        plt.legend(ncol=3,fontsize=9,loc='upper right')
        plt.xlabel('wave [$\mu$m]')
        plt.ylabel(r'$\lambda{f}_\lambda$')
        plt.savefig('meanspec_z%.1f.pdf'%z)
        plt.close()
    plt.ion()

def compare_model_spec(models='all'):
    plt.figure()
    if models == 'all':
        models = list(ebossmodels.qso_models.keys())
    for name in models:
        model = ebossmodels.qso_models[name]
        spec,comp,_ = model_spectrum(model)
        i = np.searchsorted(spec.wave,1450)
        plt.plot(spec.wave/1e4,
                 spec.wave*spec.f_lambda/spec.f_lambda[i],
                 label=name)
    plt.legend()
    plt.xscale('log')
    plt.yscale('log')

