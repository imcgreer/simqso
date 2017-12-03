#!/usr/bin/env python

import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.constants import pc
from astropy import units as u
from astropy.table import Table,join
from astropy.stats import sigma_clip

from .sqbase import datadir

trlines_qsfit = {
  'CIVb':'BR_CIV_1549__EW',
  'MgIIb':'BR_MGII_2798__EW',
  'Hbeta':'BR_HB__EW',
  'HAb':'BR_HA__EW',
}

trlines_shen07 = {
  'CIVb':'EW_CIV',
  'MgIIb':'EW_MGII',
  'Hbeta':'EW_BROAD_HB',
  'HAb':'EW_BROAD_HA',
}

def get_qsfit_M1450(qsfit,alpha_nu=-0.4):
    cwave = np.zeros(len(qsfit))
    qsfitCont = np.zeros(len(qsfit))
    for wv in ['1450','2245','3000','4210','5100']:
        contk = 'CONT%s__LUM' % wv
        contwk = 'CONT%s__WAVE' % wv
        ii = np.where((cwave==0) & ~np.isnan(qsfit[contk]))[0]
        if len(ii) > 0:
            cwave[ii] = qsfit[contwk][ii]
            qsfitCont[ii] = qsfit[contk][ii]
    cnu = (cwave*u.Angstrom).to(u.Hz,equivalencies=u.spectral())
    cnu1450 = (1450.*u.Angstrom).to(u.Hz,equivalencies=u.spectral())
    qsfitLnu1450 = (qsfitCont*1e42/cnu)*(cnu1450/cnu)**alpha_nu
    fourpidsqr = 4*np.pi*(10*pc.to('cm').value)**2
    qsfitM1450 = -2.5*np.log10(qsfitLnu1450.value/fourpidsqr) - 48.6
    return qsfitM1450

def compare_qsfit_shen07(line,qsfit,shen07,minEw=10,**kwargs):
    trendFn = kwargs.get('EmissionLineTrendFilename','emlinetrends_v6')
    lineCatalog = Table.read(os.path.join(datadir,trendFn+'.fits'))
    line_i = np.where(lineCatalog['name']==line)[0][0]
    #
    qsfitEw = qsfit[trlines_qsfit[line]]
    qsfitM1450 = get_qsfit_M1450(qsfit)
    ii1 = np.where(~np.isnan(qsfitEw) & ~np.isnan(qsfitM1450) &
                   (qsfitM1450<-19) & (qsfitM1450>-30) &
                   (qsfitEw>minEw))[0]
    x = sigma_clip(np.log10(qsfitEw[ii1]),sigma=3)
    qsfit_mn,qsfit_std = x.mean(),x.std()
    qsfitstr = 'QSFIT: {0:.2f} +/- {1:.2f}'.format(qsfit_mn,qsfit_std)
    ii1 = ii1[~x.mask]
    qsfitEw = np.array(qsfitEw[ii1])
    qsfitM1450 = qsfitM1450[ii1]
    #
    shenEw = shen07[trlines_shen07[line]]
    ii2 = np.where(shenEw>minEw)[0]
    x = sigma_clip(np.log10(shenEw[ii2]),sigma=3)
    shen_mn,shen_std = x.mean(),x.std()
    shenstr = 'SHEN07: {0:.2f} +/- {1:.2f}'.format(shen_mn,shen_std)
    ii2 = ii2[~x.mask]
    shenEw = shenEw[ii2]
    shenM1450 = shen07['MI_Z2'][ii2] + 1.486
    #
    tab1 = qsfit[ii1].copy()
    tab1['QSFIT_M1450'] = qsfitM1450
    tab2 = shen07[ii2].copy()
    tab2['SHEN_M1450'] = shenM1450
    both = join(tab1,tab2,'SDSS_NAME')
    #
    plt.figure(figsize=(8.6,3.0))
    plt.subplots_adjust(0.08,0.15,0.98,0.98,0.35)
    ax = plt.subplot(131)
    n,xx,yy = np.histogram2d(shenM1450,np.log10(shenEw),15)
    plt.contour(n.transpose(),extent=[xx[0],xx[-1],yy[0],yy[-1]],colors='C0')
    n,xx,yy = np.histogram2d(qsfitM1450,np.log10(qsfitEw),15)
    plt.contour(n.transpose(),extent=[xx[0],xx[-1],yy[0],yy[-1]],colors='C1')
    #
    m1450 = np.linspace(-28,-22,10)
    for coeff_j in range(3):
        M_i = m1450 - 1.486 + 0.596
        plt.plot(m1450,np.polyval(lineCatalog['logEW'][line_i,coeff_j],M_i),
                 c='C2',ls='--')
    #
    ax.text(0.07,0.21,line,transform=ax.transAxes)
    ax.text(0.07,0.14,shenstr,transform=ax.transAxes)
    ax.text(0.07,0.07,qsfitstr,transform=ax.transAxes)
    plt.xlabel('M1450')
    plt.ylabel('log EW')
    plt.subplot(132)
    plt.hexbin(both['SHEN_M1450'],both['QSFIT_M1450'],
               gridsize=25,bins='log',cmap='gray_r')
    plt.plot([-28,-22],[-28,-22],c='r')
    plt.xlabel('Shen M1450')
    plt.ylabel('QSFit M1450')
    plt.subplot(133)
    plt.hexbin(np.log10(both[trlines_shen07[line]]),
               np.log10(both[trlines_qsfit[line]]),
               gridsize=25,bins='log',cmap='gray_r')
    plt.plot([1.0,3],[1.0,3],c='r')
    plt.xlabel('Shen EW')
    plt.ylabel('QSFit EW')

