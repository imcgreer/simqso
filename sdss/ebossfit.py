#!/usr/bin/env python

import os,sys
import numpy as np
from sklearn.mixture import GaussianMixture
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy import units as u

def make_coreqso_table(dr14qso,ebosstarg):
    if isinstance(dr14qso,str):
        dr14qso = Table.read(dr14qso)
    if isinstance(ebosstarg,str):
        ebosstarg = Table.read(ebosstarg)
    #
    dr14coo = SkyCoord(dr14qso['RA'],dr14qso['DEC'],unit=u.deg)
    # restrict to CORE quasar targets
    ii = np.where(ebosstarg['EBOSS_TARGET1'] & (1<<10) > 0)[0]
    ebosstarg = ebosstarg[ii]
    ebosstargcoo = SkyCoord(ebosstarg['RA'],ebosstarg['DEC'],unit=u.deg)
    # now identify confirmed quasars from DR14 in the target list
    m1,m2,sep,_ = dr14coo.search_around_sky(ebosstargcoo,2*u.arcsec)
    # for some reason there is a repeated entry...
    _,ii = np.unique(m1,return_index=True)
    dr14qso = dr14qso[m2[ii]]
    # just a sanity check
    jj = np.where(dr14qso['EXTINCTION']>0)[0]
    assert np.allclose(dr14qso['EXTINCTION'][jj],
                       ebosstarg['EXTINCTION'][m1[ii[jj]]],atol=1e-3)
    # extract all the WISE columns from targeting
    wisecols = ['W1_MAG','W1_MAG_ERR',
                'W1_NANOMAGGIES','W1_NANOMAGGIES_IVAR',
                'W2_NANOMAGGIES','W2_NANOMAGGIES_IVAR',
                'HAS_WISE_PHOT']
    # overwriting the DR14Q flux fields because they have invalid entries
    for k in wisecols + ['EXTINCTION','PSFFLUX','PSFFLUX_IVAR']:
        dr14qso[k] = ebosstarg[k][m1[ii]]
    dr14qso.write('ebosscore_dr14q.fits',overwrite=True)

def get_column_ratio(a,j):
    col = a[:,[j]].copy()
    a /= col
    a = np.ma.hstack([a[:,:j],a[:,j+1:]])
    return col,a

class eBossQsos(object):
    def __init__(self,fileName='ebosscore_dr14q.fits',zrange=(0.9,4.0)):
        dat = Table.read(fileName)
        if zrange is not None:
            dat = dat[np.logical_and(dat['Z']>zrange[0],dat['Z']<zrange[1])]
        self.data = dat
        self.set_specz_col('Z')
        self._extract_fluxes()
    def set_specz_col(self,colname):
        self.speczName = colname
        self.specz = self.data[colname]
    def _extract_fluxes(self,ratios='byref',refNum=3):
        # SDSS
        sdssFluxes = np.array(self.data['PSFFLUX'])
        extCorr = np.array(self.data['EXTINCTION'])
        self.sdssFluxes = np.ma.array(sdssFluxes*10**(0.4*extCorr))
        # XXX need the galex ext values
        # Galex
        fuv = np.ma.array(self.data['FUV'],mask=self.data['FUV_IVAR']==0)
        nuv = np.ma.array(self.data['NUV'],mask=self.data['NUV_IVAR']==0)
        self.galexFluxes = np.ma.vstack([fuv,nuv]).transpose()
        # UKIDSS
        self.ukidssFluxes = np.ma.vstack(
                 [ np.ma.array(self.data[b+'FLUX'],
                               mask=self.data['UKIDSS_MATCHED']==0)
                      for b in 'YJHK' ] ).transpose()
        # units are W/m^2/Hz, 1e26 converts to Jy and 5.44 to nanomaggie
        self.ukidssFluxes *= 1e26*10**5.44
        # XXX the fluxes seem to be consistently offset when compared to
        #     catalog Vega magnitudes... why?
        self.ukidssFluxes *= 1.07
        # WISE (convert from Vega)
        # using extinction conversions from ebosstarget_qso_selection.pro
        w1 = np.ma.array(self.data['W1_NANOMAGGIES'],
                         mask=self.data['HAS_WISE_PHOT']!='T')
        w1 *= 10**(0.4*(0.184*extCorr[:,2]/2.285))
        w2 = np.ma.array(self.data['W2_NANOMAGGIES'],
                         mask=self.data['HAS_WISE_PHOT']!='T')
        w2 *= 10**(0.4*(0.113*extCorr[:,2]/2.285))
        self.wiseVegaFluxes = np.ma.vstack([w1,w2]).transpose()
        w1 *= 10**(-0.4*2.699)
        w2 *= 10**(-0.4*3.339)
        self.wiseFluxes = np.ma.vstack([w1,w2]).T
    def get_optwise(self):
        f_WISE = np.average(self.wiseVegaFluxes,axis=1,weights=[1,0.5])
        f_opt = np.average(self.sdssFluxes[:,1:4],axis=1,weights=[1,0.8,0.6])
        return f_opt,f_WISE
    def extract_features(self,featureset=['sdss','z'],
                         refband='i',ratios='byref'):
        fluxes = []
        names = []
        if 'galex' in featureset:
            fluxes.append(self.galexFluxes)
            names.extend(['f','n'])
        if 'sdss' in featureset:
            fluxes.append(self.sdssFluxes)
            names.extend(list('ugriz'))
        if 'ukidss' in featureset:
            fluxes.append(self.ukidssFluxes)
            names.extend(list('YJHK'))
        if 'wise' in featureset:
            fluxes.append(self.wiseFluxes)
            names.extend(['W1','W2'])
        #
        fluxes = np.ma.hstack(fluxes)
        j = names.index(refband)
        if ratios=='byref':
            refFlux,features = get_column_ratio(fluxes,j)
            names = [ b+'/'+refband for b in names 
                                      if b is not refband ]
        elif ratios=='neighboring':
            refFlux = fluxes[:,[j]]
            features = fluxes[:,:-1]/fluxes[:,1:]
            names = [ b1+'/'+b2 for b1,b2 in zip(names[:-1],names[1:]) ]
        else:
            raise ValueError
        if 'z' in featureset:
            z = np.ma.array(self.specz)[:,None]
            z = np.ma.log10(1+z)
            features = np.ma.hstack([z,features])
            names = ['logz_q'] + names
        return features,names,refFlux

def prep_simqsos(simqsos,refband='i',sdssonly=False):
    j = 3 # XXX
    ii = np.where(simqsos['selected'])[0]
    fluxes = np.array(simqsos['obsFlux'][ii])
    if sdssonly:
        fluxes = fluxes[:,:5]
    print('WARNING: HACK to get around bad forest transmission vals')
    print(fluxes.max())
    jj = np.where(np.all(fluxes < 1e5,axis=1))[0]
    ii = ii[jj]
    fluxes = fluxes[jj]
    print(fluxes.max())
    refFlux,fratios = get_column_ratio(fluxes,j)
    logz = np.log10(1 + simqsos['z'][ii,None])
    X = np.ma.hstack([logz,fratios])
    return X,refFlux

def fit_mixtures(X,mag,mbins,binwidth=0.2,seed=None,
                 keepscore=False,keepbic=False,**kwargs):
    kwargs.setdefault('n_components',25)
    kwargs.setdefault('covariance_type','full')
    fits = []
    if keepscore:
        scores = []
    if keepbic:
        bics = []
    if seed:
        np.random.seed(seed)
    for bincenter in mbins:
        # this is not an efficient way to assign bins, but the time
        # is negligible compared to the GMM fitting anyway
        ii = np.where( np.abs(mag-bincenter) < binwidth )[0]
        if False:
            print('{:.2f}: {} qsos'.format(bincenter,len(ii)))
        gmm = GaussianMixture(**kwargs)
        gmm.fit(X[ii])
        fits.append(gmm)
        if keepscore:
            scores.append(gmm.score(X[ii]))
        if keepbic:
            bics.append(gmm.bic(X[ii]))
    rv = (fits,)
    if keepscore:
        rv += (scores,)
    if keepbic:
        rv += (bics,)
    return rv

def simqso_density_estimation(simqsos,refband='i',mbins=None,seed=None,
                              sdssonly=False):
    if mbins is None:
        mbins = np.arange(17.7,22.51,0.1)
    X,refFlux = prep_simqsos(simqsos,refband,sdssonly=sdssonly)
    mag = 22.5 - 2.5*np.log10(refFlux.clip(1e-10,np.inf))
    fits, = fit_mixtures(X,mag,mbins,seed=seed)
    return fits

def model_selection(simqsos,refband='i',mbin=20.):
    X,refFlux = prep_simqsos(simqsos,refband)
    mag = 22.5 - 2.5*np.log10(refFlux.clip(1e-10,np.inf))
    cv_types = ['full'] #['spherical', 'tied', 'diag', 'full']
    n_components = np.arange(5,51,5)
    bics = []
    for cv_type in cv_types:
        for ncomp in n_components:
            fits,bic = fit_mixtures(X,mag,np.array([mbin]),
                                    keepbic=True,
                                    n_components=ncomp,
                                    covariance_type=cv_type)
            bics.append(bic[0])
            print(cv_type,ncomp,bic[0])
    return np.array(bics)

def plot_model_selection(simqsos):
    import matplotlib.pyplot as plt
    n_components = np.arange(5,51,5)
    plt.figure()
    for mbin in [19.,20.,21.]:
        bics = model_selection(simqsos,mbin=mbin)
        plt.plot(n_components,bics-bics.min(),label=str(mbin))
    plt.legend()

def fit_ebossqsos(simqsos,qsos=None,seed=None,navg=1,
                  sdssonly=False,verbose=0):
    if isinstance(simqsos,str):
        simqsos = Table.read(simqsos)
    if qsos is None:
        qsos = eBossQsos()
    if sdssonly:
        featureset = ['sdss','z']
    else:
        featureset = ['sdss','ukidss','wise','z']
    features,names,refFlux = qsos.extract_features(featureset=featureset)
    mags = 22.5 - 2.5*np.log10(refFlux.clip(1e-10,np.inf))
    mbins = np.arange(17.7,22.51,0.1)
    binNums = np.digitize(mags,mbins-0.1/2)
    print('fitting features ',','.join(names))
    print('in {} magnitude bins'.format(len(mbins)))
    if verbose:
        score_zedgs = np.linspace(np.log10(1.9),np.log10(5.0),10)
        score_zvals = np.zeros((len(score_zedgs),2))
    allscore = []
    for iterNum in range(navg):
        fits = simqso_density_estimation(simqsos,seed=seed,sdssonly=sdssonly)
        n,score = 0,0
        for i,fit in enumerate(fits):
            ii = np.where(binNums==i)[0]
            if len(ii) > 0:
                s = fit.score_samples(features[ii])
                score += s.sum()
                n += len(ii)
                if verbose:
                    zi = np.digitize(features[ii,0],score_zedgs) - 1
                    assert np.all( (zi>=0) & (zi<len(score_zedgs)) )
                    np.add.at( score_zvals, (zi,0), s )
                    np.add.at( score_zvals, (zi,1), 1 )
        score /= n
        print('  iter {} score {:.3f}'.format(iterNum+1,score))
        allscore.append(score)
        # don't reuse the seed!
        seed = None
    if verbose:
        avzscore = score_zvals[:,0] / score_zvals[:,1]
        print('#{:3} {:5} {:5}'.format('z','score','n'))
        score_zbins = 10**score_zedgs-1
        score_zbins = score_zbins[:-1] + np.diff(score_zbins)/2
        for i,z in enumerate(score_zbins):
            print('{:4.2f} {:5.2f} {:5d}'.format(z,
                                         avzscore[i],int(score_zvals[i,1])))
    if navg > 1:
        allscore = np.array(allscore)
        print('final avg {:.3f} with rms {:.3f}'.format(
               allscore.mean(),allscore.std()))

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser(
                              description='fit simulated quasars to eboss')
    parser.add_argument('fitsfile',nargs='+',type=str,
        help='simulation output file name(s)')
    parser.add_argument('-n','--navg',type=int,default=1,
        help='number of GMM fits to average')
    parser.add_argument('-s','--seed',type=int,default=12345,
        help='random seed')
    parser.add_argument('--sdss',action='store_true',
        help='only use SDSS fluxes in fit')
    parser.add_argument('-v','--verbose',action='count',
        help='increase output verbosity')
    args = parser.parse_args()
    #make_coreqso_table(sys.argv[1],sys.argv[2])
    coreqsos = eBossQsos()
    for ff in args.fitsfile:
        print(ff)
        fit_ebossqsos(ff,coreqsos,args.seed,args.navg,
                      sdssonly=args.sdss,verbose=args.verbose)
        print()

