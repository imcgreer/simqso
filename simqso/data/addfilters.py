#!/usr/bin/env python

import os
from astropy.io import fits
from astropy.table import Table

def add_filters(speclitepath, toadd=None):
    if toadd is None:
        return
    slfiltdir = os.path.join(speclitepath,'speclite','data','filters')
    filtDat = fits.open('filtercurves.fits',mode='update')
    for name,pfx,bands in toadd:
        for b in bands:
            fdat = Table.read(os.path.join(slfiltdir,pfx+'-'+b+'.ecsv'),
                              format='ascii.ecsv')
            wave = fdat['wavelength'].to('Angstrom').value
            ftab = Table(dict(lam=wave,Rlam=fdat['response']))
            filtDat.append(fits.BinTableHDU(ftab,name=name+'-'+b))
    filtDat.close()

def add_speclite_filters(speclitepath):
    toadd = [ ('DECam','decam2014','ugrizY'),
              ('HSC','hsc2017','grizy'),
              ('LSST','lsst2016','ugrizy') ]
    add_filters(speclitepath, toadd)

def add_mzls_bass_filters(speclitepath):
    toadd = [ ('BASS-MzLS','BASS','gr'), ('BASS-MzLS','MzLS','z') ]
    add_filters(speclitepath, toadd)
