#!/usr/bin/env python

import os
from astropy.io import fits
from astropy.table import Table

def add_speclite_filters(speclitepath):
    slfiltdir = os.path.join(speclitepath,'speclite','data','filters')
    filtDat = fits.open('filtercurves.fits',mode='update')
    toadd = [ ('DECam','decam2014','ugrizY'),
              ('HSC','hsc2017','grizy'),
              ('LSST','lsst2016','ugrizy') ]
    for name,pfx,bands in toadd:
        for b in bands:
            fdat = Table.read(os.path.join(slfiltdir,pfx+'-'+b+'.ecsv'),
                              format='ascii.ecsv')
            wave = fdat['wavelength'].to('Angstrom').value
            ftab = Table(dict(lam=wave,Rlam=fdat['response']))
            filtDat.append(fits.BinTableHDU(ftab,name=name+'-'+b))
    filtDat.close()

