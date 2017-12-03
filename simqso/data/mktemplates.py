#!/usr/bin/env python

import os
import numpy as np
from astropy.io import fits
from astropy.table import Table

def VestergaardWilkes01_Fe(fepath=None):
    if fepath is None:
        fepath = 'VW01_Fe'
    rv = []
    for fn in ['Fe_UVtemplt_B.asc','Fe2_UV191.asc','Fe3_UV47.asc',
               'Fe_UVOPT_V01_T06_BR92.asc']:
        wave,flux = np.loadtxt(os.path.join(fepath,fn),unpack=True)
        # continuum parameters given in VW01 pg. 6
        a_nu = -1.9
        fcont1 = 3.45e-14 * (wave/1500.)**(-2-a_nu)
        a_nu = -1.0
        fcont2 = 3.89e-14 * (wave/1500.)**(-2-a_nu)
        fcont = np.choose(wave>1716,[fcont1,fcont2])
        flux /= fcont
        tmplname = fn.replace('.asc','')
        if not 'UVOPT' in tmplname:
            tmplname = 'VW01_'+tmplname
        tab = Table(dict(wave=wave,f_lambda=flux))
        hdu = fits.table_to_hdu(tab)
        hdu.name = tmplname
        rv.append(hdu)
    return rv

def VandenBerkSDSSCompositeTemplate():
    all_lines = Table.read('VandenBerk2001_AJ122_549_table2.txt',
                           format='ascii')
    hdu = fits.table_to_hdu(all_lines)
    hdu.name = 'VdB01CompEmLines'
    return hdu

def append_one(hdu):
    hdu_list = fits.open('simqso_templates.fits',mode='update')
    hdu_list.append(hdu)
    hdu_list.close()
#    hdu_list.writeto('simqso_templates.fits',overwrite=True)

def from_scratch():
    hdu_list = fits.HDUList([fits.PrimaryHDU()])
    hdu_list.extend( VestergaardWilkes01_Fe() )
    hdu_list.extend( VandenBerkSDSSCompositeTemplate() )
    return hdu_list

if __name__=='__main__':
    hdu_list = from_scratch()
    hdu_list.writeto('simqso_templates.fits',overwrite=True)

