#!/usr/bin/env python

import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline

SMCDustTable = np.array( [
  [ 1275.0, 13.54 ],
  [ 1330.0, 12.52 ],
  [ 1385.0, 11.51 ],
  [ 1435.0, 10.80 ],
  [ 1490.0, 9.84 ],
  [ 1545.0, 9.28 ],
  [ 1595.0, 9.06 ],
  [ 1647.0, 8.49 ],
  [ 1700.0, 8.01 ],
  [ 1755.0, 7.71 ],
  [ 1810.0, 7.17 ],
  [ 1860.0, 6.90 ],
  [ 1910.0, 6.76 ],
  [ 2000.0, 6.38 ],
  [ 2115.0, 5.85 ],
  [ 2220.0, 5.30 ],
  [ 2335.0, 4.53 ],
  [ 2445.0, 4.24 ],
  [ 2550.0, 3.91 ],
  [ 2665.0, 3.49 ],
  [ 2778.0, 3.15 ],
  [ 2890.0, 3.00 ],
  [ 2995.0, 2.65 ],
  [ 3105.0, 2.29 ],
  [ 3704.0, 1.81 ],
  [ 4255.0, 1.00 ],
  [ 5291.0, 0.00 ],
  [ 12500.0, -2.02 ],
  [ 16500.0, -2.36 ],
  [ 22000.0, -2.47 ] ] )

# provides for a reasonable extrapolation to shorter wavelengths
SMCdust_ev = InterpolatedUnivariateSpline(SMCDustTable[:,0],
                                          SMCDustTable[:,1],k=1)

SMC_R_V = 2.93

def SMCdust_Alam(w,E_BmV): 
    A_V = E_BmV / SMC_R_V
    return A_V * (SMCdust_ev(w)/SMC_R_V + 1.0)

Calzetti_R_V = 4.05

klam_blue = lambda lam: 2.659*(-2.156 + 
                                1.509*lam**-1 - 
                                0.198*lam**-2 +
                                0.011*lam**-3) + Calzetti_R_V

klam_red = lambda lam: 2.659*(-1.857 + 1.040*lam**-1) + Calzetti_R_V

def Calzetti_klam(lam):
    lam_um = 1e-4*np.asarray(lam)
    k_lam = np.empty_like(lam)
    # blue side
    i1 = np.searchsorted(lam_um,0.12)
    i2 = np.searchsorted(lam_um,0.63,side='right')
    if i2-i1 > 0:
        k_lam[i1:i2] = klam_blue(lam_um[i1:i2])
    # extrapolate blueward
    if i1 > 0:
        # linear extrapolation a la hyperz...
        lamb = np.array([0.11,0.12])
        klamb = klam_blue(lamb)
        slopeb = np.diff(klamb)/np.diff(lamb)
        k_lam[:i1] = klamb[0] + slopeb*(lam_um[:i1]-lamb[0])
    # red side
    i1 = np.searchsorted(lam_um,0.63)
    i2 = np.searchsorted(lam_um,2.20,side='right')
    if i2-i1 > 0:
        k_lam[i1:i2] = klam_red(lam_um[i1:i2])
    # extrapolate redward
    if i2 < len(lam)-1:
        lamr = np.array([2.19,2.20])
        klamr = klam_red(lamr)
        sloper = np.diff(klamr)/np.diff(lamr)
        k_lam[i2:] = klamr[0] + sloper*(lam_um[i2:]-lamr[0])
    return k_lam

def Calzetti_Alam(lam,E_BmV):
    #A_V = E_BmV / Calzetti_R_V
    #return Calzetti_klam(lam)*A_V/Calzetti_R_V
    return Calzetti_klam(lam)*E_BmV

#def SMCdust_fn(rfwave,flux,E_BmV):
#    return flux * 10**(-0.4*SMCdust_Alam(rfwave,E_BmV))

def SMCdust_fn(rfwave,flux,E_BmV):
    return flux * 10**(-0.4*(1.39*E_BmV*(rfwave/1e4)**-1.2))

def CalzettiDust_fn(rfwave,flux,E_BmV):
    return flux * 10**(-0.4*Calzetti_Alam(rfwave,E_BmV))

dust_fn = {'SMC':SMCdust_fn,'CalzettiSB':CalzettiDust_fn}

