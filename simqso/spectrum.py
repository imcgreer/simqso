#!/usr/bin/env python

import numpy as np

def fixed_R_dispersion(lam1,lam2,R):
	'''Generate a wavelength grid with fixed resolution (i.e., logarithmic)'''
	loglam1 = np.log(lam1)
	loglam2 = np.log(lam2)
	loglam = np.arange(loglam1,loglam2,1./R)
	return np.exp(loglam)

