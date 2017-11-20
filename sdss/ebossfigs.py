#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

def binnedlf(ebossqsos,simqsos,qlf):
	nbins = 6
	zedges = np.linspace(0.9,4.0,nbins+1)
	zbins = zedges[:-1] + np.diff(zedges)/2
	medges = np.linspace(17,22.2,15)
	mbins = medges[:-1] + np.diff(medges)/2
	mm = np.linspace(16,23,100)
	flux = ebossqsos.sdssFluxes[:,2].filled(0).clip(1e-10,np.inf)
	e_mag = 22.5-2.5*np.log10(flux) # XXX
	s_mag = simqsos['obsMag'][:,2]
	e_zi = np.digitize(ebossqsos.specz,zedges)
	s_zi = np.digitize(simqsos['z'],zedges)
	plt.figure(figsize=(7.5,7))
	plt.subplots_adjust(0.08,0.08,0.97,0.95,0.23,0.44)
	for i in range(nbins):
		ax = plt.subplot(3,2,i+1)
		ii = np.where(e_zi==i+1)[0]
		n_e,_ = np.histogram(e_mag[ii],medges)
		ii = np.where((s_zi==i+1) & simqsos['selected'])[0]
		n_s,_ = np.histogram(s_mag[ii],medges)
		ii = np.where(s_zi==i+1)[0]
		n_t,_ = np.histogram(s_mag[ii],medges)
		#
		ii = np.where(n_t>0)[0]
		plt.plot(mbins[ii],n_t[ii],label='sim-int')
		ii = np.where(n_s>0)[0]
		plt.plot(mbins[ii],n_s[ii],drawstyle='steps-mid',label='sim-obs')
		ii = np.where(n_e>0)[0]
		plt.plot(mbins[ii],n_e[ii],drawstyle='steps-mid',label='DR14qso')
		if i==0:
			plt.legend()
		plt.yscale('log')
		plt.xlim(22.5,16.9)
		ax.set_title(r'$%.1f<z<%.1f$'%(zedges[i],zedges[i+1]))
		if i%2==0:
			ax.set_ylabel('N')
		if i>=4:
			ax.set_xlabel('r mag')

