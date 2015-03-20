#!/usr/bin/env python

import os
import numpy as np
from scipy.interpolate import RectBivariateSpline

from .sqrun import readSimulationData

class Interp2DSeries:
	def __init__(self,interpFun):
		self.interpFun = interpFun
	def __call__(self,x,y):
		try:
			# inputs are sequences
			return np.array([self.interpFun(x,y)[0][0]
			                   for x,y in zip(x,y)])
		except TypeError:
			# inputs are scalars
			return self.interpFun(x,y).squeeze()

def calcKCorrFromFluxGrid(fileName,outputDir='./',retGridFun=False,bandNum=0):
	simData,simPars = readSimulationData(fileName,outputDir,retParams=True)
	mEdges = np.arange(*simPars['GridParams']['mRange'])
	zEdges = np.arange(*simPars['GridParams']['zRange'])
	gridShape = (len(mEdges)-1,len(zEdges)-1,simPars['GridParams']['nPerBin'])
	mBins = mEdges[:-1] + np.diff(mEdges)/2
	zBins = zEdges[:-1] + np.diff(zEdges)/2
	DM_z = simPars['Cosmology'].distmod(zBins).value
	appMag = simData['synMag'][...,bandNum].reshape(gridShape)
	absMag = simData['M'].reshape(gridShape)
	# XXX should just save the DMs that were used?
	kCorrGrid = (appMag - DM_z[np.newaxis,:,np.newaxis]) - absMag
	kCorrGrid = np.mean(kCorrGrid,axis=-1)
	kCorr = RectBivariateSpline(mBins,zBins,kCorrGrid,kx=3,ky=3,s=1)
	if retGridFun:
		return kCorr
	else:
		return Interp2DSeries(kCorr)

