#!/usr/bin/env python

import os
import numpy as np
from scipy.interpolate import RectBivariateSpline

from .sqrun import readSimulationData

def getGridBins(simPars):
	mEdges = np.arange(*simPars['GridParams']['mRange'])
	zEdges = np.arange(*simPars['GridParams']['zRange'])
	mBins = mEdges[:-1] + np.diff(mEdges)/2
	zBins = zEdges[:-1] + np.diff(zEdges)/2
	gridShape = mBins.shape + zBins.shape + (simPars['GridParams']['nPerBin'],)
	return mBins,zBins,gridShape

def calcKCorrFromGrid(fileName,outputDir='./',retGrid=False,bandNum=0):
	from astropy import cosmology
	simData,simPars = readSimulationData(fileName,outputDir,retParams=True)
	mBins,zBins,gridShape = getGridBins(simPars)
	# XXX have to map name to object...
	simPars['Cosmology'] = {
	  'WMAP9':cosmology.WMAP9,
	}[simPars['Cosmology']]
	DM_z = simPars['Cosmology'].distmod(zBins).value
	# XXX should be loading sim data reshaped already
	appMag = simData['synMag'][...,bandNum].reshape(gridShape)
	absMag = simData['M'].reshape(gridShape)
	# XXX should just save the DMs that were used?
	kCorrGrid = (appMag - DM_z[np.newaxis,:,np.newaxis]) - absMag
	kCorrGrid = np.median(kCorrGrid,axis=-1)
	if retGrid:
		return kCorrGrid
	else:
		return RectBivariateSpline(mBins,zBins,kCorrGrid,kx=3,ky=3,s=1)

class ClippedFunction(object):
	def __init__(self,fun,minval=0.0,maxval=1.0):
		self.fun = fun
		self.minval = minval
		self.maxval = maxval
	def clip(self,arr):
		return arr.clip(self.minval,self.maxval)
	def __call__(self,*args,**kwargs):
		return self.clip(self.fun(*args,**kwargs))
	def ev(self,*args,**kwargs):
		return self.clip(self.fun.ev(*args,**kwargs))

def calcSelectionFunctionFromGrid(fileName,selector,outputDir='./'):
	simData,simPars = readSimulationData(fileName,outputDir,retParams=True)
	mBins,zBins,gridShape = getGridBins(simPars)
	is_selected = selector(simData['obsMag'],simData['obsMagErr'],
	                       simData['obsFlux'],simData['obsFluxErr'])
	is_selected = is_selected.reshape(gridShape)
	selGrid = np.sum(is_selected,axis=-1).astype(np.float32) / gridShape[-1]
	selFun = RectBivariateSpline(mBins,zBins,selGrid,kx=3,ky=3,s=1)
	return ClippedFunction(selFun)

