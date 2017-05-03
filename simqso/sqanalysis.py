#!/usr/bin/env python

import os
import numpy as np
from scipy.interpolate import RectBivariateSpline,SmoothBivariateSpline
from scipy.stats import binned_statistic_2d

from . import sqgrids as grids
from .sqrun import readSimulationData

class SimKCorr(object):
	def __init__(self,qsoGrid,minNobj=10):
		self.qsoGrid = qsoGrid
		self.minNobj = minNobj
		#if not isinstance(qsoGrid,grids.QsoSimGrid):
		#	raise NotImplementedError
		if len(qsoGrid.gridShape) > 3:
			raise NotImplementedError(">2D k-corrections not supported yet")
		if qsoGrid.units == 'flux':
			self.mBins = qsoGrid.gridCenters[qsoGrid.varIndex('appMag')]
			self.gridBand = qsoGrid.getVars('appMag').obsBand
		else:
			self.mBins = qsoGrid.gridCenters[qsoGrid.varIndex('absMag')]
			self.gridBand = qsoGrid.getVars('absMag').restWave
		#
		self.zDimIndex = qsoGrid.varIndex('z')
		self.zBins = qsoGrid.gridCenters[self.zDimIndex]
		jj = [ np.s_[0] for i in range(qsoGrid.nGridDim) ]
		jj[self.zDimIndex] = np.s_[:]
		self.zEdges = qsoGrid.gridEdges[self.zDimIndex][jj]
		#
		self.DM = qsoGrid.distMod(qsoGrid.asGrid('z'))
		self.appMag = qsoGrid.asGrid('synMag')
		self.absMag = qsoGrid.asGrid('absMag')
	def calc_grid(self,band,withInverse=True,invBand=None):
		# extract the apparent mag array for the input band
		if band != self.gridBand:
			raise NotImplementedError
		try:
			b_j = self.qsoGrid.getBandIndex(band)
		except StopIteration:
			try:
				b_j = self.qsoGrid.getBandIndex(invBand)
			except:
				raise ValueError("Band provided (%s) not available" % band)
		appMag = self.appMag[...,b_j]
		# calculate k-correction by subtracting off the distance modulus
		self.kCorrVals = (appMag - self.DM) - self.absMag
		self.kCorrGrid = np.median(self.kCorrVals,axis=-1)
		self.kCorrFun = RectBivariateSpline(self.mBins,self.zBins,
		                                    self.kCorrGrid,
		                                    kx=3,ky=3,s=1)
		if withInverse:
			if self.qsoGrid.units == 'luminosity':
				band = invBand
			else:
				band = 'absMag'
			self.inv_mBins,self.inv_zBins,self.inv_kCorrGrid = \
			                                    self._resample_grid(band)
			mm,zz = np.meshgrid(self.inv_mBins,self.inv_zBins,indexing='ij')
			ii = np.where(np.isfinite(self.inv_kCorrGrid))
			self.inv_kCorrFun = SmoothBivariateSpline(mm[ii],zz[ii],
			                                          self.inv_kCorrGrid[ii],
			                                          kx=3,ky=3,s=1)
	def _resample_grid(self,band):
		mStep = np.median(np.diff(self.mBins))
		if band=='absMag':
			mag = self.absMag.ravel()
		else:
			b_j = self.qsoGrid.getBandIndex(band)
			mag = self.appMag[...,b_j].ravel()
		mLow = mag.min() - mStep
		mHigh = mag.max() + mStep
		mEdges = np.arange(mLow,mHigh,mStep)
		#
		zEdges = self.zEdges
		z = np.array(self.qsoGrid.z)
		#
		kv = self.kCorrVals.ravel()
		kCorrGrid,_,_,ii = binned_statistic_2d(mag,z,kv,'median',
		                                       [mEdges,zEdges])
		n,_,_,_ = binned_statistic_2d(mag,z,kv,'count',[mEdges,zEdges])
		kCorrGrid[n<self.minNobj] = np.nan
		mBins = mEdges[:-1] + np.diff(mEdges)/2
		zBins = zEdges[:-1] + np.diff(zEdges)/2
		return mBins,zBins,kCorrGrid
	def __call__(self,m,z,inverse=False):
		if ( (inverse and self.qsoGrid.units=='luminosity') or 
		     (not inverse and self.qsoGrid.units=='flux') ):
			return self.kCorrFun.ev(m,z)
		else:
			return self.inv_kCorrFun.ev(m,z)

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

