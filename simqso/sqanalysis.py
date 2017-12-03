#!/usr/bin/env python

import os
import numpy as np
from scipy.interpolate import RectBivariateSpline,SmoothBivariateSpline, \
                              griddata
from scipy.stats import binned_statistic_2d

from . import sqgrids as grids
from .sqrun import readSimulationData

class ClippedFunction(object):
    def __init__(self,fun,minval,maxval):
        self.fun = fun
        self.minval = minval
        self.maxval = maxval
    def clip(self,arr):
        return arr.clip(self.minval,self.maxval)
    def __call__(self,*args,**kwargs):
        return self.clip(self.fun(*args,**kwargs))
    def ev(self,*args,**kwargs):
        return self.clip(self.fun.ev(*args,**kwargs))

class Interpolator(object):
    def __init__(self,x,y,z):
        self.points = np.array([x,y]).T
        self.values = z
    def ev(self,x,y):
        x = np.asarray(x)
        y = np.asarray(y)
        xi = np.array([x.ravel(),y.ravel()]).T
        rv = griddata(self.points,self.values,xi,method='linear')
        return rv.reshape(x.shape)
    def __call__(self,x,y):
        xx,yy = np.meshgrid(x,y,indexing='ij')
        return self.ev(xx.ravel(),yy.ravel()).reshape(len(x),len(y))

class SimGridOutput(object):
    def __init__(self,qsoGrid,minNobj=10,clip=None):
        self.qsoGrid = qsoGrid
        self.minNobj = minNobj
        self.clip = clip
        self.splineKwargs = dict(kx=3,ky=3,s=0)
        #if not isinstance(qsoGrid,grids.QsoSimGrid):
        #    raise NotImplementedError
        if len(qsoGrid.gridShape) > 3:
            raise NotImplementedError(">2D grids not supported yet")
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
        self.absMag = self.qsoGrid.asGrid('absMag')
    def _from_grid(self):
        g = self._collapse_grid()
        f = RectBivariateSpline(self.mBins,self.zBins,g,**self.splineKwargs)
        if self.clip:
            f = ClippedFunction(f,*self.clip)
        return dict(mBins=self.mBins,zBins=self.zBins,grid=g,interp=f)
    def _resample_grid(self,mag,vals):
        mag = mag.ravel()
        vals = vals.ravel()
        #
        mStep = np.median(np.diff(self.mBins)) / 2
        mLow = mag.min() - mStep
        mHigh = mag.max() + mStep
        mEdges = np.arange(mLow,mHigh,mStep)
        #
        zEdges = self.zEdges
        z = np.array(self.qsoGrid.z)
        #
        g = self._grid_binstats(mag,z,vals,[mEdges,zEdges])
        mBins = mEdges[:-1] + np.diff(mEdges)/2
        zBins = zEdges[:-1] + np.diff(zEdges)/2
        mm,zz = np.meshgrid(mBins,zBins,indexing='ij')
        ii = np.where(np.isfinite(g))
        #f = SmoothBivariateSpline(mm[ii],zz[ii],g[ii],**self.splineKwargs)
        f = Interpolator(mm[ii],zz[ii],g[ii])
        if self.clip:
            f = ClippedFunction(f,*self.clip)
        return dict(mBins=mBins,zBins=zBins,grid=g,interp=f)
    def calc_grid(self,*args,**kwargs):
        self._calc_vals(*args,**kwargs)
        #
        if self.qsoGrid.units == 'luminosity':
            self.f_M = self._from_grid()
            self.f_m = self._resample_grid(appMag,self.gridVals)
        else:
            self.f_m = self._from_grid()
            self.f_M = self._resample_grid(self.absMag,self.gridVals)
    def __call__(self,m,z,inverse=False):
        if inverse:
            return self.f_M['interp'].ev(m,z)
        else:
            return self.f_m['interp'].ev(m,z)

class SimKCorr(SimGridOutput):
    fillValue = np.nan
    def __init__(self,qsoGrid,**kwargs):
        super(SimKCorr,self).__init__(qsoGrid,**kwargs)
        self.DM = self.qsoGrid.distMod(self.qsoGrid.asGrid('z'))
        self.appMag = self.qsoGrid.asGrid('synMag')
    def _calc_vals(self,band):
        # extract the apparent mag array for the input band
        b_j = self.qsoGrid.getBandIndex(band)
        appMag = self.appMag[...,b_j]
        # calculate k-correction by subtracting off the distance modulus
        self.gridVals = (appMag - self.DM) - self.absMag
    def _collapse_grid(self):
        return np.median(self.gridVals,axis=-1)
    def _grid_binstats(self,mag,z,vals,bins):
        g = binned_statistic_2d(mag,z,vals,'median',bins)[0]
        n = binned_statistic_2d(mag,z,vals,'count',bins)[0]
        g[n<self.minNobj] = np.nan
        return g

class SelectionFunction(SimGridOutput):
    fillValue = 0.0
    def __init__(self,qsoGrid,m2M,**kwargs):
        kwargs.setdefault('clip',(0,1))
        super(SelectionFunction,self).__init__(qsoGrid,**kwargs)
        self.m2M = m2M
        self.nPerBin = self.qsoGrid.gridShape[-1]
    def _calc_vals(self):
        s = self.selector(self.qsoGrid.obsMag,self.qsoGrid.obsMagErr,
                          self.qsoGrid.obsFlux,self.qsoGrid.obsFluxErr,
                          self.qsoGrid.photoBands)
        self.gridVals = s.reshape(self.qsoGrid.gridShape)
    def _collapse_grid(self):
        return np.sum(self.gridVals,axis=-1) / float(self.nPerBin)
    def _grid_binstats(self,mag,z,vals,bins):
        g = binned_statistic_2d(mag,z,vals,'sum',bins)[0]
        n = binned_statistic_2d(mag,z,vals,'count',bins)[0]
        g[n<self.minNobj] = 0.0
        g[n>=self.minNobj] /= n[n>=self.minNobj]
        return g
    def _photo_complete_call(self,m,z,absMag=False):
        if type(self.photo_complete) is float:
            return self.photo_complete
        elif absMag:
            return self.photo_complete(m+self.m2M(m,z,inverse=True),z)
        else:
            return self.photo_complete(m,z)
    def _spec_complete_call(self,m,z,absMag=False):
        if type(self.spec_complete) is float:
            return self.spec_complete
        elif absMag:
            return self.spec_complete(m+self.m2M(m,z,inverse=True),z)
        else:
            return self.spec_complete(m,z)
    def _color_complete_call(self,m,z,absMag=False):
        return super(SelectionFunction,self).__call__(m,z,absMag)
    def __call__(self,m,z,absMag=False):
        return self._color_complete_call(m,z,absMag) * \
               self._photo_complete_call(m,z,absMag) * \
               self._spec_complete_call(m,z,absMag)

