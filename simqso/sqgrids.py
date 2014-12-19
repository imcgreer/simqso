#!/usr/bin/env python

import numpy as np
from astropy import cosmology
from astropy.table import Table
from astropy.io.fits import Header
from astropy.io import ascii as ascii_io

from .sqbase import datadir
from . import dustextinction

class MzGrid(object):
	'''
	A grid of points in (M,z) space.
	The space is divided into cells with n points per cell, such that
	the final grid dimensions are (M,z,nPerBin).
	The grid can be iterated to obtain (M,z) pairs spanning the full grid. 
	Individual cells can be accessed with grid.bin(i,j).
	'''
	def __iter__(self):
		itM = np.nditer(self.Mgrid,flags=['multi_index'])
		itz = np.nditer(self.zgrid)
		while not itM.finished:
			yield itM[0],itz[0],itM.multi_index
			itM.iternext()
			itz.iternext()
	def getRedshifts(self,sorted=False,return_index=False):
		zv = self.zgrid.flatten()
		if sorted:
			zi = zv.argsort()
			if return_index:
				return zv[zi],zi
			else:
				return zv[zi]
		else:
			return zv
	def bin(self,i,j):
		return self.Mgrid[i,j,:],self.zgrid[i,j,:]
	def get(self,i):
		if type(i) is tuple:
			idx = i
		else:
			idx = np.unravel_index(i,self.Mgrid.shape)
		return self.Mgrid[idx],self.zgrid[idx]
	def numQSO(self):
		return self.Mgrid.size
	def get_Medges(self):
		return self.Medges
	def get_zbincenters(self):
		return self.zbincenters
	def setCosmology(self,cosmodef):
		if type(cosmodef) is dict:
			self.cosmo = cosmology.FlatLambdaCDM(**cosmodef)
		elif type(cosmodef) is str:
			# XXX are they indexed by name in astropy?
			self.cosmo = {'WMAP9':cosmology.WMAP9}[cosmodef]
		elif isinstance(cosmodef,cosmology.FLRW):
			self.cosmo = cosmodef
		elif cosmodef is None:
			self.cosmo = cosmology.get_current()
		else:
			raiseValueError
	def distMod(self,z):
		return self.cosmo.distmod(z).value

class MzGridFromData(MzGrid):
	def __init__(self,mzdata,gridpar):
		if len(gridpar['mRange'])==3:
			self.Medges = np.arange(*gridpar['mRange'])
			self.zedges = np.arange(*gridpar['zRange'])
			self.nPerBin = gridpar['nPerBin']
			self.nM = self.Medges.shape[0] - 1
			self.nz = self.zedges.shape[0] - 1
			self.zbincenters = (self.zedges[:-1]+self.zedges[1:])/2
			gridshape = (self.nM,self.nz,self.nPerBin)
			self.Mgrid = mzdata['M'].copy().reshape(gridshape)
			self.zgrid = mzdata['z'].copy().reshape(gridshape)
		else:
			self.Mgrid = mzdata['M'].copy()
			self.zgrid = mzdata['z'].copy()
			self.nM = 1
			self.nz = 1
			self.Medges = np.array(gridpar['mRange'])
			self.zedges = np.array(gridpar['zRange'])
			self.nPerBin = self.Mgrid.size
			self.zbincenters = np.mean(self.zedges)
	def get_zrange(self):
		return self.zedges[0],self.zedges[-1]

class LuminosityRedshiftGrid(MzGrid):
	def __init__(self,Medges,zedges,nPerBin,lumUnits='M1450'):
		if lumUnits != 'M1450':
			raise NotImplementedError('only M1450 supported for now')
		self.Medges = Medges
		self.zedges = zedges
		self.nPerBin = nPerBin
		self.nM = Medges.shape[0] - 1
		self.nz = zedges.shape[0] - 1
		self.Mgrid = np.zeros((self.nM,self.nz,nPerBin))
		self.zgrid = np.zeros((self.nM,self.nz,nPerBin))
		dM = np.diff(Medges)
		dz = np.diff(zedges)
		for i in range(self.nM):
			for j in range(self.nz):
				binM = Medges[i] + dM[i]*np.random.rand(nPerBin)
				binz = zedges[j] + dz[j]*np.random.rand(nPerBin)
				zi = binz.argsort()
				self.Mgrid[i,j,:] = binM[zi]
				self.zgrid[i,j,:] = binz[zi]
	def getLuminosities(self,units='ergs/s/Hz'):
		# convert M1450 -> ergs/s/Hz
		pass

class FluxRedshiftGrid(MzGrid):
	'''
	Construct a grid in (mag,z) having a fixed number of points within each 
	bin. The bin spacings need not be uniform, as long as they are 
	monotonically increasing.
	'''
	def __init__(self,medges,zedges,nPerBin,obsBand='i',restBand='1450'):
		self.medges = medges
		self.zedges = zedges
		self.nPerBin = nPerBin
		self.nM = medges.shape[0] - 1
		self.nz = zedges.shape[0] - 1
		self.mgrid = np.zeros((self.nM,self.nz,nPerBin))
		self.zgrid = np.zeros((self.nM,self.nz,nPerBin))
		self.zbincenters = (zedges[:-1]+zedges[1:])/2
		dm = np.diff(medges)
		dz = np.diff(zedges)
		m2M = sqbase.Mconverter(obsBand,restBand)
		# distribute quasars into bins of flux 
		for i in range(self.nM):
			for j in range(self.nz):
				binm = medges[i] + dm[i]*np.random.rand(nPerBin)
				binz = zedges[j] + dz[j]*np.random.rand(nPerBin)
				zi = binz.argsort()
				self.mgrid[i,j,:] = binm[zi]
				self.zgrid[i,j,:] = binz[zi]
		# convert to luminosity
		self.Mgrid = self.mgrid - m2M(self.zgrid)
		self.Medges = np.empty((self.nz,self.nM+1))
		for j in range(self.nz):
			self.Medges[j,:] = self.medges - m2M(self.zedges[j])
	def get_zrange(self):
		return self.zedges[0],self.zedges[-1]
	def resetAbsMag(self,Mgrid):
		self.Mgrid[:] = Mgrid

class FixedPLContinuumGrid(object):
	def __init__(self,M,z,slopes,breakpoints):
		self.slopes = np.asarray(slopes)
		self.breakpoints = np.asarray(breakpoints)
	def get(self,*args):
		return self.slopes,self.breakpoints
	def __iter__(self):
		while True:
			yield self.slopes,self.breakpoints

class GaussianPLContinuumGrid(object):
	def __init__(self,M,z,slopeMeans,slopeStds,breakpoints):
		self.slopeMeans = slopeMeans
		self.slopeStds = slopeStds
		self.breakpoints = np.concatenate([[0,],breakpoints])
		shape = z.shape+(len(slopeMeans),)
		x = np.random.randn(*shape)
		mu = np.asarray(slopeMeans)
		sig = np.asarray(slopeStds)
		self.slopes = mu + x*sig
	def get(self,idx):
		return self.slopes[idx],self.breakpoints
	def getTable(self,hdr):
		'''Return a Table of all parameters and header information'''
		flatgrid = np.product(self.slopes.shape[:-1])
		t = Table({'slopes':self.slopes.reshape(flatgrid,-1)})
		hdr.update('CNTBKPTS',','.join(['%.1f' % bkpt 
		                            for bkpt in self.breakpoints]))
		return t

class FixedVdBcompositeEMLineGrid(object):
	def __init__(self,M,z,minEW=1.0,noFe=False):
		self.minEW = minEW
		self.all_lines = ascii_io.read(datadir+
		                            'VandenBerk2001_AJ122_549_table2.txt')
		# blended lines are repeated in the table.  
		l,li = np.unique(self.all_lines['OWave'],return_index=True)
		self.unique_lines = self.all_lines[li]
		li = np.where(self.unique_lines['EqWid'] > minEW)[0]
		self.lines = self.unique_lines[li]
		if noFe:
			isFe = self.lines['ID'].find('Fe') == 0
			self.lines = self.lines[~isFe]
		print 'using the following lines from VdB template: ',self.lines['ID']
	def addLine(self,name,rfwave,eqWidth,profileWidth):
		self.lines = np.resize(self.lines,(self.lines.size+1,))
		self.lines = self.lines.view(np.recarray)
		self.lines.OWave[-1] = rfwave
		self.lines.EqWid[-1] = eqWidth
		self.lines.Width[-1] = profileWidth
		self.lines.ID[-1] = name
	def addSBB(self):
		# XXX this is a hack!
		self.addLine('SBBhack',3000.,300.,500.)
	def _idmap(self,name):
		if name.startswith('Ly'):
			return 'Ly'+{'A':'{alpha}','B':'{beta}',
			             'D':'{delta}','E':'{epsilon}'}[name[-1]]
		else:
			return name
	def set(self,name,**kwargs):
		li = np.where(self._idmap(name) == self.lines.ID)[0]
		if len(li) == 0:
			print self.lines.ID
			raise ValueError
		for k,v in kwargs.items():
			if k == 'width':
				self.lines['Width'][li] = v
			elif k == 'eqWidth':
				self.lines['EqWid'][li] = v
			elif k == 'wave':
				self.lines['OWave'][li] = v
	def __iter__(self):
		'''returns wave, equivalent width, gaussian sigma'''
		while True:
			yield self.get(None)
	def get(self,idx):
		return self.lines['OWave'],self.lines['EqWid'],self.lines['Width']
	def getTable(self,hdr):
		'''Return a Table of all parameters and header information'''
		hdr.update('LINEMODL','Fixed Vanden Berk et al. 2001 emission lines')
		return None

class FixedDustGrid(object):
	def __init__(self,M,z,dustModel,E_BmV):
		self.dustModel = dustModel
		self.E_BmV = E_BmV
		self.dust_fn = dustextinction.dust_fn[dustModel]
	def __iter__(self):
		while True:
			yield self.dust_fn,self.E_BmV
	def getTable(self,hdr):
		'''Return a Table of all parameters and header information'''
		hdr.update('DUSTMODL',self.dustModel)
		hdr.update('FIXEDEBV',self.E_BmV)
		return None

class ExponentialDustGrid(object):
	def __init__(self,M,z,dustModel,E_BmV_scale,fraction=1):
		self.dustModel = dustModel
		self.E_BmV_scale = E_BmV_scale
		self.dust_fn = dustextinction.dust_fn[dustModel]
		if fraction==1:
			self.EBVdist = np.random.exponential(E_BmV_scale,M.shape)
		else:
			print 'using dust LOS fraction ',fraction
			self.EBVdist = np.zeros_like(M).astype(np.float32)
			N = fraction * M.size
			ii = np.random.randint(0,M.size,(N,))
			self.EBVdist.flat[ii] = np.random.exponential(E_BmV_scale,(N,)).astype(np.float32)
	def __iter__(self):
		for ebv in self.EBVdist:
			yield self.dust_fn,ebv
	def get(self,idx):
		return self.dust_fn,self.EBVdist[idx]
	def getTable(self,hdr):
		'''Return a Table of all parameters and header information'''
		t = Table({'E(B-V)':self.EBVdist.flatten()})
		hdr.update('DUSTMODL',self.dustModel)
		hdr.update('EBVSCALE',self.E_BmV_scale)
		return t

