#!/usr/bin/env python

import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import simps
from scipy.signal import convolve
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

class VW01FeTemplateGrid(object):
	def __init__(self,M,z,wave,fwhm=5000.,scales=None):
		self.zbins = np.arange(z.min(),z.max()+0.005,0.005)
		self.feGrid = np.empty((self.zbins.shape[0],wave.shape[0]))
		# the Fe template is an equivalent width spectrum
		wave0,ew0 = self._restFrameFeTemplate(fwhm,scales)
		for i,zbin in enumerate(self.zbins):
			# in units of EW - no (1+z) when redshifting
			rfEW = interp1d(wave0*(1+zbin),ew0,kind='slinear',
			                bounds_error=False,fill_value=0.0)
			self.feGrid[i] = rfEW(wave)
		self.zi = np.searchsorted(self.zbins,z)
	def _loadVW01Fe(self,wave):
		fepath = datadir+'VW01_Fe/'
		feTemplate = np.zeros_like(wave)
		for fn in ['Fe_UVtemplt_B.asc','Fe2_UV191.asc','Fe3_UV47.asc']:
			w,f = np.loadtxt(fepath+fn,unpack=True)
			spec = interp1d(w,f,kind='slinear')
			w1,w2 = np.searchsorted(wave,[w[0],w[-1]])
			feTemplate[w1:w2] += spec(wave[w1:w2])
		# continuum parameters given in VW01 pg. 6
		a_nu = -1.9
		fcont = 3.45e-14 * (wave/1500.)**(-2-a_nu)
		w1 = np.searchsorted(wave,1716.)
		a_nu = -1.0
		fcont[w1:] = 3.89e-14 * (wave[w1:]/1500.)**(-2-a_nu)
		feTemplate /= fcont
		return feTemplate
	def _restFrameFeTemplate(self,FWHM_kms,feScalings):
		wave = np.logspace(np.log(1075.),np.log(3089.),5000,base=np.e)
		feTemplate = self._loadVW01Fe(wave)
		# rescale segments of the Fe template
		if feScalings is None:
			feScalings = [(0,3500,1.0),]
		print 'using Fe scales: ',feScalings
		for w1,w2,fscl in feScalings:
			wi1,wi2 = np.searchsorted(wave,(w1,w2))
			feTemplate[wi1:wi2] *= fscl
		# calculate the total flux (actually, EW since continuum is divided out)
		flux0 = simps(feTemplate,wave)
		FWHM_1Zw1 = 900.
		c_kms = 3e5
		sigma_conv = np.sqrt(FWHM_kms**2 - FWHM_1Zw1**2) / \
		                     (2*np.sqrt(2*np.log(2))) / c_kms
		dloglam = np.log(wave[1]) - np.log(wave[0])
		x = np.arange(-5*sigma_conv,5*sigma_conv,dloglam)
		gkern = np.exp(-x**2/(2*sigma_conv**2)) / (np.sqrt(2*np.pi)*sigma_conv)
		broadenedTemp = convolve(feTemplate,gkern,mode='same')
		feFlux = broadenedTemp
		feFlux *= flux0/simps(feFlux,wave)
		return wave,feFlux
	def get(self,idx):
		return self.feGrid[self.zi[idx]]
	def getTable(self,hdr):
		'''Return a Table of all parameters and header information'''
		hdr.update('FETEMPL','Vestergaard & Wilkes 2001 Iron Emission')
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

