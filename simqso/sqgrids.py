#!/usr/bin/env python

import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import simps
from scipy.signal import convolve
from astropy import cosmology
from astropy.table import Table
from astropy.io.fits import Header,getdata
from astropy.io import ascii as ascii_io

from .sqbase import datadir,mag2lum
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
		elif isinstance(cosmodef,cosmology.FLRW):
			self.cosmo = cosmodef
		elif cosmodef is None:
			self.cosmo = cosmology.get_current()
		else:
			raiseValueError
	def distMod(self,z):
		return self.cosmo.distmod(z).value

class MzGridFromData(MzGrid):
	def __init__(self,mzdata,gridpar,hdr):
		self.units = hdr['GRIDUNIT']
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
		self.units = 'luminosity'
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

class FluxGrid(MzGrid):
	def __init__(self,cosmodef,**kwargs):
		self.obsBand = kwargs.get('obsBand','SDSS-i')
		self.restBand = kwargs.get('restBand',1450.)
		self.setCosmology(cosmodef)
		self.m2M = lambda z: mag2lum(self.obsBand,self.restBand,z,self.cosmo)
		self.units = 'flux'
	def updateMags(self,m):
		dm = m - self.mgrid
		print '--> delta mag mean = %.7f, rms = %.7f, max = %.7f' % \
		              (dm.mean(),dm.std(),dm.max())
		self.Mgrid[:] -= dm
	def resetAbsMag(self,Mgrid):
		self.Mgrid[:] = Mgrid

class FluxRedshiftGrid(FluxGrid):
	'''
	Construct a grid in (mag,z) having a fixed number of points within each 
	bin. The bin spacings need not be uniform, as long as they are 
	monotonically increasing.
	'''
	def __init__(self,medges,zedges,nPerBin,cosmodef,**kwargs):
		super(FluxRedshiftGrid,self).__init__(cosmodef,**kwargs)
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
		# distribute quasars into bins of flux 
		for i in range(self.nM):
			for j in range(self.nz):
				binm = medges[i] + dm[i]*np.random.rand(nPerBin)
				binz = zedges[j] + dz[j]*np.random.rand(nPerBin)
				zi = binz.argsort()
				self.mgrid[i,j,:] = binm[zi]
				self.zgrid[i,j,:] = binz[zi]
		# convert to luminosity
		self.Mgrid = self.mgrid - self.m2M(self.zgrid)
		self.Medges = np.empty((self.nz,self.nM+1))
		for j in range(self.nz):
			self.Medges[j] = self.medges - self.m2M(self.zedges[j])

class LuminosityFunctionFluxGrid(FluxGrid):
	def __init__(self,mRange,zRange,qlf,cosmodef,**kwargs):
		super(LuminosityFunctionFluxGrid,self).__init__(cosmodef,**kwargs)
		m,z = qlf.sample_from_fluxrange(mRange,zRange,self.m2M,cosmodef,**kwargs)
		self.mgrid = m
		self.zgrid = z
		self.nPerBin = len(z)
		self.Mgrid = m - self.m2M(z)

class FixedPLContinuumGrid(object):
	def __init__(self,M,z,slopes,breakpoints):
		self.slopes = np.asarray(slopes)
		self.breakpoints = np.asarray(breakpoints)
	def update(self,M,z):
		# continuum are fixed in luminosity and redshift
		return
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
	def update(self,M,z):
		# continuum are fixed in luminosity and redshift
		return
	def get(self,idx):
		return self.slopes[idx],self.breakpoints
	def getTable(self,hdr):
		'''Return a Table of all parameters and header information'''
		flatgrid = np.product(self.slopes.shape[:-1])
		t = Table({'slopes':self.slopes.reshape(flatgrid,-1)})
		hdr['CNTBKPTS'] = ','.join(['%.1f' % bkpt 
		                            for bkpt in self.breakpoints])
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
	def update(self,M,z):
		# lines are fixed in luminosity and redshift
		return
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
		hdr['LINEMODL'] = 'Fixed Vanden Berk et al. 2001 emission lines'
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
	def update(self,M,z):
		# template fixed in luminosity and redshift
		return
	def get(self,idx):
		return self.feGrid[self.zi[idx]]
	def getTable(self,hdr):
		'''Return a Table of all parameters and header information'''
		hdr['FETEMPL'] = 'Vestergaard & Wilkes 2001 Iron Emission'
		return None

class VariedEmissionLineGrid(object):
	def __init__(self,M1450,z,**kwargs):
		trendfn = kwargs.get('EmissionLineTrendFilename','emlinetrends_v5',)
		self.fixed = kwargs.get('fixLineProfiles',False)
		self.minEW = kwargs.get('minEW',0.0)
		indy = kwargs.get('EmLineIndependentScatter',False)
		# emission line trends are calculated w.r.t. M_i, so
		#  convert M1450 to M_i(z=0) using Richards+06 eqns. 1 and 3
		M_i = M1450 - 1.486 + 0.596
		# load the emission line trend data, and restrict to lines for which
		# the mean EW trend at the lowest luminosity exceeds minEW
		t = getdata(datadir+trendfn+'.fits')
		maxEW = 10**(t['logEW'][:,0,1]+t['logEW'][:,0,0]*M_i.max())
		ii = np.where(maxEW > self.minEW)[0]
		self.lineTrends = t[ii]
		nlines = len(ii)
		# random deviate used to sample line profile values - if independent
		# scatter, each line gets its own random scatter, otherwise each 
		# object gets a single deviation and the lines are perfectly correlated
		nx = nlines if indy else 1
		xshape = z.shape + (nx,)
		self.xv = {}
		for k in ['wavelength','logEW','logWidth']:
			self.xv[k] = np.random.standard_normal(xshape)
		# store the line profile values in a structured array with each
		# element having the same shape as the input grid
		nf4 = str(z.shape)+'f4'
		self.lineGrids = np.zeros(nlines,
		                      dtype=[('name','S15'),('wavelength',nf4),
		                             ('eqWidth',nf4),('width',nf4)])
		self.lineGrids['name'] = self.lineTrends['name']
		# the optional argument 'scaleEWs' allows the caller to arbitrarily
		# rescale both the output equivalent widths and the input scatter
		# to the EW distribution for individual lines. construct the
		# scaling vector here
		self.sigscl = np.ones(nlines)
		self.ewscl = np.ones(nlines)
		for line,scl in kwargs.get('scaleEWs',{}).items():
			i = np.where(self.lineGrids['name']==line)[0][0]
			if type(scl) is tuple:
				self.ewscl[i],self.sigscl[i] = scl
			else:
				self.ewscl[i] = scl
		self.update(M1450,z)
	def update(self,M1450,z):
		# derive the line profile parameters using the input luminosities
		# and redshifts
		M_i = M1450 - 1.486 + 0.596
		M = M_i[...,np.newaxis]
		# loop over the three line profile parameters and obtain the
		# randomly sampled values using the input trends
		for k in ['wavelength','logEW','logWidth']:
			a = self.lineTrends[k][...,0]
			b = self.lineTrends[k][...,1]
			meanVal = a[...,0]*M + b[...,0]
			siglo = a[...,1]*M + b[...,1] - meanVal
			sighi = a[...,2]*M + b[...,2] - meanVal
			x = self.xv[k]
			sig = np.choose(x<0,[sighi,-siglo])
			if k=='logEW':
				sig *= self.sigscl
			v = meanVal + x*sig
			if k.startswith('log'):
				v[:] = 10**v
				k2 = {'logEW':'eqWidth','logWidth':'width'}[k]
			else:
				k2 = k
			if k=='logEW':
				v *= self.ewscl
			self.lineGrids[k2] = np.rollaxis(v,-1)
	def get(self,idx):
		# shape of lineGrids is (Nlines,)+gridshape, and idx is an index
		# into the grid, so add a dummy slice along the first axis
		idx2 = (np.s_[:],)+idx
		return (self.lineGrids['wavelength'][idx2],
		        self.lineGrids['eqWidth'][idx2],
		        self.lineGrids['width'][idx2])
	def getTable(self,hdr):
		'''Return a Table of all parameters and header information'''
		hdr['LINEMODL'] = 'Log-linear trends with luminosity'
		hdr['LINENAME'] = ','.join(self.lineGrids['name'])
		flatgrid = np.product(self.lineGrids['wavelength'].shape[1:])
		def squash(a):
			return a.reshape(-1,flatgrid).transpose()
		t = Table({'lineWave':squash(self.lineGrids['wavelength']),
		           'lineEW':squash(self.lineGrids['eqWidth']),
		           'lineWidth':squash(self.lineGrids['width']),})
		return t

class FixedDustGrid(object):
	def __init__(self,M,z,dustModel,E_BmV):
		self.dustModel = dustModel
		self.E_BmV = E_BmV
		self.dust_fn = dustextinction.dust_fn[dustModel]
	def update(self,M,z):
		# fixed in luminosity and redshift
		return
	def get(self,idx):
		return self.dust_fn,self.E_BmV
	def getTable(self,hdr):
		'''Return a Table of all parameters and header information'''
		hdr['DUSTMODL'] = self.dustModel
		hdr['FIXEDEBV'] = self.E_BmV
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
	def update(self,M,z):
		# fixed in luminosity and redshift
		return
	def get(self,idx):
		return self.dust_fn,self.EBVdist[idx]
	def getTable(self,hdr):
		'''Return a Table of all parameters and header information'''
		t = Table({'E(B-V)':self.EBVdist.flatten()})
		hdr['DUSTMODL'] = self.dustModel
		hdr['EBVSCALE'] = self.E_BmV_scale
		return t

