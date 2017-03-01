#!/usr/bin/env python

from copy import deepcopy
from collections import OrderedDict
import numpy as np
import astropy.units as u
from scipy.interpolate import interp1d
from scipy.integrate import quad
from scipy.special import hyp2f1

skyDeg2 = 41253.

def interp_dVdzdO(zrange,cosmo):
	zz = np.arange(zrange[0]-0.01,zrange[1]+0.0251,0.025)
	diff_co_vol = [cosmo.differential_comoving_volume(z).value for z in zz]
	return interp1d(zz,diff_co_vol)

def doublePL_Lintegral(x,a,b):
	return (b*x**(1-a) / ((a-1)*(a-b)) - 
	         x**(1-a)*((a-b)* 
	          hyp2f1(1.,(a-1)/(a-b),(a-1)/(a-b)+1,-x**(b-a)) + b) / 
	           ((a-1)*(a-b)))

# handy conversions are given in Hopkins, Richards, & Hernquist 2007
# eqns. 6-8
def integrateDPL(Mrange,logPhiStar,MStar,alpha,beta):
	PhiStar_M = 10**logPhiStar
	MM = np.asarray(Mrange)
	L_min,L_max = 10**(-0.4*(MM-MStar))
	try:
		Lsum = doublePL_Lintegral(L_max,-alpha,-beta) - \
		        doublePL_Lintegral(L_min,-alpha,-beta) 
	except:
		Lsum = np.inf # proceed to the failure case below
	if not np.isfinite(Lsum):
		# if the analytic function failed to evaluate, revert to a numerical
		# integration
		Lsum,err = quad(lambda x: 1/(x**-alpha + x**-beta), L_min, L_max)
	return 1.0857 * PhiStar_M * Lsum

class QlfEvolParam(object):
	def __init__(self,par,fixed=False,z0=0.):
		'''par: initial values
		   fixed: False means none, True means all, otherwise same shape as x
		   z0: evaluate at z+z0; i.e., z0=1 means (1+z), z0=-6 means (z-6.0)
		'''
		# this assures even float values are converted to length-1 arrays
		par = np.asarray(par).astype(np.float64) * np.ones(1)
		self.par = np.ma.array(par,mask=fixed)
		self.z0 = z0
	@staticmethod
	def _tostr(x,m):
		s = '%8g' % x
		if m:
			s = '[%s]' % s
		return s
	def __str__(self):
		return ','.join([self._tostr(p,m) 
		             for p,m in zip(self.par.data,self.par.mask)])
	def set(self,val,i=None):
		if self.par.mask.all():
			return
		if i is None:
			i = np.where(~self.par.mask)[0]
		n = 1 if np.isscalar(i) else len(i)
		self.par.data[i] = [ val.pop(0) for j in range(n) ]
	def get(self):
		return self.par.compressed()
	def fix(self,i=None):
		if i is None:
			i = np.s_[:]
		self.par[i] = np.ma.masked
	def free(self,i=None):
		if i is None:
			i = np.s_[:]
		self.par[i] = self.par.data[i]
	def _extract_par(self,par):
		rv = self.par.data.copy()
		if par is not None:
			n = (~self.par.mask).sum()
			rv[~self.par.mask] = [ par.pop(0) for i in range(n) ]
		return rv

class PolyEvolParam(QlfEvolParam):
	def eval_at_z(self,z,par=None):
		par = self._extract_par(par)
		return np.polyval(par,z+self.z0)

class LogPhiStarEvolFixedK(PolyEvolParam):
	def __init__(self,logPhiStar_zref,k=-0.47,fixed=False,zref=6.0):
		super(LogPhiStarEvolFixedK,self).__init__([k,logPhiStar_zref],
		                                          fixed=[True,fixed],
		                                          z0=-zref)

# mostly just a placeholder in case other forms of LF get added
class LuminosityFunction(object):
	def __init__(self):
		self.set_scale('log')
	def __str__(self):
		s = ''
		for pname,p in self.params.items():
			s += '%15s:  %s\n' % (pname,str(p))
		return s
	def logPhi(self,M,z,*args):
		raise NotImplementedError
	def Phi(self,M,z,*args):
		return 10**self.logPhi(M,z,*args)
	def __call__(self,M,z,par=None):
		return self._call(M,z,par)
	def set_scale(self,scale):
		if scale not in ['log','linear']:
			raise ValueError
		self.scale = scale
		if scale == 'log':
			self._call = self.logPhi
		else:
			self._call = self.Phi
	def copy(self):
		return deepcopy(self)

class DoublePowerLawLF(LuminosityFunction):
	def __init__(self,logPhiStar=None,MStar=None,alpha=None,beta=None):
		'''each param is either a QlfEvolParam, or values to initialize
		   a PolyEvolParam, which is the default
		'''
		super(DoublePowerLawLF,self).__init__()
		self.params = OrderedDict()
		self.params['logPhiStar'] = self._resolvepar(logPhiStar)
		self.params['MStar'] = self._resolvepar(MStar)
		self.params['alpha'] = self._resolvepar(alpha)
		self.params['beta'] = self._resolvepar(beta)
	@staticmethod
	def _resolvepar(p):
		if isinstance(p,QlfEvolParam):
			return p
		else:
			return PolyEvolParam(p)
	def _iterpars(self):
		for p in self.params.values():
			yield p
	def getpar(self):
		return np.concatenate([ p.get() for p in self._iterpars() ])
	def setpar(self,par):
		par = list(par)
		for p in self._iterpars():
			p.set(par)
	def logPhi(self,M,z,par=None):
		if par is not None:
			par = list(par)
		logPhiStar,Mstar,alpha,beta = [ p.eval_at_z(z,par)
		                                   for p in self._iterpars() ]
		if par is not None and len(par) > 0:
			raise ValueError
		return logPhiStar - \
		        np.log10(10**(0.4*(alpha+1)*(M-Mstar)) + \
		                 10**(0.4*( beta+1)*(M-Mstar)))
	def _sample(self,Mrange,zrange,p,cosmo,**kwargs):
		# XXX make this more sensible
		nz = 100
		nM = 30
		skyfrac = kwargs.get('skyArea',skyDeg2) / skyDeg2
		dVdzdO = interp_dVdzdO(zrange,cosmo)
		phi_z = lambda z: integrateDPL(Mrange(z),*self.eval_at_z(z,*p)) * \
		                        dVdzdO(z)
		zbins = np.linspace(zrange[0],zrange[1],nz)
		zsamp = [quad(phi_z,*zr)[0] for zr in zip(zbins[:-1],zbins[1:])]
		zsamp = [0,] + zsamp
		Ntot = np.sum(zsamp)
		zfun = interp1d(np.cumsum(zsamp)/Ntot,zbins)
		Ntot = np.int(np.round(Ntot * skyfrac * 4*np.pi))
		print 'integration returned ',Ntot,' objects'
		x = np.random.random(Ntot)
		y = np.random.random(Ntot)
		z = zfun(x)
		M = np.zeros_like(z)
		for i in range(Ntot):
			Mr = Mrange(z[i])
			Mbins = np.linspace(Mr[0],Mr[1],nM)
			_p = self.eval_at_z(z[i],*p)
			Msamp = [integrateDPL(Mr,*_p) for Mr in zip(Mbins[:-1],Mbins[1:])]
			Msamp = [0.,] + Msamp
			N_M = np.sum(Msamp)
			Mfun = interp1d(np.cumsum(Msamp)/N_M,Mbins)
			M[i] = Mfun(y[i])
			if ((i+1)%(Ntot//10))==0:
				print i+1,' out of ',Ntot
		return M,z
	def sample_from_fluxrange(self,mrange,zrange,m2M,cosmo,p=(),**kwargs):
		_mrange = mrange[::-1]
		_Mrange = lambda z: np.array(_mrange) - m2M(z)
		M,z = self._sample(_Mrange,zrange,p,cosmo,**kwargs)
		m = M + m2M(z)
		return m,z
	def sample_from_Lrange(self,Mrange,zrange,cosmo,p=(),**kwargs):
		_Mrange = lambda z: Mrange
		return self._sample(_Mrange,zrange,p,cosmo,**kwargs)
	def _get_Lcdf_fun(self,Mrange,z,p):
		nM = 30
		Mbins = np.linspace(Mrange[0],Mrange[1],nM)
		logPhiStar,MStar,alpha,beta = self.eval_at_z(z,*p)
		Lbins = 10**(-0.4*(Mbins-MStar))
		Lcdf = doublePL_Lintegral(Lbins[1:],-alpha,-beta) - \
		        doublePL_Lintegral(Lbins[0],-alpha,-beta) 
		Lcdf /= Lcdf[-1]
		Lcdf = np.concatenate([[0.,],Lcdf])
		return interp1d(Lcdf,Mbins)
	def sample_at_flux_intervals(self,mrange,zbins,m2M,Nintervals,nPerBin,p=()):
		_mrange = np.array(mrange[::-1])
		medges = np.empty((Nintervals+1,len(zbins)))
		mgrid = np.empty((Nintervals,len(zbins),nPerBin))
		xedges = np.linspace(0.,1,Nintervals+1)
		for j,z in enumerate(zbins):
			Mrange = _mrange - m2M(z)
			Lcdf_fun = self._get_Lcdf_fun(Mrange,z,p)
			medges[:,j] = Lcdf_fun(xedges)[::-1] + m2M(z)
			for i in range(Nintervals):
				x = xedges[i] + (xedges[i+1]-xedges[i])*np.random.random(nPerBin)
				mgrid[i,j,:] = Lcdf_fun(x) + m2M(z)
		return medges,mgrid
	def integrate(self,mrange,zrange,m2M,cosmo,p=()):
		dVdzdO = interp_dVdzdO(zrange,cosmo)
		Mrange = lambda z: np.array(mrange) - m2M(z)
		phi_z = lambda z: integrateDPL(Mrange(z),*self.eval_at_z(z,*p)) * \
		                        dVdzdO(z)
		nqso,err = quad(phi_z,*zrange)
		nqso *= 4*np.pi
		return nqso
	def ionizing_emissivity(self,z,Mrange,p=(),**kwargs):
		logPhiStar,MStar,alpha,beta = self.eval_at_z(z,*p)
		# phi(L) integral is over (L/L*)^-alpha
		# Lphi(L) integral is over (L/L*)^-alpha-1, so send (alpha+1)
		x = integrateDPL(Mrange,logPhiStar,MStar,alpha+1,beta+1)
		if True:
			# until astropy provides AB mag -> Lnu conversion
			c = 4.*np.pi*(10*u.pc.to(u.cm))**2
			LStar_nu = c * 10**(-0.4*(MStar + 48.6))
		if True:
			# until I get something more flexible in here
			# ... use the power-law conversion in .spectrum
			break_wave = kwargs.get('break_wave',1100.)
			alpha1 = kwargs.get('alpha1',-0.5)
			alpha2 = kwargs.get('alpha2',-1.5)
			#print 'warning: using SED model (%.2f,%.1f,%.2f)' % \
			#       (alpha2,break_wave,alpha1)
			l912 = (1450./break_wave)**alpha1 * (break_wave/912.)**alpha2
		# for now return e1450, e912
		return LStar_nu * x, LStar_nu * l912 * x

QLF_McGreer_2013 = DoublePowerLawLF(LogPhiStarEvolFixedK(-8.94),
                                    -27.21,-2.03,-4.0)

