#!/usr/bin/env python

import inspect
from collections import OrderedDict
import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import quad
from scipy.special import hyp2f1

skyDeg2 = 41253.

def interp_dVdzdO(zrange,cosmo):
	zz = np.arange(zrange[0]-0.01,zrange[1]+0.0251,0.025)
	diff_co_vol = [cosmo.differential_comoving_volume(z).value for z in zz]
	return interp1d(zz,diff_co_vol)

class LuminosityFunction(object):
	def __init__(self):
		self.paramEvol = OrderedDict()
		self.nargs = {}
	def logPhi(self,M,z,*args):
		raise NotImplementedError
	def __call__(self,*args):
		raise NotImplementedError
	def logPhi(self,M,z,*args):
		raise NotImplementedError
	def Phi(self,M,z,*args):
		return 10**self.logPhi(M,z,*args)
	def set_scale(self,scale):
		if scale == 'log':
			self.__call__ = self.logPhi
			self.scale = 'log'
		elif scale == 'linear':
			self.__call__ = self.Phi
			self.scale = 'linear'
		else:
			raise ValueError
	def get_scale(self):
		return self.scale
	def set_param_evol(self,paramName,z_evol):
		'''first argument of z_evol must be z'''
		if type(z_evol) is float:
			# constant with redshift
			self.paramEvol[paramName] = lambda z: z_evol
		else:
			self.paramEvol[paramName] = z_evol
		argspec = inspect.getargspec(self.paramEvol[paramName])
		self.nargs[paramName] = len(argspec[0]) - 1
	def set_param_values(self,p):
		self.paramVals = p
	def eval_at_z(self,z,*args):
		argnum = 0
		vals = []
		for p in self.paramEvol:
			if len(args) > 0:
				pp = args[argnum:argnum+self.nargs[p]]
			else:
				if self.nargs[p]==0:
					pp = ()
				else:
					pp = self.paramVals[argnum:argnum+self.nargs[p]]
			vals.append(self.paramEvol[p](z,*pp))
			argnum += self.nargs[p]
		return vals

def doublePL_Lintegral(x,a,b):
	return (b*x**(1-a) / ((a-1)*(a-b)) - 
	         x**(1-a)*((a-b)* 
	          hyp2f1(1.,(a-1)/(a-b),(a-1)/(a-b)+1,-x**(b-a)) + b) / 
	           ((a-1)*(a-b)))

def integrateDPL(Mrange,logPhiStar,MStar,alpha,beta):
	PhiStar_M = 10**logPhiStar
	MM = np.asarray(Mrange)
	L_min,L_max = 10**(-0.4*(MM-MStar))
	Lsum = doublePL_Lintegral(L_max,-alpha,-beta) - \
	        doublePL_Lintegral(L_min,-alpha,-beta) 
	return 1.0857 * PhiStar_M * Lsum

class DoublePowerLawLF(LuminosityFunction):
	def __init__(self,logPhiStar,MStar,alpha,beta):
		super(DoublePowerLawLF,self).__init__()
		self.scale = 'log'
		self.set_param_evol('logPhiStar',logPhiStar)
		self.set_param_evol('MStar',MStar)
		self.set_param_evol('alpha',alpha)
		self.set_param_evol('beta',beta)
	def logPhi(self,M,z,*args):
		logPhiStar,Mstar,alpha,beta = self.eval_at_z(z,*args)
		return logPhiStar - \
		        np.log10(10**(0.4*(alpha+1)*(M-Mstar)) + \
		                 10**(0.4*( beta+1)*(M-Mstar)))
	__call__ = logPhi
	def _sample(self,Mrange,zrange,p,cosmo,**kwargs):
		nz = 100
		nM = 30
		skyfrac = kwargs.get('skyArea',skyDeg2) / skyDeg2
		Mmin = Mrange(zrange[0])[0]
		Mmax = Mrange(zrange[1])[1]
		dVdzdO = interp_dVdzdO(zrange,cosmo)
		phi_z = lambda z: integrateDPL((Mmin,Mmax),*self.eval_at_z(z,*p)) * \
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
			if ((i+1)%100)==0:
				print i+1,' out of ',Ntot
		return M,z
	def _sample_at_intervals(self,Mrange,z,p,cosmo,Nintervals):
		# for colorz
		raise NotImplementedError
	def sample_from_fluxrange(self,mrange,zrange,m2M,cosmo,p=(),**kwargs):
		_mrange = mrange[::-1]
		_Mrange = lambda z: np.array(_mrange) - m2M(z)
		M,z = self._sample(_Mrange,zrange,p,cosmo,**kwargs)
		m = M + m2M(z)
		return m,z
	def sample_from_Lrange(self,Mrange,zrange,cosmo,p=(),**kwargs):
		_Mrange = lambda z: Mrange
		return self._sample(_Mrange,zrange,p,cosmo,**kwargs)

