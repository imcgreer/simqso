#!/usr/bin/env python

import inspect
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

