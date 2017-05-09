#!/usr/bin/env python

from astropy.cosmology import FlatLambdaCDM
from . import sqgrids as grids
from .lumfun import PolyEvolParam,DoublePowerLawLF
from .hiforest import IGMTransmissionGrid

Fan99_model = {
  'forest':{'zrange':(0.0,6.0),
            'logNHrange':(13.0,17.3),
            'N0':50.3,
            'gamma':2.3,
            'beta':1.41,
            'b':30.0},
     'LLS':{'zrange':(0.0,6.0),
            'logNHrange':(17.3,20.5),
            'N0':0.27,
            'gamma':1.55,
            'beta':1.25,
            'b':70.0},
     'DLA':{'zrange':(0.0,6.0),
            'logNHrange':(20.5,22.0),
            'N0':0.04,
            'gamma':1.3,
            'beta':1.48,
            'b':70.0},
}

WP11_model = {
 'forest0':{'zrange':(0.0,1.5),
            'logNHrange':(12.0,19.0),
            'gamma':0.2,
            'beta':1.55,
            'B':0.0170,
            'N0':340.,
            'brange':(10.,100.),
            'bsig':24.0},
 'forest1':{'zrange':(1.5,4.6),
            'logNHrange':(12.0,14.5),
            'gamma':2.04,
            'beta':1.50,
            'B':0.0062,
            'N0':102.0,
            'brange':(10.,100.),
            'bsig':24.0},
 'forest2':{'zrange':(1.5,4.6),
            'logNHrange':(14.5,17.5),
            'gamma':2.04,
            'beta':1.80,
            'B':0.0062,
            'N0':4.05,
            'brange':(10.,100.),
            'bsig':24.0},
 'forest3':{'zrange':(1.5,4.6),
            'logNHrange':(17.5,19.0),
            'gamma':2.04,
            'beta':0.90,
            'B':0.0062,
            'N0':0.051,
            'brange':(10.,100.),
            'bsig':24.0},
    'SLLS':{'zrange':(0.0,4.6),
            'logNHrange':(19.0,20.3),
            'N0':0.0660,
            'gamma':1.70,
            'beta':1.40,
            'brange':(10.,100.),
            'bsig':24.0},
     'DLA':{'zrange':(0.0,4.6),
            'logNHrange':(20.3,22.0),
            'N0':0.0440,
            'gamma':1.27,
            'beta':2.00,
            'brange':(10.,100.),
            'bsig':24.0},
}

McG13hiz_model = {
 'forest1':{'zrange':(1.5,10.1),
            'logNHrange':(12.0,14.5),
            'gamma':3.5,
            'beta':1.50,
            'N0':8.5 * 1.1,
            'brange':(10.,100.),
            'bsig':24.0},
 'forest2':{'zrange':(1.5,10.1),
            'logNHrange':(14.5,17.2),
            'gamma':3.5,
            'beta':1.70,
            'N0':0.33 * 1.1,
            'brange':(10.,100.),
            'bsig':24.0},
     'LLS':{'zrange':(1.5,10.1),
            'logNHrange':(17.2,20.3),
            'gamma':2.0,
            'beta':1.3,
            'N0':0.13 * 1.1,
            'brange':(10.,100.),
            'bsig':24.0},
  'subDLA':{'zrange':(0.0,10.1),
            'logNHrange':(20.3,21.0),
            'N0':0.13 / 7.5 * 1.1,
            'gamma':1.70,
            'beta':1.28,
            'brange':(10.,100.),
            'bsig':24.0},
     'DLA':{'zrange':(0.0,10.1),
            'logNHrange':(21.0,22.0),
            'N0':0.13 / 33 * 1.1,
            'gamma':2.0,
            'beta':1.40,
            'brange':(10.,100.),
            'bsig':24.0},
}

forestModels = {'Fan1999':Fan99_model,
                'Worseck&Prochaska2011':WP11_model,
                'McGreer+2013':McG13hiz_model}

BossDr9_fiducial_continuum = grids.BrokenPowerLawContinuumVar([
                                    grids.GaussianSampler(-1.50,0.3),
                                    grids.GaussianSampler(-0.50,0.3),
                                    grids.GaussianSampler(-0.37,0.3),
                                    grids.GaussianSampler(-1.70,0.3),
                                    grids.GaussianSampler(-1.03,0.3) ],
                                    [1100.,5700.,9730.,22300.])

BossDr9_expDust_cont = grids.BrokenPowerLawContinuumVar([
                                    grids.GaussianSampler(-0.50,0.2),
                                    grids.GaussianSampler(-0.30,0.2),
                                    grids.GaussianSampler(-0.37,0.3),
                                    grids.GaussianSampler(-1.70,0.3),
                                    grids.GaussianSampler(-1.03,0.3) ],
                                    [1100.,5700.,9730.,22300.])

BossDr9_FeScalings = [ (0,1540,0.5),(1540,1680,2.0),(1680,1868,1.6),
                       (1868,2140,1.0),(2140,3500,1.0) ]

def BossDr9_EmLineTemplate(*args):
	kwargs = {'scaleEWs':{'LyAb':1.1,'LyAn':1.1,
	                      'CIVb':0.75,'CIVn':0.75,
	                      'CIII]b':0.8,'CIII]n':0.8,
	                      'MgIIb':0.8,'MgIIn':0.8}}
	return grids.generateBEffEmissionLines(*args,**kwargs)

def get_BossDr9_model_vars(qsoGrid,wave,nSightLines):
	igmGrid = IGMTransmissionGrid(wave,
	                              forestModels['Worseck&Prochaska2011'],
	                              nSightLines,zmax=qsoGrid.z.max())
	fetempl = grids.VW01FeTemplateGrid(qsoGrid.z,wave,
	                                   scales=BossDr9_FeScalings)
	mvars = [ BossDr9_fiducial_continuum,
	          BossDr9_EmLineTemplate(qsoGrid.absMag),
	          grids.FeTemplateVar(fetempl),
	          grids.HIAbsorptionVar(igmGrid) ]
	return mvars


Yang16_continuum = grids.BrokenPowerLawContinuumVar([
                                    grids.GaussianSampler(-1.50,0.3),
                                    grids.GaussianSampler(-0.44,0.3),
                                    grids.GaussianSampler(-0.48,0.3),
                                    grids.GaussianSampler(-1.74,0.3),
                                    grids.GaussianSampler(-1.17,0.3) ],
                                    [1100.,5700.,9730.,23820.])

def get_Yang16_EmLineTemplate(*args):
	kwargs = {'scaleEWs':{'LyAb':1.1,'LyAn':1.1,
	                      'CIVb':1.2,'CIVn':1.2,
	                      'CIII]b':1.0,'CIII]n':1.0,
	                      'MgIIb':1.2,'MgIIn':1.2,
	                      'Hbeta':1.2,'[OIII]4364':1.2,
	                      'HAn':1.0,'HAb':1.0}}
	return grids.generateBEffEmissionLines(*args,**kwargs)

class LogPhiStarEvolFixedK(PolyEvolParam):
	def __init__(self,logPhiStar_zref,k=-0.47,fixed=False,zref=6.0):
		super(LogPhiStarEvolFixedK,self).__init__([k,logPhiStar_zref],
		                                          fixed=[True,fixed],
		                                          z0=zref)

QLF_McGreer_2013 = DoublePowerLawLF(LogPhiStarEvolFixedK(-8.94),
                                    -27.21,-2.03,-4.0,
                                    )#cosmo=FlatLambdaCDM(H0=70, Om0=0.272))

