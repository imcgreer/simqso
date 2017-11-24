#!/usr/bin/env python

from simqso import sqmodels
from simqso import sqgrids as grids

cont_models = {
  'bossdr9':[[(-1.50,0.3),(-0.50,0.3),(-0.37,0.3),(-1.70,0.3),(-1.03,0.3)],
             [1100.,5700.,9730.,22300.]],
  'dr9expdust':[[(-0.50,0.3),(-0.30,0.3),(-0.37,0.3),
                 (-1.70,0.3),(-1.03,0.3)],
                 [1100.,5700.,9730.,22300.]],
  'def_plcontinuum':[[(-1.5,0.3),(-0.4,0.3)],[1200.]],
}

emline_models = {
  'newb':{
    'scaleEWs':{'LyAb':1.4,'LyAn':1.4,'CIVb':0.75,'CIVn':0.75,
                 'CIII]b':0.8,'CIII]n':0.8,'MgIIb':0.8,'MgIIn':0.8,
                 'Hbeta':1.2,'HAb':1.5,'HAn':1.5},
    },
}

dustem_models = {
  'LR17':{'sublimdust':[(0.05,None),(1800.,None)],
             'hotdust':[(0.2,None),(880.,None)]},
  'LR17b':{'sublimdust':[(0.05,None),(1800.,None)],
             'hotdust':[(0.1,None),(880.,None)]},
  'GHW06':{'hotdust':[(0.1,None),(1260.,None)]},
  'GHW06b':{'sublimdust':[(0.05,None),(1800.,None)],
           'hotdust':[(0.1,None),(1260.,None)]},
}

qso_models = {
  'bossdr9':{'continuum':'bossdr9','emlines':'bossdr9','iron':'def_iron'},
  'dr9expdust':{'continuum':'dr9expdust','emlines':'bossdr9',
                'iron':'def_iron','dustext':'dr9expdust'},
  'new':{'continuum':'def_plcontinuum','emlines':'bossdr9',
         'dustem':'LR17','iron':'def_iron'},
  'newb':{'continuum':'def_plcontinuum','emlines':'newb',
         'dustem':'LR17b','iron':'def_iron'},
  'newc':{'continuum':'def_plcontinuum','emlines':'newb',
         'dustem':'GHW06','iron':'def_iron'},
  'newd':{'continuum':'def_plcontinuum','emlines':'newb',
         'dustem':'GHW06b','iron':'def_iron'},
}

def add_continuum(qsos,model='def_plcontinuum',const=False):
	try:
		slopes,breakpts = cont_models[model]
	except:
		slopes,breakpts = model
	if const:
		slopes = [ grids.ConstSampler(s[0]) for s in slopes]
	else:
		slopes = [ grids.GaussianSampler(*s) for s in slopes]
	contVar = grids.BrokenPowerLawContinuumVar(slopes,breakpts)
	qsos.addVar(contVar)
	return qsos

def add_dust_emission(qsos,model='LR17',const=False):
	contVar = qsos.getVars(grids.ContinuumVar)[0]
	if isinstance(model,basestring):
		model = dustem_models[model]
	dustVars = []
	for name,par in model.items():
		dustVar = grids.DustBlackbodyVar([grids.ConstSampler(par[0][0]),
		                                  grids.ConstSampler(par[1][0])],
	                                     name=name)
		dustVar.set_associated_var(contVar)
		dustVars.append(dustVar)
	qsos.addVars(dustVars)
	return qsos

def add_emission_lines(qsos,model='bossdr9',const=False):
	if model == 'bossdr9':
		emLineVar = sqmodels.BossDr9_EmLineTemplate(qsos.absMag,
		                                            NoScatter=const)
	elif model == 'yang16':
		emLineVar = sqmodels.get_Yang16_EmLineTemplate(qsos.absMag,
		                                               NoScatter=const)
	else:
		if isinstance(model,basestring):
			kwargs = emline_models[model]
		else:
			kwargs = model
		kwargs['NoScatter'] = const
		emLineVar = grids.generateBEffEmissionLines(qsos.absMag,**kwargs)
	qsos.addVar(emLineVar)
	return qsos

def add_iron(qsos,wave,model='def_iron',const=False):
	fetempl = grids.VW01FeTemplateGrid(qsos.z,wave,
	                                   scales=sqmodels.BossDr9_FeScalings)
	feVar = grids.FeTemplateVar(fetempl)
	qsos.addVar(feVar)
	return qsos

def add_dust_extinction(qsos,model='dr9expdust',const=False):
	if const:
		s = grids.ConstSampler(0.033)
	else:
		s = grids.ExponentialSampler(0.033)
	dustVar = grids.SMCDustVar(s)
	qsos.addVar(dustVar)
	return qsos

