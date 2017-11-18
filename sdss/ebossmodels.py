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
}

def add_continuum(qsos,name='def_plcontinuum',const=False):
	slopes,breakpts = cont_models[name]
	if const:
		slopes = [ grids.ConstSampler(s[0]) for s in slopes]
	else:
		slopes = [ grids.GaussianSampler(*s) for s in slopes],
	contVar = grids.BrokenPowerLawContinuumVar(slopes,breakpts)
	qsos.addVar(contVar)
	return qsos

def add_dust_emission(qsos,name='LR17',const=False):
	contVar = qsos.getVars(grids.ContinuumVar)[0]
	subDustVar = grids.DustBlackbodyVar([grids.ConstSampler(0.05),
	                                     grids.ConstSampler(1800.)],
	                                     name='sublimdust')
	subDustVar.set_associated_var(contVar)
	hotDustVar = grids.DustBlackbodyVar([grids.ConstSampler(0.2),
	                                     grids.ConstSampler(880.)],
	                                    name='hotdust')
	hotDustVar.set_associated_var(contVar)
	qsos.addVars([subDustVar,hotDustVar])
	return qsos

def add_emission_lines(qsos,name='bossdr9',const=False):
	if name == 'bossdr9':
		emLineVar = sqmodels.BossDr9_EmLineTemplate(qsos.absMag,
		                                            NoScatter=const)
	elif name == 'yang16':
		emLineVar = sqmodels.get_Yang16_EmLineTemplate(qsos.absMag,
		                                               NoScatter=const)
	else:
		kwargs = emline_models.get(name,{})
		kwargs['NoScatter'] = const
		emLineVar = grids.generateBEffEmissionLines(qsos.absMag,**kwargs)
	qsos.addVar(emLineVar)
	return qsos

def add_iron(qsos,wave,name='def_iron',const=False):
	fetempl = grids.VW01FeTemplateGrid(qsos.z,wave,
	                                   scales=sqmodels.BossDr9_FeScalings)
	feVar = grids.FeTemplateVar(fetempl)
	qsos.addVar(feVar)
	return qsos

def add_dust_extinction(qsos,name='dr9expdust',const=False):
	if const:
		s = grids.ConstSampler(0.03)
	else:
		s = grids.ExponentialSampler(0.03)
	dustVar = grids.SMCDustVar(s)
	qsos.addVar(dustVar)
	return qsos

