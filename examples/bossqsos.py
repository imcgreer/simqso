#!/usr/bin/env python

from astropy.cosmology import FlatLambdaCDM
from simqso import qsoSimulation,lumfun,sqmodels

dr9cosmo = FlatLambdaCDM(70,1-0.7,name='BOSSDR9')

def BOSS_DR9_PLE(which=1):
    if which==1:
        row = -1.16,-3.37,-22.85,1.241,-0.249,-5.96
    alpha,beta,MStar_i_z0,k1,k2,logPhiStar = row
    MStar1450_z0 = MStar_i_z0 + 1.486
    MStar = lumfun.PolyEvolParam([-2.5*k2,-2.5*k1,MStar1450_z0])
    return lumfun.DoublePowerLawLF(logPhiStar,MStar,alpha,beta,
                                   cosmo=dr9cosmo)

def BOSS_DR9_LEDE():
    c1,c2 = -0.689, -0.809
    logPhiStar_z2_2 = -5.83
    MStar_i_z2_2 = -26.49
    MStar1450_z0 = MStar_i_z2_2 + 1.486 # --> M1450
    MStar = lumfun.PolyEvolParam([c2,MStar1450_z0],z0=2.2)
    logPhiStar = lumfun.PolyEvolParam([c1,logPhiStar_z2_2],z0=2.2)
    alpha = -1.31
    beta = -3.45
    return lumfun.DoublePowerLawLF(logPhiStar,MStar,alpha,beta,
                                   cosmo=dr9cosmo)

simParams = {
  # filename for simulation output (".fits" is appended)
  'FileName':'boss_dr9qlf_sim',
  # wavelength range of simulated spectra in Angstroms
  'waveRange':(3000.,3.0e5),
  # constant spectral dispersion per pixel as R = d(log(lam))
  'SpecDispersion':500,
  # dispersion scale is logarithmic [only option for now]
  'DispersionScale':'logarithmic',
  # set the cosmology, any appropriate instance from astropy.cosmology allowed
  'Cosmology':dr9cosmo,
  # setting a global random seed allows the simulation to be repeatable
  'RandomSeed':12345,
  # Define the "grid" of points in (M,z) space for the simulation
  # In this case the grid is a distribution of points sampled from the    
  # Ross et al. 2013 QLF determined from BOSS DR9.
  'GridParams':{
    # Define the grid as coming from a luminosity function
    'GridType':'LuminosityFunction', 
    # Specify the functional form of the LF, using a double power-law
    # with evolutionary parameters from Ross et al. 2013
    'QLFmodel':BOSS_DR9_LEDE(),
    # simulate a 10k deg2 survey
    #'QLFargs':{'skyArea':1e4},       
    'QLFargs':{'skyArea':10}, # but only 10 deg2 will run a lot faster
    # set bright and faint flux limits
    'mRange':(17.0,21.85),
    # and redshift range
    'zRange':(2.0,4.0),
    # flux range defined in r-band
    'ObsBand':'SDSS-r',
    # rest luminosities defined at 1450A
    'RestBand':1450.,
  },
  # Define the model for generating quasar emission spectra
  'QuasarModelParams':{
    # underlying continuum, only option is power law
    'ContinuumParams':{
      # power law slopes have a gaussian distribution
      'ContinuumModel':'BrokenPowerLaw',
      # the continuum consists of a series of broken power laws with
      # independent slope distributions 
      # the format is [ (meanSlope, stdSlope), breakWavelength, ... ]
      'PowerLawSlopes':[(-1.5,0.3),1100,(-0.5,0.3),
                    5700,(-0.37,0.3),9730,(-1.7,0.3),22300,(-1.03,0.3)],
    },
    # the emission line model
    'EmissionLineParams':{
      # the emission line profile distribution comes from the BOSS DR9 model
      # allowing for the Baldwin Effect
      'EmissionLineModel':'VariedEmissionLineGrid',
      # these are rescalings of the equivalent widths in the model, determined
      # emprically by matching colors with BOSS quasars. I.e., the Lyman alpha
      # EW is 10% greater than the nominal value in the model.
      'scaleEWs':{'LyAb':1.1,'LyAn':1.1,
                  'CIVb':0.75,'CIVn':0.75,
                  'CIII]b':0.8,'CIII]n':0.8,
                  'MgIIb':0.8,'MgIIn':0.8},
    },
    # the Fe emission template from Vestergaard & Wilkes 2001
    'IronEmissionParams':{
      # rescalings of sections of the template, empirically determined fitting
      # of composite BOSS quasar spectra
      'FeScalings':[(0,1540,0.5),(1540,1680,2.0),(1680,1868,1.6),
                    (1868,2140,1.0),(2140,3500,1.0)],
    },
  },
  # define the model for transmission spectra through the HI forest, based on
  # Monte Carlo realizations of absorption systems
  'ForestParams':{
    # filename to save the forest transmission spectra
#    'FileName':'boss_dr9qlf_forest',
    # name of the model for the distribution of absorbers 
    'ForestModel':sqmodels.forestModels['Worseck&Prochaska2011'], 
    # redshift range over which to sample absorbers
    'zRange':(0.0,4.5),
    # the number of independent sightlines to generate
    # WP11 suggest a minimum of 2000 to properly sample the scatter
    #'NumLinesOfSight':2000,
    'NumLinesOfSight':200, # however, 200 will run a lot faster for testing
    # the minimum spectral dispersion to use when generating the transmission
    # spectra; R=30000 => 10 km/s is a good value to capture the weak systems
    'Rmin':30000.,
  },
  # define the photometric systems for the survey, namely, the bandpasses
  # for calculating synthetic photometry from the spectra, and an error model
  # for producing realistic fluxes and errors
  'PhotoMapParams':{
    # list the systems individually, the output 'synMag' and 'obsMag' fields
    # will have a final dimension equal to the total number of bandpasses,
    # in the order listed here. I.e., for this simulation the shape is
    # synMag[...,9], with synMag[...,0:5] representing SDSS 'ugriz' and
    # synMag[...,5:9] representing UKIDSS 'YJHK'
    'PhotoSystems':[
      # SDSS ugriz with an error model for single-epoch imaging from the
      # Legacy footprint. A subset of the bandpasses could be specified using,
      # e.g., ('SDSS','Legacy','griz'),
      ('SDSS','Legacy'),
      # UKIDSS YJHK with an error model for the Large Area Survey
      ('UKIRT','UKIDSS_LAS'),
    ]
  },
}

# this was the 'expdust' model described in Ross+13
bossdr9_expdust_model = {
  'ContinuumParams':{
    'ContinuumModel':'GaussianPLawDistribution',
    'PowerLawSlopes':[(-0.5,0.2),1100,(-0.3,0.2),
                      5700,(-0.78,0.3),10850,(-1.81,0.3),22300,(-1.03,0.3)],
  },
  'EmissionLineParams':{
    'EmissionLineModel':'VariedEmissionLineGrid',
    'fixLineProfiles':False,
    'minEW':0.0,
    'EmLineIndependentScatter':False,
    'scaleEWs':{'LyAb':1.1,'LyAn':1.1,
                'CIVb':0.75,'CIVn':0.75,
                'CIII]b':0.8,'CIII]n':0.8,
                'MgIIb':0.8,'MgIIn':0.8},
  },
  'IronEmissionParams':{
    'FeScalings':[(0,1540,0.5),(1540,1680,2.0),(1680,1868,1.6),
                  (1868,2140,1.0),(2140,3500,1.0)],
  },
  'DustExtinctionParams':{
    'DustExtinctionModel':'Exponential E(B-V) Distribution',
    'DustModelName':'SMC',
    'E(B-V)':0.033,
    #'DustLOSfraction':1.0,
  },
}

def make_grid():
    from copy import deepcopy
    simpar = deepcopy(simParams)
    simpar['FileName'] = 'boss_grid_sim'
    simpar['GridParams']['GridType'] = 'FluxRedshiftGrid'
    del simpar['GridParams']['QLFmodel']
    del simpar['GridParams']['QLFargs']
    simpar['GridParams']['mRange'] += (30,)
    simpar['GridParams']['zRange'] += (20,)
    simpar['GridParams']['nPerBin'] = 100
    simpar['ForestParams']['NumLinesOfSight'] = 1000
    simpar['ForestParams']['FileName'] = 'boss_grid_forest'
    qsoSimulation(simpar,verbose=5)

if __name__=='__main__':
    qsoSimulation(simParams)


