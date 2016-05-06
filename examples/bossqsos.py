#!/usr/bin/env python

from astropy.cosmology import FlatLambdaCDM
from simqso import qsoSimulation,lumfun

def BOSS_DR9_PLE(which=1):
	if which==1:
		row = -1.16,-3.37,-22.85,1.241,-0.249,-5.96
	alpha,beta,MStar0,k1,k2,logPhiStar = row
	MStar = lambda z: MStar0 - 2.5*(k1*z + k2*z**2)
	return lumfun.DoublePowerLawLF(logPhiStar,MStar,alpha,beta)

def BOSS_DR9_LEDE():
	c1,c2 = -0.689, -0.809
	logPhiStar22 = -5.83
	MStar22 = -26.49
	logPhiStar = lambda z: logPhiStar22 + c1*(z-2.2)
	MStar = lambda z: MStar22 + c2*(z-2.2)
	alpha = -1.31
	beta = -3.45
	return lumfun.DoublePowerLawLF(logPhiStar,MStar,alpha,beta)

simParams = {
  # filename for simulation output (".fits" is appended)
  'FileName':'boss_dr9qlf_sim',
  # wavelength range of simulated spectra in Angstroms
  'waveRange':(3000.,3.0e4),
  # constant spectral dispersion per pixel as R = d(log(lam))
  'SpecDispersion':500,
  # dispersion scale is logarithmic [only option for now]
  'DispersionScale':'logarithmic',
  # set the cosmology, any appropriate instance from astropy.cosmology allowed
  'Cosmology':FlatLambdaCDM(70,1-0.7,name='BOSSDR9'),
  # setting a global random seed allows the simulation to be repeatable
  'RandomSeed':1,
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
    'QLFargs':{'skyArea':100}, # but only 100 deg2 will run a lot faster
    # set bright and faint flux limits
    'mRange':(17.0,22.2),
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
      'ContinuumModel':'GaussianPLawDistribution',
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
    'FileName':'boss_dr9qlf_forest',
    # name of the model for the distribution of absorbers [only WP11 for now]
    'ForestModel':'Worseck&Prochaska2011', 
    # redshift range over which to sample absorbers
    'zRange':(0.0,4.5),
    # type of forest grid to use; 'Sightlines' means generate N independent
    # sightlines, and then randomly associate each quasar with a sightline
    'ForestType':'Sightlines',
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

if __name__=='__main__':
	qsoSimulation(simParams)


