#!/usr/bin/env python

import os
import numpy as np
from .sqrun import readSimulationData

def calcKCorrFromFluxGrid(fileName,outputDir='./'):
	simdat,simpars = readSimulationData(fileName,outputDir,retParams=True)
	mEdges = np.arange(simpars['mRange'])
	zEdges = np.arange(simpars['zRange'])

