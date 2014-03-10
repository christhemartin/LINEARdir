import numpy as np
import pylab as py
import Functions as func
import itertools
from astroML.plotting import hist

allData = func.read_data()
#how are coefficients correlated to lcType?
func.correlateCoeffs(allData)