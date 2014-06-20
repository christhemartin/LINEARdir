#===============================================================================
#0.0 - Imports
#===============================================================================
import numpy as np
import pylab as plt
import Functions as func
from astroML.datasets import fetch_LINEAR_sample
from astroML.datasets import fetch_LINEAR_geneva
#===============================================================================
# 1.0 - Main
#===============================================================================
#----1.1(data)------------------------------
# Get the Geneva periods data
g_data = fetch_LINEAR_geneva()
#pull in all data from FTCoeffs.dat
coeff_data = func.read_data()
all_data = func.merge_data(coeff_data, g_data)

#----1.2(RFP)-------------------------------
RFP = func.ComputeRFP(all_data)#compute the relative fourier parameters
#func.plot_a(all_data, RFP)
#func.plotRFP(RFP, all_data)
#func.plotRFP_vs_P(RFP, all_data)

#func.plot_types(all_data['period'],RFP['R21'],allData['lcType'])
#----1.3(GMM)-------------------------------
#X = np.array([RFP['R31'],RFP['Phi21']]).T
#func.gaussian_mixture_model(X)#unsupervised learning via gaussian mixture models

#----1.4(KNN)-------------------------------
#all_data['period'],all_data['gi'],all_data['ra'],all_data['dec'], 
#all_data['amp'],all_data['ug'], all_data['iK'], all_data['JK'],
#all_data['skew'],all_data['kurt'],RFP['R21'],RFP['R41']
X = np.array([all_data['period'],RFP['R21'],all_data['gi'],all_data['ra'],all_data['dec'],all_data['amp'],all_data['ug'], all_data['iK'], all_data['JK'],all_data['skew'],all_data['kurt'],RFP['R41']]).T 
y = all_data['lcType']
func.KNN(X, y)#supervised learning via K-nearest neighbors
func.Trees(X, y)#supervised learning via decision trees