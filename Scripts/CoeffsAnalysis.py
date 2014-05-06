#===============================================================================
#0.0 - Imports
#===============================================================================
import numpy as np
import pylab as plt
import Functions as func

#===============================================================================
# 1.0 - Main
#===============================================================================
#----1.1(data)------------------------------
allData = func.read_data()#pull in all data from FTCoeffs.dat

#----1.2(RFP)-------------------------------
RFP = func.ComputeRFP(allData)#compute the relative fourier parameters
#func.plot_a(allData, RFP)
#func.plotRFP(RFP, allData)
#func.plotRFP_vs_P(RFP, allData)

#func.plot_types(allData['period'],RFP['R21'],allData['lcType'])
#----1.3(GMM)-------------------------------
#X = np.array([RFP['R31'],RFP['Phi21']]).T
#func.gaussian_mixture_model(X)#unsupervised learning via gaussian mixture models

#----1.4(KNN)-------------------------------
X = np.array([allData['period'],RFP['R21'],RFP['R41']]).T
y = [func.convert_lcType(i) for i in allData['lcType']]
func.KNN(X, y)#supervised learning via K-nearest neighbors