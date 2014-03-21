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

#----1.1(RFP)-------------------------------
RFP = func.ComputeRFP(allData)#compute the relative fourier parameters
func.plotRFP(RFP,allData['lcType'])
#func.plotRFP_vs_P(RFP, allData['period'], lcType)

#----1.1(GMM)-------------------------------
#X = np.array([RFP['R31'],RFP['Phi21']]).T
#func.gaussian_mixture_model(X)#unsupervised learning via gaussian mixture models

#----1.1(KNN)-------------------------------
#X = np.array([allData['period'],np.log(RFP['R21'])]).T
#y = [func.convert_lcType(i) for i in lcType]
#func.KNN(X, y)#supervised learning via K-nearest neighbors