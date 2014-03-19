"""
Compute FT coefficients for all LINEAR light curves from Paper III
-------------------------
"""
# Modeled after http://www.astroml.org/book_figures/chapter10/fig_LINEAR_LS.html
# License: BSD
#------------------------------------------------------------
#1.0 - Imports
#------------------------------------------------------------
import numpy as np
import Functions as func
from matplotlib import pyplot as plt
from matplotlib import colors, colorbar
from scipy.interpolate import interp1d #for interpolation -BEN
from astroML.decorators import pickle_results
from astroML.time_series import search_frequencies, lomb_scargle, MultiTermFit
from astroML.datasets import fetch_LINEAR_sample
import time
import copy

#------------------------------------------------------------
#1.1 - Data
#------------------------------------------------------------
#get start time of script for benchmarking purposes -CM
start = time.time()
# Load the light curves
data = fetch_LINEAR_sample()
# read PLV catalog
objlist = np.loadtxt('PLV_LINEAR.dat.txt')

# LINEAR ID for this star
ids = objlist[:,0]
# Lightcurve type for this star
lcType = objlist[:,1]

# its period from Paper III (Palaversa et al. 2013) 
periods = objlist[:,2] 
chi2dofArray = 0*periods
chi2RArray = 0*periods
sigmaGArray = 0*periods
nptAllArray = 0*periods
nptGoodArray = 0*periods

#open outfile for results before loop
resultsfile = 'FTcoeffs\\FTcoeffs.dat'
ofile = open(resultsfile, 'a')
#open figure for plotting before loop
#fig = plt.figure()

#------------------------------------------------------------
#1.2 - Star Fitting & Metrics
#------------------------------------------------------------
for i in range(np.size(ids)):
    #----1.2.0(data)--------------------------------------------------------
    # this star: 
    id = int(ids[i])
    lc_type = func.convert_lcType(lcType[i])
    period = periods[i]
    if period < 0:
        continue
    # get the light curve data
    # this assumes that file allDAT.tar was unpacked (>tar -xvf allDAT.tar)
    # in local subdirectory /LightCurves
    filename = 'LightCurves/{0}.dat'.format(id)
    data = np.loadtxt(filename)
    t = data[:,0]
    y = data[:,1]
    dy = data[:,2]

    #----1.2.1(fitting)--------------------------------------------------------
    # best-fit angular frequency
    omega_best = 2 * 3.14159 / period
    print "working on LINEAR ID=", id, " omega_0 = %.10g" % omega_best
    
    # do a fit to the first noFTterms Fourier components
    # for eclipsing binaries, we need at least 10 terms to avoid bias in 
    # depth of narrow minima; however, for RR Lyrae 10 terms results in 
    # too many wiggles; if we will study eclipsing binaries in detail, 
    # we will rerun with 10 terms, for now let's stick to 6 terms
    noFTterms = 6
    mtf = MultiTermFit(omega_best, noFTterms)
    mtf.fit(t, y, dy)
    phase_fit, y_fit, phased_t = mtf.predict(1000, return_phased_times=True)

    ### chi^2 calculated from direct comparison of fit to data -Ben
    # create fit data that can directly compare to datapoints -Ben
    y_fit2 = np.dot(mtf._make_X(t), mtf.w_)  
    # ZI: the number of model parameters is: 
    noMP = 2 * noFTterms + 1
    
    #----1.2.2(fit evaluation metrics)---------------------------------------------
    # ZI: we will try to get better chi2dof behavior by adding a systematic
    # error in quadrature to the random errors and clip obviously bad points
    sysErr = 0.02
    #the difference between the data and the model, normalized by the expected error.
    chi = (y - y_fit2)/np.sqrt(dy**2.0 + sysErr**2.0)
    # we will clip deviations at 4 sigma
    chimax = 4
    # same-length array as y with values set to 1
    weight = 0*y + 1
    # now flip weight to 0 if a deviant point
    weight[abs(chi)>chimax] = 0 
    # count all good points
    sizeGood = np.sum(weight) 
    nptGoodArray[i] = sizeGood
    sizeAll = np.size(y)
    nptAllArray[i] = sizeAll
    # the number of degrees of freedom
    noDOF = sizeGood - noMP
    
    # and now we compute the chi2dof only with plausibly reliable points
    if noDOF > 0: 
        chi2R = np.sum(weight*chi**2.0) / noDOF
    else: 
        # it can happen that the 4-sigma clipping was too hard (especially in 
        # case of small number of data points, if so, ignore the clipping
        chi2R = np.sum(chi**2.0) / (np.size(y) - noMP)
    chi2dof = np.sum(chi**2.0) / noDOF
    
    #find quartiles and compute the IQR
    Q25 = np.percentile(chi, 25)
    Q75 = np.percentile(chi, 75)
    IQR = (Q75 - Q25)
    #calculate sigma-G from IQR
    sigmaG = .741*IQR
    
    chi2dofArray[i] = chi2dof
    chi2RArray[i] = chi2R
    sigmaGArray[i] = sigmaG
    print i, chi2R, chi2dof, sigmaG
    
    #----1.2.3(saving data)-------------------------------------------------------- 
    #printing ID, period, chi2dof, sizeAll, sizeGood, noMP, lcTpye
    #and FFT eigencoefficients to  FTcoeffs/ID_FTcoeffs.dat for each object -CM
    FTcoeffs = [id, period, chi2R, chi2dof, sigmaG, sizeAll, sizeGood, noMP, lc_type]
    FTcoeffs.extend(mtf.w_)
    FTcoeffs = [str(i) for i in FTcoeffs]

    #ofile.write(" ".join(FTcoeffs) + "\n")
    
    #----1.2.4(lightcurve plots)--------------------------------------------------------  
    #Plot the phased lightcurves with their fit and save to FTplots/ID_FTplot.png -CM
    #func.plot_lcFit(phased_t, y, dy, phase_fit, y_fit, fig, id, omega_best)

#----1.2.5(excluding unwanted subsets)-------------------------------------------
#removing objects contained in 'other' category, or with periods less than 0.
z_period = copy.deepcopy(periods)
badZ = list(np.where(z_period < 0)[0])
badType = list(np.where(lcType == 0)[0])
#combine index lists of all unwanted objects
bad_idx = []
bad_idx.extend(badZ)
bad_idx.extend(badType)
#make numpy arrays of data without unwanted objects
lcType = np.delete(copy.deepcopy(lcType), bad_idx)
chi2dofArray = np.delete(copy.deepcopy(chi2dofArray), bad_idx)
chi2RArray = np.delete(copy.deepcopy(chi2RArray), bad_idx)
sigmaGArray = np.delete(copy.deepcopy(sigmaGArray), bad_idx)
ids = np.delete(copy.deepcopy(ids), bad_idx)

objectIdx = {}
objectIdx['RR_Lyrae_ab'] = np.where(lcType == 1)
objectIdx['RR_Lyrae_c'] = np.where(lcType == 2)
objectIdx['algol_1'] = np.where(lcType == 3)
objectIdx['algol_2'] = np.where(lcType == 4)
objectIdx['contact_bin'] = np.where(lcType == 5)
objectIdx['DSSP'] = np.where(lcType == 6)
objectIdx['LPV'] = np.where(lcType == 7)
objectIdx['heartbeat'] = np.where(lcType == 8)
objectIdx['BL_hercules'] = np.where(lcType == 9)
objectIdx['anom_ceph'] = np.where(lcType == 11)

print 'benchmarked at:', time.time() - start, 'seconds'
#------------------------------------------------------------------
#2.0 - Plotting
#------------------------------------------------------------------
##plot ratio of sigmaGArray/chi2dofArray vs. chi2RArray/chi2dofArray
tpIdx = objectIdx['contact_bin']
func.plot_ratios(chi2dofArray[tpIdx], chi2RArray[tpIdx], sigmaGArray[tpIdx], lcType[tpIdx], ids[tpIdx])

##plot 3D scatter of chi2dof, chi2R, and sigmaG
#func.plot_3DScatter(chi2dofArray, chi2RArray, sigmaGArray, lcType)

##plot histogram for each metric, quick check of distributions
#func.plot_metricDists(chi2dofArray, chi2RArray, sigmaGArray)

##plot 2D scatter-plots correlating each metric to another
#func.plot_2DScatter(chi2dofArray, chi2RArray, sigmaGArray, lcType)

##2D scatter-plots separating each lcType into it's own plot
###func.plot_lcType_scatter(sigmaGArray, chi2dofArray, lcType, 'SigmaG', 'Chi2dof')

#plot distributions for each lcType on top of each other to visualize overlap
#func.plot_metricDists_overlap(chi2dofArray, chi2RArray, sigmaGArray, lcType)

plt.show()

