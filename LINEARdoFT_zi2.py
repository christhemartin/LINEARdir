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
    chi2max = 16
    # same-length array as y with values set to 1
    weight = 0*y + 1
    # now flip weight to 0 if a deviant point
    weight[chi>chi2max] = 0 
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
    #printing ID, period, chi2dof, sizeAll, sizeGood, noMP, 
    #and FFT eigencoefficients to  FTcoeffs/ID_FTcoeffs.dat for each object -CM
    FTcoeffs = [id, period, chi2R, chi2dof, sigmaG, sizeAll, sizeGood, noMP]
    FTcoeffs.extend(mtf.w_)
    FTcoeffs = [str(i) for i in FTcoeffs]

    #ofile.write(" ".join(FTcoeffs) + "\n")
    
    #----1.2.4(lightcurve plots)--------------------------------------------------------  
    #Plot the phased lightcurves with their fit and save to FTplots/ID_FTplot.png -CM
    #func.plot_lcFit(phased_t, y, dy, phase_fit, y_fit, fig, id, omega_best)

print 'benchmarked at:', time.time() - start, 'seconds'


#------------------------------------------------------------------
#2.0 - Plotting
#------------------------------------------------------------------
##plot 3D scatter of chi2dof, chi2R, and sigmaG
func.plot_3DScatter(chi2dofArray, chi2RArray, sigmaGArray, lcType)

##plot histogram for each metric, quick check of distributions
func.plot_metricDists(chi2dofArray, chi2RArray, sigmaGArray)

##plot 2D scatter-plots correlating each metric to another
func.plot_2DScatter(chi2dofArray, chi2RArray, sigmaGArray, lcType)

##2D scatter-plots separating each lcType into it's own plot
func.plot_lcType_scatter(sigmaGArray, chi2dofArray, lcType, 'SigmaG', 'Chi2dof')

#plot distributions for each lcType on top of each other to visualize overlap
func.plot_metricDists_overlap(chi2dofArray, chi2RArray, sigmaGArray, lcType)

plt.show()

