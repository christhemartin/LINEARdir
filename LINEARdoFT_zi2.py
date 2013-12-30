"""
Compute FT coefficients for all LINEAR light curves from Paper III
-------------------------
"""
# Modeled after http://www.astroml.org/book_figures/chapter10/fig_LINEAR_LS.html
# License: BSD
import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d #for interpolation -BEN

from astroML.decorators import pickle_results
from astroML.time_series import search_frequencies, lomb_scargle, MultiTermFit
from astroML.datasets import fetch_LINEAR_sample
from astroML.plotting import hist
import time

#------------------------------------------------------------
#get start time of script for benchmarking purposes -CM
start = time.time()
# Load the light curves
data = fetch_LINEAR_sample()
# read PLV catalog
objlist = np.loadtxt('PLV_LINEAR.dat.txt')

# LINEAR ID for this star
ids = objlist[:,0]
# its period from Paper III (Palaversa et al. 2013) 
periods = objlist[:,2] 
chi2dofArray = 0*periods
nptAllArray = 0*periods
nptGoodArray = 0*periods

#------------------------------------------------------------
resultsfile = 'FTcoeffs.dat'
for i in range(np.size(ids)):
    # this star: 
    id = int(ids[i])
    period = periods[i]

    # get the light curve data
    # this assumes that file allDAT.tar was unpacked (>tar -xvf allDAT.tar)
    # in local subdirectory /LightCurves
    filename = 'LightCurves/{0}.dat'.format(id)
    data = np.loadtxt(filename)
    t = data[:,0]
    y = data[:,1]
    dy = data[:,2]

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
    # ZI: we will try to get better chi2dof behavior by adding a systematic
    # error in quadrature to the random errors and clip obviously bad points
    sysErr = 0.02
    chi2rat = (y - y_fit2)**2.0 / (dy**2. + sysErr**2.)
    # we will clip deviations at 4 sigma
    chi2max = 16 
    # same-length array as y with values set to 1
    weight = 0*y + 1
    # now flip weight to 0 if a deviant point
    weight[chi2rat>chi2max] = 0 
    # count all good points
    sizeGood = np.sum(weight) 
    nptGoodArray[i] = sizeGood
    sizeAll = np.size(y)
    nptAllArray[i] = sizeAll
    # the number of degrees of freedom
    noDOF = sizeGood - noMP
    # and now we compute the chi2dof only with plausibly reliable points
    if noDOF > 0: 
        chi2dof = np.sum(weight*chi2rat) / noDOF
    else: 
        # it can happen that the 4-sigma clipping was too hard (especially in 
        # case of small number of data points, if so, ignore the clipping
        chi2dof = np.sum(chi2rat) / (np.size(y) - noMP)

    chi2dofArray[i] = chi2dof
    print i, chi2dof
    
    FTcoeffs = [id, period, chi2dof, sizeAll, sizeGood, noMP]
    FTcoeffs.extend(mtf.w_)
    FTcoeffs = [str(i) for i in FTcoeffs]
    ofile = open(resultsfile, 'a')
    ofile.write(" ".join(FTcoeffs) + "\n")
    
print 'it took:', time.time() - start, 'seconds'
    ## ZI: here we should add code to save into a file (say, FTcoeffs/ID_FTcoeffs.dat) 
    ##      ID, period, chi2dof, sizeAll, sizeGood, noMP, and FFT eigencoefficients
    ## it would be nice to add a plot of the best-fit and data like we had before
    ## but instead of 6 panels, each star gets its own plot (say, FTplots/ID_FTplot.png) 
    ## for example, then we can search for all stars with chi2dof > 6 and look at
    ## their best fits



# quick plot here of chi2dof distribution for sanity check 
ax = plt.axes()
hist(chi2dofArray, bins='freedman', ax=ax, histtype='stepfilled', ec='k', fc='#AAAAAA')
ax.set_xlim(0, 10)
ax.set_xlabel('chi2dof')
ax.set_ylabel('dN/dchi2dof')
plt.show()