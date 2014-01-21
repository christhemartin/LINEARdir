'''
contains auxiliary functions for pltthon scripts in LINEARdir
'''
import numpy as np
import os
import pylab as plt
import shutil
from astroML.plotting import hist
from astroML.time_series import search_frequencies, lomb_scargle, MultiTermFit

def read_data():
    '''
    input: n/a
    function: 
    output: 
    '''
    data = {'id':[], 'period':[], 'chi2dof':[], 'sizeAll':[], 'sizeGood':[], 'noMP':[], 'coefficients':[]}
    od = os.getcwd()
    if od.endswith('\\Scripts'):
        os.chdir('..\\FTcoeffs')
    else:
        os.chdir('\\FTcoeffs')
        
    wd = os.getcwd()
    f = open('iterFTcoeffs.dat', 'r')
    lines = f.readlines()
    temp_data = [line.split() for line in lines]
#    temp_data = np.loadtxt('iterFTcoeffs.dat')
    print len(temp_data)
    data['id'] = [float(i[0]) for i in temp_data]
    data['period'] = [float(i[1]) for i in temp_data]
    data['chi2R'] = [float(i[2]) for i in temp_data]
    data['chi2dof'] = [float(i[3]) for i in temp_data]
    data['sigmaG'] = [float(i[4]) for i in temp_data]
    data['sizeAll'] = [float(i[5]) for i in temp_data]
    data['sizeGood'] = [float(i[6]) for i in temp_data]
    data['noMP'] = [float(i[7]) for i in temp_data]
    data['coefficients'] = [map(float, i[8:]) for i in temp_data]
    os.chdir(od)
    return data

def bad_plots(badObjs):
    '''
    input: IDs of objects with abnormally high chi2dof
    function: to find and coplt plots for all bad objects into a separate directory for viewing
    output: n/a
    '''
    od = os.getcwd()
    if od.endswith('\\Scripts'):
        os.chdir('..\\FTplots')
    else:
        os.chdir('\\FTplots')
    for ID in badObjs:
        shutil.coplt(str(ID) + '_FTplot.png', '..\\Bad_Fit_Plots')
        
def correlateCoeffs(data):
    od = os.getcwd()
    if od.endswith('\\Scripts'):
        os.chdir('..')
    objlist = np.loadtxt('PLV_LINEAR.dat.txt')
    ids = objlist[:,0]
    lcType = objlist[:,1]
    
    dataIds = data['id']
    coeffs = data['coefficients']

    positions = [dataIds.index(id) for id in ids]
    coeffs = [coeffs[i] for i in positions]
    coeffsDict = {}
    for paramNum in range(len(coeffs)):
        tempKey = 'P' + str(paramNum)
    coeffsDict[tempKey] = [i[0] for i in coeffs]

    for key in coeffsDict.keys():
        print key, np.median(coeffsDict[key])
        ax = plt.figure()
        plt.title(key)
        #plt.plot(lcType, coeffsDict[key], '.b')
        hist(coeffsDict[key], bins='freedman')
        ax.ylim = [-5,15]
    plt.show()

def iterateFit(i, omega_best, t, y, dy, nptGoodArray, chi2dofArray, nptAllArray, chi2dofOld):
    '''
    depracated side project to iterate fit across numper of model parameters until chi2 is brought under threshold of 6
    '''
    print "############### - Re-fitting - ################"
    chi2dof = highest = chi2dofOld
    noFTterms = 6
    operation = True
    while chi2dof > 6:
        print noFTterms, chi2dof, chi2dofOld
        if chi2dof > highest:
            highest = chi2dof
            operation = not operation 
        if operation == True:
            noFTterms +=2
        if operation == False:
            noFTterms -=2
        
        chi2dofOld = chi2dof
        
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
    
    print noFTterms, chi2dof, chi2dofOld
    print "############### - End of iterations - ################"
    return noFTterms, chi2dof, mtf
