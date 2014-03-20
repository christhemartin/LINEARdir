'''
contains auxiliary functions for python scripts in LINEARdir
'''
import numpy as np
import os
import pylab as plt
import shutil
from astroML.plotting import hist
from astroML.time_series import search_frequencies, lomb_scargle, MultiTermFit
from astroML.plotting.tools import discretize_cmap
from astroML.plotting.tools import discretize_cmap
from mpl_toolkits.mplot3d import axes3d, Axes3D
from matplotlib import colors, colorbar
from sklearn.mixture import GMM
from astroML.datasets import fetch_sdss_sspp
from astroML.plotting.tools import draw_ellipse

def read_data():
    '''
    input: n/a
    function: 
    output: 
    '''
    data = {'id':[], 'period':[], 'chi2dof':[], 'chi2R':[], 'sigmaG':[], 'sizeAll':[], 'sizeGood':[], 'noMP':[], 'lcType':[],  'coefficients':[]}
    od = os.getcwd()
    if od.endswith('\\Scripts'):
        os.chdir('..\\FTcoeffs')
    else:
        os.chdir('\\FTcoeffs')
        
    wd = os.getcwd()
    f = open('FTcoeffs.dat', 'r')
    lines = f.readlines()
    temp_data = [line.split() for line in lines]

#    temp_data = np.loadtxt('iterFTcoeffs.dat')
    data['id'] = [float(i[0]) for i in temp_data]
    data['period'] = [float(i[1]) for i in temp_data]
    data['chi2R'] = [float(i[2]) for i in temp_data]
    data['chi2dof'] = [float(i[3]) for i in temp_data]
    data['sigmaG'] = [float(i[4]) for i in temp_data]
    data['sizeAll'] = [float(i[5]) for i in temp_data]
    data['sizeGood'] = [float(i[6]) for i in temp_data]
    data['noMP'] = [float(i[7]) for i in temp_data]
    data['lcType'] = [str(i[8]) for i in temp_data]
    data['coefficients'] = [map(float, i[9:]) for i in temp_data]
    os.chdir(od)
    return data

def save_plots(badObjs, path = ''):
    '''
    input: IDs of objects 
    function: Find objects, copy individual plots, and place them into a separate directory for later viewing
    output: n/a
    '''
    od = os.getcwd()
    print os.listdir(od)
    if od.endswith('\\Scripts'):
        os.chdir('..\\FTplots')
    else:
        os.chdir('FTplots')
    for ID in badObjs:
        id = int(ID)
        shutil.copy(str(id) + '_FTplot.png', '..\\Regions_of_Interest\\' + path)
    os.chdir(od)
        
def convert_lcType(object):
    '''
    input: number from 0 - 11 or string of object type, (should exclude 10 as an artifact of PLV_LINEAR.dat! explains placeholder entry for i = 10)
    function: converts between corresponding ints and strings pr. lcType
    output: lightcurve type, opposite the input
    '''
    if isinstance(object, float):
        types = ['Other','RR_Lyrae_ab','RR_Lyrae_c','algol_1','algol_2','contact_bin','DSSP','LPV','heartbeat','BL_hercules', 'listed_as_type_10','anom_ceph']
        lcType = types[int(object)]
    if isinstance(object,str):
        types = {'Other': 0 , 'RR_Lyrae_ab': 1,'RR_Lyrae_c': 2,'algol_1': 3,'algol_2': 4,'contact_bin': 5,'DSSP': 6,'LPV': 7,'heartbeat': 8,\
                 'BL_hercules': 9, 'listed_as_type_10': 10,'anom_ceph': 11}
        lcType = types[object]
    return lcType

def correlateCoeffs(data):
    '''
    input: dictionary of data read from coefficients.dat file
    function: read in data on lcType and match to objects, plotting lcType against model parameters
    output: n/a
    '''
    od = os.getcwd()
    if od.endswith('\\Scripts'):
        os.chdir('..')
    
    dataIds = data['id']
    coeffs = np.array(data['coefficients'])
    lcType = data['lcType']
    
    objectIdx = {}
    objectIdx['RR_Lyrae_ab'] = [i for i in range(len(lcType)) if lcType[i] == 'RR_Lyrae_ab']
    objectIdx['RR_Lyrae_c'] = [i for i in range(len(lcType)) if lcType[i] == 'RR_Lyrae_c']
    objectIdx['algol_1'] = [i for i in range(len(lcType)) if lcType[i] == 'algol_1']
    objectIdx['algol_2'] = [i for i in range(len(lcType)) if lcType[i] == 'algol_2']
    objectIdx['contact_bin'] = [i for i in range(len(lcType)) if lcType[i] == 'contact_bin']
    objectIdx['DSSP'] = [i for i in range(len(lcType)) if lcType[i] == 'DSSP']
    objectIdx['LPV'] = [i for i in range(len(lcType)) if lcType[i] == 'LPV']
    objectIdx['heartbeat'] = [i for i in range(len(lcType)) if lcType[i] == 'heartbeat']
    objectIdx['BL_hercules'] = [i for i in range(len(lcType)) if lcType[i] == 'BL_hercules']
    objectIdx['anom_ceph'] = [i for i in range(len(lcType)) if lcType[i] == 'anom_ceph']
    
    
    #plt.suptitle('Relative Fourier Parameters')
    cm = plt.get_cmap('jet')
    count = 0
    for key in objectIdx.keys():
        
        tpIdx = objectIdx[key]
        coeffs1 = coeffs[tpIdx]
        
        A1 = np.array([np.sqrt(object[7]**2.0 + object[1]**2.0) for object in coeffs1])
        A2 = np.array([np.sqrt(object[8]**2.0 + object[2]**2.0) for object in coeffs1])
        A3 = np.array([np.sqrt(object[9]**2.0 + object[3]**2.0) for object in coeffs1])
        Phi1 = np.array([np.arctan(-object[1]/object[7]) for object in coeffs1])
        Phi2 = np.array([np.arctan(-object[2]/object[8]) for object in coeffs1])
        Phi3 = np.array([np.arctan(-object[3]/object[9]) for object in coeffs1])
        

    #    plt.title(key)
        R21 = A2/A1
        R31 = A3/A1
        Phi21 = Phi2 - 2.0*Phi1
        Phi31 = Phi3 - 3.0*Phi3
        
        #testing GMM
        if key =='RR_Lyrae_c':
            X = np.array([R31,Phi21]).T
            gaussian_mixture_model(X)
        
        fig = plt.figure()
        plt.suptitle('Relative Fourier Parameters - ' + key)
        color = cm(1.*float(count)/len(objectIdx.keys()))
    
        ax1 = fig.add_subplot(3,2,1)
        plt.scatter(R21, R31, c = color, marker ='o', alpha = .7, label = key)
        plt.title('R21 vs R31')
        plt.xlabel('R21')
        plt.ylabel('R31')
        #legend1 = ax1.legend(loc='right', ncol=2, shadow=True)
        
        fig.add_subplot(3,2,2)
        plt.scatter(R21, Phi21, c = color, marker ='o', alpha = .7)
        plt.title('R21 vs. Phi21')
        plt.xlabel('R21')
        plt.ylabel('Phi21')
        
        fig.add_subplot(3,2,3)
        plt.scatter(R21, Phi31, c = color, marker ='o', alpha = .7)
        plt.title('R21 vs. Phi31')
        plt.xlabel('R21')
        plt.ylabel('Phi31')
        
        fig.add_subplot(3,2,4)
        plt.scatter(R31, Phi21, c = color, marker ='o', alpha = .7)
        plt.title('R31 vs. Phi21')
        plt.xlabel('R31')
        plt.ylabel('Phi21')
        
        fig.add_subplot(3,2,5)
        plt.scatter(R31, Phi31, c = color, marker ='o', alpha = .7)
        plt.title('R31 vs Phi31')
        plt.xlabel('R31')
        plt.ylabel('Phi31')
        
        fig.add_subplot(3,2,6)
        plt.scatter(Phi21, Phi31, c = color, marker ='o', alpha = .7)
        plt.title('Phi21 vs Phi31')
        plt.xlabel('Phi21')
        plt.ylabel('Phi31')
        count += 1
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

def plot_lcFit(phased_t, y, dy, phase_fit, y_fit, fig, id, omega_best):
    '''
    input: Array of times corresponding to phase folded flux data(phased_t), flux data from star (y), error in flux (dy), 
            time array corresponding to phase folded fit data (phase_fit), fit data (y_fit).
    function: Create a plot of an individual star's lightcurve with errorbars and our fit, phase folded to best fir period. 
    output: n/a
    '''
    ax = fig.add_subplot(111)
    ax.errorbar(phased_t, y, dy, fmt='.k', ecolor='gray', 
                lw=1, ms=4, capsize=1.5)
    ax.plot(phase_fit, y_fit, '-b', lw=2, alpha = .7)
    
    
    ax.set_xlim(0, 1)
    ax.set_ylim(plt.ylim()[::-1])
    ax.yaxis.set_major_locator(plt.MaxNLocator(4))
    ax.grid(True, which='major', linestyle='--', color = "#a6a6a6", 
            zorder=2, alpha = .8)
    ax.text(0.97, 0.96, "ID = %i" % id, ha='right', 
            va='top',transform=ax.transAxes)
    ax.text(0.03, 0.96, "P = %.2f hr" % (2 * np.pi / omega_best * 24.), 
            ha='left', va='top', transform=ax.transAxes)

    ylim = ax.get_ylim()
    ax.set_ylim(ylim[0], ylim[0] + 1.1 * (ylim[1] - ylim[0]))
    ax.set_ylabel('mag')
    ax.set_xlabel('phase')
    
    #save figure to FTplots directory
    plt.savefig('FTplots\\' + str(id) + '_FTplot.png')
    #clear figure for next object
    plt.clf()

def plot_lcType_scatter(xData, yData, lcType, xName, yName):
    '''
    input: Arrays to be plotted against each other (xData,yData), usually chi2dof/chi2R/sigmaG. Strings of their names to identify plots (xName,yName).
    function: To plot separately, different lcTypes for one metric vs. another. 
    output: n/a
    '''
    fig = plt.figure()
    fig.suptitle(xName + ' Vs. ' + yName + ' across lcTypes')
    # Set up color-map properties
    clim = (1.5, 6.5)
    cmap = discretize_cmap(plt.cm.jet, 15)
    cdict = ['Other', 'RR Lyrae a&b', 'RR Lyrae C', 'Algol-like w/ 1 minimum', 'Algol-like w/ 2 minima', 'Contact Binary', 'Delta Scu/Sx Phe', 'Long Per. Var.', 'Heartbeat Candidates', 'BL Hercules', 'Anomalous Cepheids']
    cticks = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    formatter = plt.FuncFormatter(lambda t, *args: cdict[int(np.round(t))])
    # Create scatter-plots
    scatter_kwargs = dict(s=4, lw=0, edgecolors='none', cmap=cmap, alpha = .7)
    other = np.where(lcType == 0)
    RR_Lyrae_ab = np.where(lcType == 1)
    RR_Lyrae_c = np.where(lcType == 2)
    algol_1 = np.where(lcType == 3)
    algol_2 = np.where(lcType == 4)
    contact_bin = np.where(lcType == 5)
    DSSP = np.where(lcType == 6)
    LPV = np.where(lcType == 7)
    heartbeat = np.where(lcType == 8)
    BL_hercules = np.where(lcType == 9)
    anom_ceph = np.where(lcType == 11)
    
    axa = fig.add_subplot(4,3,1)
    ima = axa.scatter(np.log(xData[other]), np.log(yData[other]), **scatter_kwargs)
    axa.set_title('Other')
    
    axb = fig.add_subplot(4,3,2,sharex=axa,sharey=axa)
    imb = axb.scatter(np.log(xData[RR_Lyrae_ab]), np.log(yData[RR_Lyrae_ab]), **scatter_kwargs)
    axb.set_title('RR Lyrae a&b')
    
    axc = fig.add_subplot(4,3,4,sharex=axa,sharey=axa)
    imc = axc.scatter(np.log(xData[RR_Lyrae_c]), np.log(yData[RR_Lyrae_c]), **scatter_kwargs)
    axc.set_title('RR Lyrae c')
    
    axd = fig.add_subplot(4,3,5,sharex=axa,sharey=axa)
    imd = axd.scatter(np.log(xData[algol_1]), np.log(yData[algol_1]), **scatter_kwargs)
    axd.set_title('Algol like with 1 minimum')
    
    axe = fig.add_subplot(4,3,6,sharex=axa,sharey=axa)
    ime = axe.scatter(np.log(xData[algol_2]), np.log(yData[algol_2]), **scatter_kwargs)
    axe.set_title('Algol like with 2 minima')
    
    axf = fig.add_subplot(4,3,7,sharex=axa,sharey=axa)
    imf = axf.scatter(np.log(xData[contact_bin]), np.log(yData[contact_bin]), **scatter_kwargs)
    axf.set_title('Contact Binary')
    
    axg = fig.add_subplot(4,3,8,sharex=axa,sharey=axa)
    img = axg.scatter(np.log(xData[DSSP]), np.log(yData[DSSP]), **scatter_kwargs)
    axg.set_title('Delta Scu/Sx Phe')
    
    axh = fig.add_subplot(4,3,9,sharex=axa,sharey=axa)
    imh = axh.scatter(np.log(xData[LPV]), np.log(yData[LPV]), **scatter_kwargs)
    axh.set_title('Long Period Variable')
    
    axi = fig.add_subplot(4,3,10,sharex=axa,sharey=axa)
    imi = axi.scatter(np.log(xData[heartbeat]), np.log(yData[heartbeat]), **scatter_kwargs)
    axi.set_title('Heartbeat Candidate')
    
    axj = fig.add_subplot(4,3,11,sharex=axa,sharey=axa)
    imj = axj.scatter(np.log(xData[BL_hercules]), np.log(yData[BL_hercules]), **scatter_kwargs)
    axj.set_title('BL Hercules')
    
    axk = fig.add_subplot(4,3,12,sharey=axa)
    imk = axk.scatter(np.log(xData[anom_ceph]), np.log(yData[anom_ceph]), **scatter_kwargs)
    axk.set_title('Anomalous Cepheid')

def plot_2DScatter(chi2dofArray, chi2RArray, sigmaGArray, lcType):
    '''
    input: Arrays of the 3 metrics, chi2dof, chi2R and sigmaG, as well as the lcTypes of all objects
    function: Plot each metric vs. another in a 2D scatter with colors representing lcType
    output: n/a
    '''
    fig = plt.figure()
    # Set up color-map properties
    clim = (1.5, 6.5)
    cmap = discretize_cmap(plt.cm.jet, 15)
    cdict = ['Other', 'RR Lyrae a&b', 'RR Lyrae C', 'Algol-like w/ 1 minimum', 'Algol-like w/ 2 minima', 'Contact Binary', 'Delta Scu/Sx Phe', 'Long Per. Var.', 'Heartbeat Candidates', 'BL Hercules', 'Anomalous Cepheids']
    cticks = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    formatter = plt.FuncFormatter(lambda t, *args: cdict[int(np.round(t))])
    # Create scatter-plots
    scatter_kwargs = dict(s=4, lw=0, edgecolors='none', c=lcType, cmap=cmap, alpha = .7)
    
    ax1 = fig.add_subplot(2,2,1)
    im1 = ax1.scatter(np.log(chi2dofArray), np.log(chi2RArray), **scatter_kwargs)
    im1.set_clim(clim)
    ax1.set_xlabel('chi2dof')
    ax1.set_ylabel('chi2R')
    ax1.set_title('log(chi2dof) vs. log(chi2R)')
    
    ax2 = fig.add_subplot(2,2,3)
    im2 = ax2.scatter(np.log(sigmaGArray), np.log(chi2dofArray), **scatter_kwargs)
    im2.set_clim(clim)
    ax2.set_ylabel('chi2dof')
    ax2.set_xlabel('sigmaG')
    ax2.set_title('log(chi2dof) vs. log(sigmaG)')
    
    ax3 = fig.add_subplot(2,2,4)
    im3 = ax3.scatter(np.log(sigmaGArray), np.log(chi2RArray), **scatter_kwargs)
    im3.set_clim(clim)
    ax3.set_ylabel('chi2R')
    ax3.set_xlabel('sigmaG')
    ax3.set_title('log(chi2R) vs. log(sigmaG)')
    
    cax = plt.axes([0.525, 0.525, 0.02, 0.35])
    fig.colorbar(im3, ax=ax3, cax=cax,
                     ticks=cticks,
                     format=formatter)
    
def plot_3DScatter(chi2dofArray, chi2RArray, sigmaGArray, lcType):
    '''
    input: Arrays of the 3 metrics, chi2dof, chi2R and sigmaG, as well as the lcTypes of all objects
    function: Create a 3D scatter plot of the metric space, coloring points according to lcType
    output: n/a
    '''
    fig = plt.figure()
    fig.suptitle('Metric Space for chi2dof, chi2R & sigmaG')
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(np.log(chi2dofArray), np.log(chi2RArray), np.log(sigmaGArray), c=lcType, cmap = plt.cm.jet, alpha = .7)
    ax.set_xlabel('chi2dof')
    ax.set_ylabel('chi2R')
    ax.set_zlabel('sigmaG')
    
def plot_metricDists(chi2dofArray, chi2RArray, sigmaGArray):
    '''
    input: Arrays of the 3 metrics, chi2dof, chi2R and sigmaG
    function: Create a histogram of the distribution for each metric
    output: n/a
    '''
    fig = plt.figure()
    ax1 = fig.add_subplot(3,1,1)
    hist(chi2dofArray, bins='freedman', ax=ax1, histtype='stepfilled', ec='k', fc='#AAAAAA')
    ax1.set_xlim(0, 10)
    ax1.set_xlabel('chi2dof')
    ax1.set_ylabel('dN/dchi2dof')
    ax2 = fig.add_subplot(3,1,2)
    hist(chi2RArray, bins='freedman', ax=ax2, histtype='stepfilled', ec='k', fc='#AAAAAA')
    ax2.set_xlim(0, 10)
    ax2.set_xlabel('chi2R')
    ax2.set_ylabel('dN/dchi2R')
    ax3 = fig.add_subplot(3,1,3)
    hist(sigmaGArray, bins='freedman', ax=ax3, histtype='stepfilled', ec='k', fc='#AAAAAA')
    ax3.set_xlim(0, 10)
    ax3.set_xlabel('sigmaG')
    ax3.set_ylabel('dN/dsigmaG')
     
def plot_metricDists_overlap(chi2dofArray, chi2RArray, sigmaGArray, lcType):
    '''
    input: Arrays of the 3 metrics, chi2dof, chi2R and sigmaG, as well as the lcTypes of all objects
    function: Create a separate histogram for every lcType within each metric's distribution. (to see overlap of each distribution)
    output: n/a
    '''
    objectIdx = {}
    objectIdx['other'] = np.where(lcType == 0)
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
    
    fig = plt.figure()
    ax1 = fig.add_subplot(3,1,1)
    for key in objectIdx.keys():
        tempIdx = objectIdx[key][0]
        if len(tempIdx) > 3:
            hist(chi2dofArray[tempIdx], bins='freedman', ax=ax1, histtype='stepfilled', ec='k', label = key,  alpha = .6)
            print key, len(chi2dofArray[tempIdx])
            if key == 'other':
                specialFig = plt.figure()
                specialFig.add_subplot(1,1,1)
                plt.hist(chi2dofArray[tempIdx], bins = 30)
    ax1.set_xlim(0, 10)
    ax1.set_xlabel('chi2dof')
    ax1.set_ylabel('dN/dchi2dof')
    legend1 = ax1.legend(loc='right', ncol=2, shadow=True)
    
    ax2 = fig.add_subplot(3,1,2)
    for key in objectIdx.keys():
        tempIdx = objectIdx[key][0]
        if len(tempIdx) > 3:
            hist(chi2RArray[tempIdx], bins='freedman', ax=ax2, histtype='stepfilled', ec='k', alpha = .6)
    ax2.set_xlim(0, 10)
    ax2.set_xlabel('chi2R')
    ax2.set_ylabel('dN/dchi2R')
    
    
    ax3 = fig.add_subplot(3,1,3)
    for key in objectIdx.keys():
        tempIdx = list(objectIdx[key][0])
        if len(tempIdx) > 3:
            hist(sigmaGArray[tempIdx], bins='freedman', ax=ax3, histtype='stepfilled', ec='k', alpha = .6)
    ax3.set_xlim(0, 10)
    ax3.set_xlabel('sigmaG')
    ax3.set_ylabel('dN/dsigmaG')

def plot_ratios(chi2dofArray, chi2RArray, sigmaGArray, lcType, ids):
    '''
    input: Arrays of the 3 metrics, chi2dof, chi2R and sigmaG, as well as the lcTypes of all objects
    function: Plot ratio of sigmaGArray/chi2dofArray vs. chi2RArray/chi2dofArray
    output: n/a
    '''
    x = np.array(sigmaGArray/chi2RArray)
    y = np.array(chi2RArray/chi2dofArray)
    regions = {}
    regions['R1'] = np.where(y > .99)[0]
    regions['R2'] = [i for i in range(len(x)) if (x[i] > 1) and (y[i] < .6)] 
    regions['R3'] = [i for i in range(len(x)) if (x[i] > 1) and (.6 < y[i] < .99)]
    regions['R4'] = [i for i in range(len(x)) if (x[i] < 1) and (0 < y[i] < .2)]
    regions['R5'] = [i for i in range(len(x)) if (x[i] < 1) and (.2 < y[i] < .4)]
    regions['R6'] = [i for i in range(len(x)) if (x[i] < 1) and (.4 < y[i] < .6)]
    regions['R7'] = [i for i in range(len(x)) if (x[i] < 1) and (.6 < y[i] < .8)]
    regions['R8'] = [i for i in range(len(x)) if (x[i] < 1) and (.8 < y[i] < .99)]
    
    #option to save plots of objects by type for each sub region
#    for key in regions.keys():
#        save_plots(ids[regions[key]], path = 'Contact_Bin\\' + key)
    
    
    cm = plt.get_cmap('jet')
    fig = plt.figure()
    count = 0
    ax1 = fig.add_subplot(1,1,1)
    for key in regions.keys():
        color = cm(1.*float(count)/len(regions.keys()))
        plt.scatter(x[regions[key]], y[regions[key]], c = color, marker ='o', label = key, alpha = .7)
        count += 1
    plt.xlabel('sigmaG/chi2R')
    plt.ylabel('chi2R/chi2dof')
    plt.grid(True, which='major', linestyle='--', color = "#a6a6a6", 
             zorder=2, alpha = .8)
    plt.title('sigmaG/chi2R vs. chi2R/chi2dof')
    legend1 = ax1.legend(loc='right', ncol=2, shadow=True)
    
    
    objectIdx = {}
    objectIdx['other'] = np.where(lcType == 0)
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
    
    fig = plt.figure()
    cmap = discretize_cmap(plt.cm.jet, 15)
    count = 0
    for key in objectIdx.keys():
        tempIdx = objectIdx[key][0]
        if len(tempIdx) > 3:
            count += 1
            ratio1 = sigmaGArray[tempIdx]/chi2RArray[tempIdx]
            ratio2 = chi2RArray[tempIdx]/chi2dofArray[tempIdx]
            fig.add_subplot(3,3,count)
            plt.scatter(ratio1,ratio2, marker ='o', alpha = .7)
            plt.xlabel('sigmaG/chi2R')
            plt.ylabel('chi2R/chi2dof')
            plt.grid(True, which='major', linestyle='--', color = "#a6a6a6", 
                     zorder=2, alpha = .8)
            plt.title(str(key) + ' population: ' + str(len(tempIdx)))
  
def gaussian_mixture_model(X):  
    #------------------------------------------------------------
    # Compute GMM models & AIC/BIC
    N = np.arange(1, 14)
    
    def compute_GMM(N, covariance_type='full', n_iter=1000):
        models = [None for n in N]
        for i in range(len(N)):
            print N[i]
            models[i] = GMM(n_components=N[i], n_iter=n_iter,
                            covariance_type=covariance_type)
            models[i].fit(X)
        return models
    
    models = compute_GMM(N)
    
    AIC = [m.aic(X) for m in models]
    BIC = [m.bic(X) for m in models]
    
    i_best = np.argmin(BIC)
    gmm_best = models[i_best]
    print "best fit converged:", gmm_best.converged_
    print "BIC: n_components =  %i" % N[i_best]
    
    
    x_flat = np.r_[X[:,0].min():X[:,0].max():128j]
    
    #------------------------------------------------------------
    # Plot the results
    fig = plt.figure(figsize=(5, 1.66))
    fig.subplots_adjust(wspace=0.45,
                        bottom=0.25, top=0.9,
                        left=0.1, right=0.97)
    
    # plot density
    ax = fig.add_subplot(131)
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.plot(X[:,0], X[:,1], '.b')
    ax.xaxis.set_major_locator(plt.MultipleLocator(0.3))
    ax.set_xlim(np.min(X[:,0]), np.max(X[:,0]))
    ax.set_ylim(np.min(X[:,1]), np.max(X[:,1]))
    ax.text(0.93, 0.93, "Input",
            va='top', ha='right', transform=ax.transAxes)
    
    # plot AIC/BIC
    ax = fig.add_subplot(132)
    ax.plot(N, AIC, '-k', label='AIC')
    ax.plot(N, BIC, ':k', label='BIC')
    ax.legend(loc=1)
    ax.set_xlabel('N components')
    plt.setp(ax.get_yticklabels(), fontsize=7)
    
    # plot best configurations for AIC and BIC
    ax = fig.add_subplot(133)
    ax.plot(X[:,0], X[:,1], '.b')
    ax.scatter(gmm_best.means_[:, 0], gmm_best.means_[:, 1], c='w')
    for mu, C, w in zip(gmm_best.means_, gmm_best.covars_, gmm_best.weights_):
        draw_ellipse(mu, C, scales=[1.5], ax=ax, fc='none', ec='k')
    
    ax.text(0.93, 0.93, "Converged",
            va='top', ha='right', transform=ax.transAxes)
    
    ax.set_xlim(np.min(X[:,0]), np.max(X[:,0]))
    ax.set_ylim(np.min(X[:,1]), np.max(X[:,1]))
    ax.xaxis.set_major_locator(plt.MultipleLocator(0.3))
    ax.set_xlabel('')
    ax.set_ylabel('')
    