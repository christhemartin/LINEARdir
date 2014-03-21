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
from astroML.utils import split_samples
from astroML.utils import completeness_contamination
from sklearn.neighbors import KNeighborsClassifier

def read_data():
    '''
    Parameters
    ---------- 
    input : 
        n/a
    
    Returns
    -------
    data : dict
        dictionary of object features
    
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
    '''Find objects, copy individual plots, and place them into a separate directory for later viewing
    
    Parameters
    ----------
    badObjs : array_like
        IDs of objects 
 
    Returns
    -------
    output: 
        n/a
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
    '''converts between corresponding ints and strings pr. lcType
    
    Parameters
    ---------- 
    object : int or string
        number from 0 - 11 or string of object type, (should exclude 10 as an artifact of PLV_LINEAR.dat! explains placeholder entry for i = 10)
    
    Returns
    -------
    lcType : string or int
        number from 0 - 11 or string of object type, (opposite the input)
    '''
    if isinstance(object, float):
        types = ['Other','RR_Lyrae_ab','RR_Lyrae_c','algol_1','algol_2','contact_bin','DSSP','LPV','heartbeat','BL_hercules', 'listed_as_type_10','anom_ceph']
        lcType = types[int(object)]
    if isinstance(object,str):
        types = {'Other': 0 , 'RR_Lyrae_ab': 1,'RR_Lyrae_c': 2,'algol_1': 3,'algol_2': 4,'contact_bin': 5,'DSSP': 6,'LPV': 7,'heartbeat': 8,\
                 'BL_hercules': 9, 'listed_as_type_10': 10,'anom_ceph': 11}
        lcType = types[object]
    return lcType

def ComputeRFP(data):
    '''for each object, use coefficients from Fourier transform to calculate various relative fourier parameters.
    
    Parameters
    ---------- 
    data : dict
        data read from coefficients.dat file
    
    Returns
    -------
    RFP : dict
        dictionary of all computed relative fourier parameters
    '''
    od = os.getcwd()
    if od.endswith('\\Scripts'):
        os.chdir('..')
    
    coeffs = np.array(data['coefficients'])
    
    A1 = np.array([np.sqrt(object[7]**2.0 + object[1]**2.0) for object in coeffs])
    A2 = np.array([np.sqrt(object[8]**2.0 + object[2]**2.0) for object in coeffs])
    A3 = np.array([np.sqrt(object[9]**2.0 + object[3]**2.0) for object in coeffs])
    
    Phi1 = np.array([np.arctan(-object[1]/object[7]) for object in coeffs])%1
    Phi2 = np.array([np.arctan(-object[2]/object[8]) for object in coeffs])%1
    Phi3 = np.array([np.arctan(-object[3]/object[9]) for object in coeffs])%1
    
    RFP = {}
    RFP['R21'] = A2/A1
    RFP['R31'] = A3/A1
    RFP['Phi21'] = Phi2 - 2.0*Phi1
    RFP['Phi31'] = Phi3 - 3.0*Phi1
    
    return RFP

def plotRFP(RFP, lcType):
    '''plots all combinations of the relative fourier parameters across lcTypes
    Parameters
    ---------- 
    RFP : dict
        dictionary of all computed relative fourier parameters
    lcType : array_like
        contains strings of object types
    
    Returns
    -------
    output:
        n/a
    '''

    objectIdx = {}
    objectIdx['RR_Lyrae_ab'] = [i for i in range(len(lcType)) if lcType[i] == 'RR_Lyrae_ab']
    objectIdx['RR_Lyrae_c'] = [i for i in range(len(lcType)) if lcType[i] == 'RR_Lyrae_c']
#    objectIdx['algol_1'] = [i for i in range(len(lcType)) if lcType[i] == 'algol_1']
    objectIdx['algol_2'] = [i for i in range(len(lcType)) if lcType[i] == 'algol_2']
    objectIdx['contact_bin'] = [i for i in range(len(lcType)) if lcType[i] == 'contact_bin']
#    objectIdx['DSSP'] = [i for i in range(len(lcType)) if lcType[i] == 'DSSP']
#    objectIdx['LPV'] = [i for i in range(len(lcType)) if lcType[i] == 'LPV']
#    objectIdx['heartbeat'] = [i for i in range(len(lcType)) if lcType[i] == 'heartbeat']
#    objectIdx['BL_hercules'] = [i for i in range(len(lcType)) if lcType[i] == 'BL_hercules']
#    objectIdx['anom_ceph'] = [i for i in range(len(lcType)) if lcType[i] == 'anom_ceph']
    
    count = 0
#    fig = plt.figure()
#    plt.suptitle('Relative Fourier Parameters')
    for key in objectIdx.keys():
        idx = objectIdx[key]
        
        cm = plt.get_cmap('jet')
            
        fig = plt.figure()
        plt.suptitle('Relative Fourier Parameters - ' + key)
        color = cm(1.*float(count)/len(objectIdx.keys()))

        ax1 = fig.add_subplot(3,2,1)
        ax1.scatter(np.log(RFP['R21'][idx]), np.log(RFP['R31'][idx]), c = color, marker ='o', alpha = .7)
        ax1.set_ylim(np.min(np.log(RFP['R31'])), np.max(np.log(RFP['R31'])))
        ax1.set_xlim(np.min(np.log(RFP['R21'])), np.max(np.log(RFP['R21'])))
        ax1.set_title('R21 vs R31')
        ax1.set_xlabel('R21')
        ax1.set_ylabel('R31')
        
        ax2 = fig.add_subplot(3,2,2)
        ax2.scatter(np.log(RFP['R21'][idx]), RFP['Phi21'][idx], c = color, marker ='o', alpha = .7)
        ax2.set_ylim(np.min(RFP['Phi21']), np.max(RFP['Phi21']))
        ax2.set_xlim(np.min(np.log(RFP['R21'])), np.max(np.log(RFP['R21'])))
        ax2.set_title('R21 vs. Phi21')
        ax2.set_xlabel('R21')
        ax2.set_ylabel('Phi21')
        
        ax3 = fig.add_subplot(3,2,3)
        ax3.scatter(np.log(RFP['R21'][idx]), RFP['Phi31'][idx], c = color, marker ='o', alpha = .7)
        ax3.set_ylim(np.min(RFP['Phi31']), np.max(RFP['Phi31']))
        ax3.set_xlim(np.min(np.log(RFP['R21'])), np.max(np.log(RFP['R21'])))
        ax3.set_title('R21 vs. Phi31')
        ax3.set_xlabel('R21')
        ax3.set_ylabel('Phi31')
        
        ax4 = fig.add_subplot(3,2,4)
        ax4.scatter(np.log(RFP['R31'][idx]), RFP['Phi21'][idx], c = color, marker ='o', alpha = .7)
        ax4.set_ylim(np.min(RFP['Phi21']), np.max(RFP['Phi21']))
        ax4.set_xlim(np.min(np.log(RFP['R31'])), np.max(np.log(RFP['R31'])))
        ax4.set_title('R31 vs. Phi21')
        ax4.set_xlabel('R31')
        ax4.set_ylabel('Phi21')
        
        ax5 = fig.add_subplot(3,2,5)
        ax5.scatter(np.log(RFP['R31'][idx]), RFP['Phi31'][idx], c = color, marker ='o', alpha = .7)
        ax5.set_ylim(np.min(RFP['Phi31']), np.max(RFP['Phi31']))
        ax5.set_xlim(np.min(np.log(RFP['R31'])), np.max(np.log(RFP['R31'])))
        ax5.set_title('R31 vs Phi31')
        ax5.set_xlabel('R31')
        ax5.set_ylabel('Phi31')
        
        ax6 = fig.add_subplot(3,2,6)
        ax6.scatter(RFP['Phi21'][idx], RFP['Phi31'][idx], c = color, marker ='o', alpha = .7, label = key)
        ax6.set_ylim(np.min(RFP['Phi31']), np.max(RFP['Phi31']))
        ax6.set_xlim(np.min(RFP['Phi21']), np.max(RFP['Phi21']))
        ax6.set_title('Phi21 vs Phi31')
        ax6.set_xlabel('Phi21')
        ax6.set_ylabel('Phi31')
        legend6 = ax6.legend(loc='lower right', ncol=1, shadow=True)
        count += 1
    plt.show()

def plotRFP_vs_P(RFP, P, lcType):
    '''plots period vs relative fourier parameters across lcTypes
    Parameters
    ---------- 
    RFP : dict
        dictionary of all computed relative fourier parameters
    P : array_like
        periods of objects
    lcType : array_like
        contains strings of object types
    
    Returns
    -------
    output:
        n/a
    '''
    P = np.array(P)
    objectIdx = {}
    objectIdx['RR_Lyrae_ab'] = [i for i in range(len(lcType)) if lcType[i] == 'RR_Lyrae_ab']
    objectIdx['RR_Lyrae_c'] = [i for i in range(len(lcType)) if lcType[i] == 'RR_Lyrae_c']
#    objectIdx['algol_1'] = [i for i in range(len(lcType)) if lcType[i] == 'algol_1']
    objectIdx['algol_2'] = [i for i in range(len(lcType)) if lcType[i] == 'algol_2']
    objectIdx['contact_bin'] = [i for i in range(len(lcType)) if lcType[i] == 'contact_bin']
#    objectIdx['DSSP'] = [i for i in range(len(lcType)) if lcType[i] == 'DSSP']
#    objectIdx['LPV'] = [i for i in range(len(lcType)) if lcType[i] == 'LPV']
#    objectIdx['heartbeat'] = [i for i in range(len(lcType)) if lcType[i] == 'heartbeat']
#    objectIdx['BL_hercules'] = [i for i in range(len(lcType)) if lcType[i] == 'BL_hercules']
#    objectIdx['anom_ceph'] = [i for i in range(len(lcType)) if lcType[i] == 'anom_ceph']
    
    count = 0
    fig = plt.figure()
    plt.suptitle('Relative Fourier Parameters vs. Period')
    for key in objectIdx.keys():
        idx = objectIdx[key]
        
        cm = plt.get_cmap('jet')
            
#        fig = plt.figure()
#        plt.suptitle('Relative Fourier Parameters - ' + key)
        color = cm(1.*float(count)/len(objectIdx.keys()))
        
        ax1 = fig.add_subplot(2,2,1)
        ax1.scatter(np.log(RFP['R21'][idx]), np.log(P[idx]), c = color, marker ='o', alpha = .7, label = key)
        legend1 = ax1.legend(loc='upper left', ncol=1, shadow=True)
        ax1.set_ylabel('log(Period)')
        ax1.set_xlabel('log(R21)')
        ax1.grid(True, which='major', linestyle='--', color = "#a6a6a6", 
            zorder=2, alpha = .8)
        
        ax2 = fig.add_subplot(2,2,2)
        ax2.scatter(np.log(RFP['R31'][idx]), np.log(P[idx]), c = color, marker ='o', alpha = .7)
        ax2.set_ylabel('log(Period)')
        ax2.set_xlabel('log(R31)')
        ax2.grid(True, which='major', linestyle='--', color = "#a6a6a6", 
            zorder=2, alpha = .8)
        
        ax3 = fig.add_subplot(2,2,3)
        ax3.scatter(RFP['Phi21'][idx], np.log(P[idx]), c = color, marker ='o', alpha = .7)
        ax3.set_ylabel('log(Period)')
        ax3.set_xlabel('log(Phi21)')
        ax3.grid(True, which='major', linestyle='--', color = "#a6a6a6", 
            zorder=2, alpha = .8)
        
        ax4 = fig.add_subplot(2,2,4)
        ax4.scatter(RFP['Phi31'][idx], np.log(P[idx]), c = color, marker ='o', alpha = .7)
        ax4.set_ylabel('log(Period)')
        ax4.set_xlabel('log(Phi31)')
        ax4.grid(True, which='major', linestyle='--', color = "#a6a6a6", 
            zorder=2, alpha = .8)
        
        count += 1
    
    plt.show()

def plot_lcFit(phased_t, y, dy, phase_fit, y_fit, fig, id, omega_best):
    '''
    input: 
    function: Create a plot of an individual star's lightcurve with errorbars and our fit, phase folded to best fir period. 
    output: n/a
    
    Parameters
    ---------- 
        Phased_t : array_like
            array of times corresponding to phase folded flux data
        y : array_like
            flux data from star
        dy : array_like
            error in flux
        phase_fit : array_like
            time array corresponding to phase folded fit data
        y_fit : array_like
            fit of data
        fig : 
            current plot frame
        id : 
            id number of object
        omega_best : float
            best estimate of omega
            
    Returns
    -------
    output:
        n/a
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
    '''To plot separately, different lcTypes for one metric vs. another. Could be any of chi2dof/chi2R/sigmaG.
    input: Arrays to be plotted against each other (xData,yData), Strings of their names to identify plots (xName,yName).
    Parameters
    ---------- 
    xData : array_like
        data to be plotted on x-axis
    yData : array_like
        data to be plotted on y-axis
    lcType : array_like
        list of integers corresponding to object type
    xName : str
        label for variable plotted on x-axis
    yName : str
        label for variable plotted on y-axis
    
    Returns
    -------
    output:
        n/a
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
    '''Plot each metric vs. another in a 2D scatter with colors representing lcType
    Parameters
    ---------- 
    chi2dofArray : array_like
        contains calculated chi2 pr. degree of freedom values
    chi2RArray : array_like
        contains calculated robust chi2 values
    sigmaGArray : array_like
        contains calulated values of sigmaG
    lcType : array_like
        list of integers corresponding to object type 
    
    Returns
    -------
    output:
        n/a
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
    '''Create a 3D scatter plot of the metric space, coloring points according to lcType
    Parameters
    ---------- 
    chi2dofArray : array_like
        contains calculated chi2 pr. degree of freedom values
    chi2RArray : array_like
        contains calculated robust chi2 values
    sigmaGArray : array_like
        contains calulated values of sigmaG
    lcType : array_like
        list of integers corresponding to object type 
        
    Returns
    -------
    output: 
        n/a
    '''
    fig = plt.figure()
    fig.suptitle('Metric Space for chi2dof, chi2R & sigmaG')
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(np.log(chi2dofArray), np.log(chi2RArray), np.log(sigmaGArray), c=lcType, cmap = plt.cm.jet, alpha = .7)
    ax.set_xlabel('chi2dof')
    ax.set_ylabel('chi2R')
    ax.set_zlabel('sigmaG')
    
def plot_metricDists(chi2dofArray, chi2RArray, sigmaGArray):
    '''Create a histogram of the distribution for each metric
    Parameters
    ----------
    chi2dofArray : array_like
        contains calculated chi2 pr. degree of freedom values
    chi2RArray : array_like
        contains calculated robust chi2 values
    sigmaGArray : array_like
        contains calulated values of sigmaG
    
    Returns
    -------
    output: 
        n/a
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
    '''Create a separate histogram for every lcType within each metric's distribution. (to see overlap of each distribution)
    Parameters
    ---------- 
    chi2dofArray : array_like
        contains calculated chi2 pr. degree of freedom values
    chi2RArray : array_like
        contains calculated robust chi2 values
    sigmaGArray : array_like
        contains calulated values of sigmaG
    lcType : array_like
        list of integers corresponding to object type 
    
    Returns
    -------
    output: 
        n/a
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
    '''Plot ratio of sigmaGArray/chi2dofArray vs. chi2RArray/chi2dofArray
    Parameters
    ----------
    chi2dofArray : array_like
        contains calculated chi2 pr. degree of freedom values
    chi2RArray : array_like
        contains calculated robust chi2 values
    sigmaGArray : array_like
        contains calulated values of sigmaG
    lcType : array_like
        list of integers corresponding to object type
    ids : array_like
        contains id numbers of objects
        
    Returns
    -------
    output: 
        n/a
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
    '''For a given matrix of data, computes a variety gaussian mixture models increasing fit complexity, selects a best model 
    based on AIC and BIC results.
    
    Parameters
    ----------
    X : array_like
        2D matrix, each entry being a set of parameters for a sample (n_samples by n_features)
    
    Returns
    -------
    output : 
        n/a
    '''
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
    plt.show()
    
def KNN(X, y):
    '''
    Parameters
    ----------
    X : array_like
        2D matrix, each entry being a set of parameters for a sample (n_samples by n_features)
    y : array_like
        1D array of integers corresponding to labels for each sample
    
    Returns
    -------
    output : 
        n/a
    '''
    print len(X)
    
    # get data and split into training & testing sets
    (X_train, X_test), (y_train, y_test) = split_samples(X, y, [0.75, 0.25],
                                                         random_state=0)
    
    N_tot = len(y)
    N_st = np.sum(y == 0)
    N_rr = N_tot - N_st
    N_train = len(y_train)
    N_test = len(y_test)
    N_plot = 5000 + N_rr
    
    #----------------------------------------------------------------------
    # perform Classification
    
    classifiers = []
    predictions = []
    Ncolors = np.arange(1, X.shape[1] + 1)
    kvals = [1, 15]
    
    for k in kvals:
        classifiers.append([])
        predictions.append([])
        for nc in Ncolors:
            clf = KNeighborsClassifier(n_neighbors=k)
            clf.fit(X_train[:, :nc], y_train)
            y_pred = clf.predict(X_test[:, :nc])
    
            classifiers[-1].append(clf)
            predictions[-1].append(y_pred)
    
    completeness, contamination = completeness_contamination(predictions, y_test)
    
    print "completeness", completeness
    print "contamination", contamination
    
    #------------------------------------------------------------
    # Compute the decision boundary
    clf = classifiers[1][1]
    xlim = (0.7, 1.35)
    ylim = (-0.15, 0.4)
    
    xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 71),
                         np.linspace(ylim[0], ylim[1], 81))
    
    Z = clf.predict(np.c_[yy.ravel(), xx.ravel()])
    Z = Z.reshape(xx.shape)
    
    #----------------------------------------------------------------------
    # plot the results
    fig = plt.figure(figsize=(5, 2.5))
    fig.subplots_adjust(bottom=0.15, top=0.95, hspace=0.0,
                        left=0.1, right=0.95, wspace=0.2)
    
    # left plot: data and decision boundary
    ax = fig.add_subplot(121)
    im = ax.scatter(X[-N_plot:, 1], X[-N_plot:, 0], c=y[-N_plot:],
                    s=4, lw=0, cmap=plt.cm.binary, zorder=2)
    im.set_clim(-0.5, 1)
    
#    im = ax.imshow(Z, origin='lower', aspect='auto',
#                   cmap=plt.cm.binary, zorder=1,
#                   extent=xlim + ylim)
    im.set_clim(0, 2)
    
    #ax.contour(xx, yy, Z, [0.5], colors='k')
    
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    
    ax.set_xlabel('')
    ax.set_ylabel('')
    
    ax.text(0.02, 0.02, "k = %i" % kvals[1],
            transform=ax.transAxes)
    
    # plot completeness vs Ncolors
    ax = fig.add_subplot(222)
    
    ax.plot(Ncolors, completeness[0], 'o-k', ms=6, label='k=%i' % kvals[0])
    ax.plot(Ncolors, completeness[1], '^--k', ms=6, label='k=%i' % kvals[1])
    
    ax.xaxis.set_major_locator(plt.MultipleLocator(1))
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.2))
    ax.xaxis.set_major_formatter(plt.NullFormatter())
    
    ax.set_ylabel('completeness')
    ax.set_xlim(0.5, 4.5)
    ax.set_ylim(-0.1, 1.1)
    ax.grid(True)
    
    # plot contamination vs Ncolors
    ax = fig.add_subplot(224)
    ax.plot(Ncolors, contamination[0], 'o-k', ms=6, label='k=%i' % kvals[0])
    ax.plot(Ncolors, contamination[1], '^--k', ms=6, label='k=%i' % kvals[1])
    ax.legend(loc='lower right',
              bbox_to_anchor=(1.0, 0.79))
    
    ax.xaxis.set_major_locator(plt.MultipleLocator(1))
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.2))
    ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%i'))
    ax.set_xlabel('N colors')
    ax.set_ylabel('contamination')
    ax.set_xlim(0.5, 4.5)
    ax.set_ylim(-0.1, 1.1)
    ax.grid(True)
    
    plt.show()
