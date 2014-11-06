#!/usr/bin/evn python

import numpy as np
from matplotlib.pyplot import *


def scaleBin(ebin, factor):
    '''Make smaller-width bin centered about original bin.

    Useful for plotting discrete lines with linewidth > 1 since the lines spill over into adjacent bins, so this allows for tight (no gap) or an arbitrary gap. A factor 

    '''
    center = (ebin[0] + ebin[1])/2
    width = ebin[1] - ebin[0]
    return (center-width*factor/2, center+width*factor/2)

def gaussian(x, A, mu, sigma):
    return A * np.exp(-(x-mu)**2/2/sigma**2)

def twoGaussian(x, A1, A2, mu1, mu2, sigma1, sigma2):
    return gaussian(x, A1, mu1, sigma1) + gaussian(x, A2, mu2, sigma2)

def gaussianErr(params, x, data):
    return gaussian(x, A=params['A1'].value,
                       mu=params['mu1'].value,
                       sigma=params['sigma1'].value) - data

def twoGaussianErr(params, x, data):
    return twoGaussian(x, A1=params['A1'].value, A2=params['A2'].value,
                       mu1=params['mu1'].value, mu2=params['mu2'].value,
                       sigma1=params['sigma1'].value,
                       sigma2=params['sigma2'].value) - data

def fitGaussainHist(binEdges, binCounts):
    # TODO: log-likelihood method, Chi-squared
    binCenters = (binEdges[:-1] + binEdges[1:])/2.0
    binWidths = binEdges[1:] - binEdges[:-1]
    #binCenters = np.array(binCenters)
    binCounts = np.array(binCounts)

    #-- Eliminate 0-count bins, since gaussians are *never* 0
    validBinInd = np.where(binCounts>0)
    binCenters = binCenters[validBinInd]
    binCounts = binCounts[validBinInd]

    weightedData = binCenters * binCounts

    #-- Initial guess for central gaussian: mu, sigma simply taken from central
    #   region's mean and std dev; amplitude guess is found from largest point
    #   in this region
    mu1_0 = np.mean(weightedData)
    sigma1_0 = 3 #np.std(weightedData)
    maxInd = np.where(binCounts == np.max(binCounts))
    x = binCenters[maxInd][0]
    y = binCounts[maxInd][0]
    A1_0 = y * np.exp((x-mu1_0)**2/(2*sigma1_0**2))

    p0 = [A1_0, mu1_0, sigma1_0]

    popt, pcov = sp.optimize.curve_fit(gaussian, binCenters, binCounts, p0=p0)

    return {'popt':popt, 'pcov':pcov, 'function':gaussian}
    

def fitTwoGaussainHist(bin_centers, bin_counts, sigma0=None, auto_1d_or_2d=False):
    # TODO: throw some number of points distrubuted evenly thoughout each
    #       non-zero-count bin's binwidth to avoid the peaks that pop up
    #       between bin centers
    # TODO: Use a constrained minimizer:
    #       - No negative amplitudes
    #       - No negative sigma
    #       - Sigma must be greater than distance between bins, or bin width,
    #         or... ? ; how to deal with non-uniform bin widths/spacings?
    # TODO: Pick between the better of 1- and 2-gaussian fits, return
    #       only the results of the better one (make optional via function
    #       parameter)
    # TODO: Re-order output such that narrower gaussian comes first

    bin_centers = np.array(bin_centers)
    bin_counts = np.array(bin_counts)
    
    #-- Eliminate 0-count bins, since gaussians are *never* 0
    validBinInd = np.where(bin_counts>0)
    bin_centers = bin_centers[validBinInd]
    bin_counts = bin_counts[validBinInd]
    
    weightedData = bin_centers * bin_counts

    #-- Extract central and tail regions via quartiles
    cs = np.cumsum(bin_counts)
    tot = cs[-1]
    tailsInd = np.where((cs <= tot*0.4) + (cs >= tot*0.6))
    centralInd = np.where((cs >= tot*0.1) * (cs <= tot*0.9))
    if len(centralInd) <= 4:
        centralInd = np.arange(len(bin_centers), dtype=int)
    central_bin_centers = bin_centers[centralInd]
    #print centralInd, tailsInd
    #print central_bin_centers

    #-- Initial guess for central gaussian: mu, sigma simply taken from central
    #   region's mean and std dev; amplitude guess is found from largest point
    #   in this region
    mu1_0 = np.mean(weightedData[centralInd])
    if sigma0 == None:
        sigma1_0 = 3 #np.std(weightedData[centralInd])
        sigma2_0 = 20 #np.std(weightedData[centralInd])
    else:
        sigma1_0 = sigma0[0]
        sigma2_0 = sigma0[1]
    maxCentralInd = np.where(bin_counts[centralInd] == np.max(bin_counts[centralInd]))
    maxCentralInd_x = bin_centers[centralInd][maxCentralInd][0]
    maxCentralInd_y = bin_counts[centralInd][maxCentralInd][0]
    A1_0 = maxCentralInd_y * np.exp((maxCentralInd_x-mu1_0)**2/(2*sigma1_0**2))

    #-- Initial guess for tail gaussian: mu, sigma from mean and stddev of both
    #   tails combined; amplitude guess is found from largest point in the tail
    #   region
    mu2_0 = np.mean(weightedData[tailsInd])
    #sigma2_0 = 5 #np.std(weightedData[tailsInd])
    maxTailsInd = np.where(bin_counts[tailsInd] == np.max(bin_counts[tailsInd]))
    maxTailsInd_x = bin_centers[tailsInd][maxTailsInd][0]
    maxTailsInd_y = bin_counts[tailsInd][maxTailsInd][0]
    A2_0 = maxTailsInd_y * np.exp((maxTailsInd_x-mu2_0)**2/(2*sigma2_0**2))

    params1 = Parameters()
    params1.add('A1', value=A1_0, min=0)
    params1.add('mu1', value=mu1_0, min=min(central_bin_centers), max=max(central_bin_centers))
    params1.add('sigma1', value=sigma1_0, min=np.abs(min(np.diff(bin_centers))), max=(central_bin_centers[-1]-central_bin_centers[0])/2)

    params2 = copy.deepcopy(params1)
    params2.add('A2', value=A2_0, min=0)
    params2.add('mu2', value=mu2_0, min=min(central_bin_centers), max=max(central_bin_centers))
    params2.add('sigma2', value=sigma2_0, min=np.abs(min(np.diff(bin_centers))), max=(central_bin_centers[-1]-central_bin_centers[0])/2)

    #print 'DEBUG:', len(params), bin_centers, bin_counts
    try:
        result1 = minimize(gaussianErr, params1, args=(bin_centers, bin_counts))
        mse1 = np.sum(result1.residual**2)
    except:
        result1 = None
        mse1 = np.inf

    try:
        result2 = minimize(twoGaussianErr, params2, args=(bin_centers, bin_counts))
        mse2 = np.sum(result2.residual**2)
    except:
        result2 = None
        mse2 = np.inf

    #report_fit(params)

    #p0 = [A1_0, A2_0, mu1_0, mu2_0, sigma1_0, sigma2_0]
    #print p0
    #popt, pcov = sp.optimize.curve_fit(twoGaussian, bin_centers, bin_counts, p0=p0)
    pcov = None

    if (result1 is None) and (result2 is None):
        #raise Exception('Failed to fit either 1 or 2 gaussians to data!')
        wstderr('Failed to fit either 1 or 2 gaussians to data!\n')
        return None

    if (result2 is None) or (mse1 < mse2):
        popt = (params1['A1'].value,
                params1['mu1'].value,
                params1['sigma1'].value)
        funcfunc = gaussian

    else:
        popt = (params2['A1'].value, params2['A2'].value,
                params2['mu1'].value, params2['mu2'].value,
                params2['sigma1'].value, params2['sigma2'].value)
        funcfunc = twoGaussian

    return {'popt':popt, 'pcov':pcov, 'function': funcfunc}


def plotGaussianParams(ax, fitDict, x0, y0, fg_color='k', bg_color='w',
                       rh=0.04, tw=0.15, fs=16, title=None, title_suffix='',
                       display_title=True, title_fs=20):
    borderLW = 3
    # TODO: White outline around fit labels
    popt = fitDict['popt']
    n_gaussians = len(popt)/3
    if n_gaussians != int(n_gaussians):
        raise Exception('Need integer multiple of 3 parameters!')
    
    x = x0
    y = y0
    if (title != None) or display_title:
        if title is None:
            if n_gaussians == 1:
                title = r'$\mathrm{Gaussian\,fit}$'
            else:
                title = r'$' + numFmt(n_gaussians) + r'\mathrm{-gaussian\,fit}$'
        fullTitleText = title + title_suffix
        #ax.text(x+tw/2.0,y,title, va='center', ha='left', fontsize=fs, color=fg_color, transform=ax.transAxes)
        t = ax.text(x, y, fullTitleText,
                    va='center', ha='left', fontsize=fs, color=fg_color,
                    transform=ax.transAxes)
        t.set_path_effects([path_effects.Stroke(linewidth=borderLW, foreground=bg_color),
                           path_effects.Normal()])
        y -= rh #*1.5

    n_gaussians = int(n_gaussians)
    for g_n in range(n_gaussians):
        A = popt[g_n]
        mu = popt[g_n+n_gaussians*1]
        sigma = np.abs(popt[g_n+n_gaussians*2])
        if n_gaussians == 1:
            subscr = ''
        else:
            subscr = r'_{' + str(g_n+1) + r'}'
        t = ax.text(x,y,r'$A'+subscr+r'$', va='center', ha='left', fontsize=fs, color=fg_color, transform=ax.transAxes)
        t.set_path_effects([path_effects.Stroke(linewidth=borderLW, foreground=bg_color),
                           path_effects.Normal()])
        t = ax.text(x+tw,y,r'$'+numFmt(A,keepAllSigFigs=1)+r'$', va='center', ha='right', fontsize=fs, color=fg_color, transform=ax.transAxes)
        t.set_path_effects([path_effects.Stroke(linewidth=borderLW, foreground=bg_color),
                           path_effects.Normal()])
        y -= rh
        t = ax.text(x,y, r'$\mu'+subscr+r'$', va='center', ha='left', fontsize=fs, color=fg_color, transform=ax.transAxes)
        t.set_path_effects([path_effects.Stroke(linewidth=borderLW, foreground=bg_color),
                           path_effects.Normal()])
        t = ax.text(x+tw,y, r'$'+numFmt(mu,keepAllSigFigs=1)+r'$', va='center', ha='right', fontsize=fs, color=fg_color, transform=ax.transAxes)
        t.set_path_effects([path_effects.Stroke(linewidth=borderLW, foreground=bg_color),
                           path_effects.Normal()])
        y -= rh
        t = ax.text(x,y, r'$\sigma'+subscr+r'$', va='center', ha='left', fontsize=fs, color=fg_color, transform=ax.transAxes)
        t.set_path_effects([path_effects.Stroke(linewidth=borderLW, foreground=bg_color),
                           path_effects.Normal()])
        t = ax.text(x+tw,y, r'$'+numFmt(sigma,keepAllSigFigs=1)+r'$', va='center', ha='right', fontsize=fs, color=fg_color, transform=ax.transAxes)
        t.set_path_effects([path_effects.Stroke(linewidth=borderLW, foreground=bg_color),
                           path_effects.Normal()])
        y -= rh*1.5

    return y


