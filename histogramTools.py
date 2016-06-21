#!/usr/bin/evn python

from __future__ import division

from copy import deepcopy
from itertools import izip
import numpy as np
import matplotlib as mpl
#mpl.use('pdf')
from matplotlib.pyplot import *
import matplotlib as mpl
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm

import plotGoodies as PG


def scaleBin(ebin, factor):
    '''Make smaller-width bin centered about original bin.

    Useful for plotting discrete lines with linewidth > 1 since the lines
    spill over into adjacent bins, so this allows for tight (no gap) or an
    arbitrary gap. A factor 

    '''
    center = (ebin[0] + ebin[1])/2.0
    width = ebin[1] - ebin[0]
    return (center-width*factor/2.0, center+width*factor/2.0)


def logBinCenters(bin_edges):
    bin_edges = np.array(bin_edges)
    #width_factors = bin_edges[1:]/bin_edges[:-1]
    #centers = bin_edges[:-1] * np.sqrt(width_factors)
    #return centers
    return np.sqrt(bin_edges[:-1] * bin_edges[1:])


def linBinCenters(bin_edges):
    bin_edges = np.array(bin_edges)
    centers = (bin_edges[:-1]+bin_edges[1:])/2.0
    return centers


def isLogSpacing(bin_edges):
    bin_edges = np.array(bin_edges)
    mult_widths = bin_edges[1:] / bin_edges[:-1]
    if np.allclose(mult_widths[1:], mult_widths[0], rtol=1e-8, abstol=1e-5):
        return True


def autoBinCenters(bin_edges):
    # Are bins logarithmically spaced?
    if isLogSpacing(bin_edges):
        return logBinCenters(bin_edges)
    # of if any other spacing, return linear centers...
    else:
        return linBinCenters(bin_edges)


def histogram_with_error(a, bins=10, range=None, weights=None,
                         density=None):
    """Like numpy.histogram but also computes and returns per-bin error.
    
    See `numpy.histogram` for help on arguments.

    Returns
    -------
    hist : array
        The values of the historam. See `normed` and `weights` for a
        description of the possible semantics.
    bin_edges : array of dtype float
        Return the bin edges ``(length(hist)+1)``.
    err : array of dtype float
        Error for each bin, taking `weights` into account if supplied.

    See Also
    --------
    numpy.histogram
    """
    #if density:
    #    raise NotImplementedError('`density=True` not implemented correctly yet.')
    if weights is None:
        hist, bin_edges = np.histogram(a=a, bins=bins, range=range,
                                       weights=weights, density=False)
        err = np.sqrt(hist)
        if density:
            norm_fact = 1.0 / np.sum(hist * np.diff(bin_edges))
            hist = hist * norm_fact
            err = err * norm_fact

        return hist, bin_edges, err

    if np.isscalar(bins):
        if range is None:
            range = (np.min(a), np.max(a))
        assert len(range) == 2
        bin_edges = np.linspace(range[0], range[1], bins+1)
        n_bins = bins
    else:
        n_bins = len(bins) - 1
        assert n_bins > 1
        bin_edges = bins

    bin_nums = np.digitize(x=a, bins=bin_edges, right=False)

    hist = np.zeros(n_bins)
    err = np.zeros(n_bins)
    for idx, (bin_num, weight) in enumerate(izip(bin_nums, weights)):
        bin_num -= 1
        if bin_num < 0:
            continue
        if bin_num == n_bins:
            if a[idx] == bin_edges[-1]:
                bin_num -= 1
            else:
                continue
        elif bin_num > n_bins:
            continue
        hist[bin_num] += weight
        err[bin_num] += weight*weight

    if density:
        norm_fact = 1.0 / np.sum(hist * np.diff(bin_edges))
        hist = hist * norm_fact
        err = err * (norm_fact*norm_fact)

    err = np.sqrt(err)

    return hist, bin_edges, err


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

    # Initial guess for central gaussian: mu, sigma simply taken from
    # central region's mean and std dev; amplitude guess is found from
    # largest point in this region
    mu1_0 = np.mean(weightedData)
    sigma1_0 = 3 #np.std(weightedData)
    maxInd = np.where(binCounts == np.max(binCounts))
    x = binCenters[maxInd][0]
    y = binCounts[maxInd][0]
    A1_0 = y * np.exp((x-mu1_0)**2/(2*sigma1_0**2))

    p0 = [A1_0, mu1_0, sigma1_0]

    popt, pcov = sp.optimize.curve_fit(gaussian, binCenters, binCounts,
                                       p0=p0)

    return {'popt':popt, 'pcov':pcov, 'function':gaussian}
    

def fitTwoGaussainHist(bin_centers, bin_counts, sigma0=None,
                       auto_1d_or_2d=False):
    # TODO: throw some number of points distrubuted evenly thoughout each
    #       non-zero-count bin's binwidth to avoid the peaks that pop up
    #       between bin centers
    # TODO: Use a constrained minimizer:
    #       - No negative amplitudes
    #       - No negative sigma
    #       - Sigma must be greater than distance between bins, or bin
    #         width, or... ? ; how to deal with non-uniform bin
    #         widths/spacings?
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

    # Initial guess for central gaussian: mu, sigma simply taken from
    # central region's mean and std dev; amplitude guess is found from
    # largest point in this region
    mu1_0 = np.mean(weightedData[centralInd])
    if sigma0 == None:
        sigma1_0 = 3 #np.std(weightedData[centralInd])
        sigma2_0 = 20 #np.std(weightedData[centralInd])
    else:
        sigma1_0 = sigma0[0]
        sigma2_0 = sigma0[1]
    maxCentralInd = np.where(bin_counts[centralInd] ==
                             np.max(bin_counts[centralInd]))
    maxCentralInd_x = bin_centers[centralInd][maxCentralInd][0]
    maxCentralInd_y = bin_counts[centralInd][maxCentralInd][0]
    A1_0 = maxCentralInd_y \
            * np.exp((maxCentralInd_x-mu1_0)**2/(2*sigma1_0**2))

    # Initial guess for tail gaussian: mu, sigma from mean and stddev of
    # both tails combined; amplitude guess is found from largest point in
    # the tail region
    mu2_0 = np.mean(weightedData[tailsInd])
    #sigma2_0 = 5 #np.std(weightedData[tailsInd])
    maxTailsInd = np.where(bin_counts[tailsInd] ==
                           np.max(bin_counts[tailsInd]))
    maxTailsInd_x = bin_centers[tailsInd][maxTailsInd][0]
    maxTailsInd_y = bin_counts[tailsInd][maxTailsInd][0]
    A2_0 = maxTailsInd_y * np.exp((maxTailsInd_x-mu2_0)**2/(2*sigma2_0**2))

    params1 = Parameters()
    params1.add('A1', value=A1_0, min=0)
    params1.add('mu1', value=mu1_0, min=min(central_bin_centers),
                max=max(central_bin_centers))
    params1.add('sigma1', value=sigma1_0,
                min=np.abs(min(np.diff(bin_centers))),
                max=(central_bin_centers[-1]-central_bin_centers[0])/2)

    params2 = deepcopy(params1)
    params2.add('A2', value=A2_0, min=0)
    params2.add('mu2', value=mu2_0, min=min(central_bin_centers),
                max=max(central_bin_centers))
    params2.add('sigma2', value=sigma2_0,
                min=np.abs(min(np.diff(bin_centers))),
                max=(central_bin_centers[-1]-central_bin_centers[0])/2)

    #print 'DEBUG:', len(params), bin_centers, bin_counts
    try:
        result1 = minimize(gaussianErr, params1, args=(bin_centers,
                                                       bin_counts))
        mse1 = np.sum(result1.residual**2)
    except:
        result1 = None
        mse1 = np.inf

    try:
        result2 = minimize(twoGaussianErr, params2, args=(bin_centers,
                                                          bin_counts))
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
                       rh=0.04, tw=0.15, fs=16, title=None,
                       title_suffix='', display_title=True, title_fs=20):
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
                title = r'$' + numFmt(n_gaussians) \
                        + r'\mathrm{-gaussian\,fit}$'
        fullTitleText = title + title_suffix
        #ax.text(x+tw/2.0,y,title, va='center', ha='left', fontsize=fs, color=fg_color, transform=ax.transAxes)
        t = ax.text(x, y, fullTitleText,
                    va='center', ha='left', fontsize=fs, color=fg_color,
                    transform=ax.transAxes)
        t.set_path_effects([path_effects.Stroke(linewidth=borderLW,
                                                foreground=bg_color),
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
        t = ax.text(x,y,r'$A'+subscr+r'$', va='center', ha='left',
                    fontsize=fs, color=fg_color, transform=ax.transAxes)
        t.set_path_effects([path_effects.Stroke(linewidth=borderLW,
                                                foreground=bg_color),
                           path_effects.Normal()])
        t = ax.text(x+tw,y,r'$'+numFmt(A,keepAllSigFigs=1)+r'$',
                    va='center', ha='right', fontsize=fs, color=fg_color,
                    transform=ax.transAxes)
        t.set_path_effects([path_effects.Stroke(linewidth=borderLW,
                                                foreground=bg_color),
                           path_effects.Normal()])
        y -= rh
        t = ax.text(x,y, r'$\mu'+subscr+r'$', va='center', ha='left',
                    fontsize=fs, color=fg_color, transform=ax.transAxes)
        t.set_path_effects([path_effects.Stroke(linewidth=borderLW,
                                                foreground=bg_color),
                           path_effects.Normal()])
        t = ax.text(x+tw,y, r'$'+numFmt(mu,keepAllSigFigs=1)+r'$',
                    va='center', ha='right', fontsize=fs, color=fg_color,
                    transform=ax.transAxes)
        t.set_path_effects([path_effects.Stroke(linewidth=borderLW,
                                                foreground=bg_color),
                           path_effects.Normal()])
        y -= rh
        t = ax.text(x,y, r'$\sigma'+subscr+r'$', va='center', ha='left',
                    fontsize=fs, color=fg_color, transform=ax.transAxes)
        t.set_path_effects([path_effects.Stroke(linewidth=borderLW,
                                                foreground=bg_color),
                           path_effects.Normal()])
        t = ax.text(x+tw,y, r'$'+numFmt(sigma,keepAllSigFigs=1)+r'$',
                    va='center', ha='right', fontsize=fs, color=fg_color,
                    transform=ax.transAxes)
        t.set_path_effects([path_effects.Stroke(linewidth=borderLW,
                                                foreground=bg_color),
                           path_effects.Normal()])
        y -= rh*1.5

    return y


def autoBin(data, bins, log=False, overlap=0, overlap_is_pct=False):
    if hasattr(bins, '__iter__'):
        if hasattr(bins[0], '__iter__'):
            bins = np.array(bins)
        else:
            bins = np.array(zip(bins[0:-1], bins[1:]))
    else:
        dmin = min(data)
        dmax = max(data)
        drange = dmax-dmin
        n_bins = bins
        n_overlaps = n_bins - 1
        bin_width = (drange + n_overlaps*overlap)/n_bins
        left_edges = np.arange(dmin, dmax+bin_width, bin_width)

    overlap = bins[1::2,0] - bins[::2,1]


def medianPlot2D(x, y, z, xbins=None, ybins=None):
    xmin = min(x)
    xmax = max(x)
    xr = xmax-xmin
    ymin = min(x)
    ymax = max(y)
    yr = ymax-ymin
    if xbins is None:
        n_xbins = 50
    try:
        n_xbins = len(xbins)
        #xbins = 
    except:
        pass


def stepHist(bin_edges, y, yerr=None,
             plt_lr_edges=False, lr_edge_val=0,
             ax=None, eband_kwargs={}, **kwargs):
    x = np.array(zip(bin_edges, bin_edges)).flatten()
    y = np.array(
        [lr_edge_val] + list(np.array(zip(y, y)).flatten()) + [lr_edge_val],
        dtype=np.float64
    )

    # Populate the y-errors
    if yerr is not None:
        yerr = np.squeeze(yerr)
        if np.isscalar(yerr[0]):
            yerr_low = [-e for e in yerr]
            yerr_high = yerr
        else:
            yerr_low = yerr[0]
            yerr_high = yerr[1]
        yerr_low = np.array(
            [lr_edge_val] +
            list(np.array(zip(yerr_low, yerr_low)).flatten()) +
            [lr_edge_val],
            dtype=np.float64
        )
        yerr_high = np.array(
            [lr_edge_val] +
            list(np.array(zip(yerr_high, yerr_high)).flatten()) +
            [lr_edge_val],
            dtype=np.float64
        )

    # Remove extra values at edges if not plotting the extreme edges
    if not plt_lr_edges:
        x = x[1:-1]
        y = y[1:-1]
        if yerr is not None:
            yerr_low = yerr_low[1:-1]
            yerr_high = yerr_high[1:-1]

    # Create an axis if one isn't supplied
    if ax is None:
        f = figure()
        ax = f.add_subplot(111)
    
    # Plot the y-error bands
    err_plt = None
    if yerr is not None:
        custom_hatch = False
        if 'hatch' in eband_kwargs:
            hatch_kwargs = deepcopy(eband_kwargs)
            hatch_kwargs['facecolor'] = (1,1,1,0)
            eband_kwargs['hatch'] = None
            custom_hatch = True
        err_plt = ax.fill_between(x, y1=y+yerr_low, y2=y+yerr_high,
                                  **eband_kwargs)
        if custom_hatch:
            hatch_plt = ax.fill_between(x, y1=y+yerr_low, y2=y+yerr_high,
                                        **hatch_kwargs)
    
    # Plot the nominal values
    nom_lines = ax.plot(x, y, **kwargs)[0]

    # Match error bands' color to nominal lines
    if yerr is not None and not (('fc' in eband_kwargs) or
                                 ('facecolor' in eband_kwargs)):
        nl_color = nom_lines.get_color()
        nl_lw = nom_lines.get_linewidth()

        ep_facecolor = PG.hsvaFact(nl_color, sf=0.8, vf=1, af=0.5)
        #ep_edgecolor = PG.hsvaFact(nl_color, sf=0.8, vf=0.5, af=0.3)
        ep_edgecolor = 'none'
        err_plt.set_color(ep_facecolor)
        err_plt.set_facecolor(ep_facecolor)
        err_plt.set_edgecolor(ep_edgecolor)
        err_plt.set_linewidth(nl_lw*0.5)
    
        if custom_hatch:
            hatch_plt.set_color(eband_kwargs['edgecolor'])
            hatch_plt.set_facecolor((1,1,1,0))
            hatch_plt.set_edgecolor(eband_kwargs['edgecolor'])
            hatch_plt.set_linewidth(nl_lw*0.5)
    return ax, nom_lines, err_plt


def hist2d(x, y, bins=200, range=None, normed=False, weights=None, maskFun=None,
           ax=None, fig=None, tight_layout=False, log_normed=False,
           bgcolor=(0.6,)*3, cmap=mpl.cm.afmhot,
           vmin=None, vmax=None,
           colorbar_kwargs=None,
           #{'orientation':'vertical'}, #, 'ticks':2.0**np.arange(-10,10)},
           title=None, xlabel=None, ylabel=None, grid=True):
    
    if colorbar_kwargs is None:
        colorbar_kwargs = {'orientation':'vertical'} #, 'ticks':2.0**np.arange(-10,10)},
    H, xedges, yedges = np.histogram2d(x=x, y=y, bins=bins, range=range,
                                       normed=normed, weights=weights)

    if maskFun is None:
        Hmasked = H
    else:
        Hmasked = np.ma.masked_where(maskFun(H), H)
    
    gs = None
    if ax is None:
        if fig is None:
            fig = figure()
        gs = gridspec.GridSpec(1, 1)
        gs.update(left=0.10, right=0.95, wspace=0.05, bottom=0.1)
        ax = subplot(gs[0,0], axisbg=bgcolor)

    if log_normed:
        pmesh = ax.pcolormesh(xedges, yedges, Hmasked.T, cmap=cmap,
                              norm=LogNorm(vmin=Hmasked.min(), vmax=Hmasked.max()))
    else:
        pmesh = ax.pcolormesh(xedges, yedges, Hmasked.T, cmap=cmap,
                              vmin=vmin, vmax=vmax)

    cbar = fig.colorbar(pmesh, **colorbar_kwargs)
    colorbarTicklocs = cbar.ax.get_xticks()
    cbar.set_ticklabels([r'$'+numFmt(n)+r'$' for n in colorbarTicklocs])

    title_text = None
    xlabel_text = None
    ylabel_text = None
    if not title is None:
        title_text = ax.set_title(title)
    if not xlabel is None:
        xlabel_text = ax.set_xlabel(xlabel)
    if not ylabel is None:
        ylabel_text = ax.set_ylabel(ylabel)
    if grid:
        if isinstance(grid, bool):
            ax.grid(b=grid)
        else:
            ax.grid(**grid)

    TOP = 0.90
    BOTTOM = 0.08
    RIGHT = 1.00
    LEFT = 0.09

    if tight_layout:
        print 'tight_layout'
        fig.tight_layout()
    else:
        print 'custom layout'
        fig.subplots_adjust(top=TOP, bottom=BOTTOM, left=LEFT, right=RIGHT)

    return {'H': H, 'Hmasked':Hmasked, 'xedges':xedges, 'yedges':yedges,
            'fig':fig, 'gs': gs, 'ax':ax, 'pmesh':pmesh, 'cbar':cbar,
            'title_text':title_text, 'xlabel_text':xlabel_text, 'ylabel_text':ylabel_text}
