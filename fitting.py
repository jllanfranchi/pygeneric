# DO NOT CHANGE THIS FILE! (?)
#
# This file contains the functions linear_fit for fitting a straight
# line to data and general_fit for fitting any user-defined funciton
# to data.  To use either of them,  the first line of your program
# should be "from fitting import *".

import lmfit
import sys
import numpy as np
from scipy.optimize import curve_fit


def linear_fit(xdata, ydata, ysigma=None):

    """
    Performs a linear fit to data.

    Parameters
    ----------
    xdata : An array of length N.
    ydata : An array of length N.
    sigma : None or an array of length N,
        If provided, it is the standard-deviation of ydata.
        This vector, if given, will be used as weights in the fit.

    Returns
    -------
    a, b   : Optimal parameter of linear fit (y = a*x + b)
    sa, sb : Uncertainties of the parameters
    """
    
    if ysigma is None:
        w = np.ones(len(ydata)) # Each point is equally weighted.
    else:
        w=1.0/(ysigma**2)

    sw = sum(w)
    wx = w*xdata # this product gets used to calculate swxy and swx2
    swx = sum(wx)
    swy = sum(w*ydata)
    swxy = sum(wx*ydata)
    swx2 = sum(wx*xdata)

    a = (sw*swxy - swx*swy)/(sw*swx2 - swx*swx)
    b = (swy*swx2 - swx*swxy)/(sw*swx2 - swx*swx)
    sa = np.sqrt(sw/(sw*swx2 - swx*swx))
    sb = np.sqrt(swx2/(sw*swx2 - swx*swx))

    if ysigma is None:
        chi2 = sum(((a*xdata + b)-ydata)**2)
    else:
        chi2 = sum((((a*xdata + b)-ydata)/ysigma)**2)
    dof = len(ydata) - 2
    rchi2 = chi2/dof
    #print 'results of linear_fit:'
    #print '   chi squared = ', chi2
    #print '   degrees of freedom = ', dof
    #print '   reduced chi squared = ', rchi2

    return a, b, sa, sb, rchi2, dof


def general_fit(f, xdata, ydata, p0=None, sigma=None, **kw):
    """
    Pass all arguments to curve_fit, which uses non-linear least squares
    to fit a function, f, to data.  Calculate the uncertaities in the
    fit parameters from the covariance matrix.
    """
    popt, pcov = curve_fit(f, xdata, ydata, p0, sigma, **kw)

    if sigma is None:
        chi2 = sum(((f(xdata,*popt)-ydata))**2)
    else:
        chi2 = sum(((f(xdata,*popt)-ydata)/sigma)**2)
    dof = len(ydata) - len(popt)
    rchi2 = chi2/dof
    #print 'results of general_fit:'
    #print '   chi squared = ', chi2
    #print '   degrees of freedom = ', dof
    #print '   reduced chi squared = ', rchi2

    # The uncertainties are the square roots of the diagonal elements
    punc = np.zeros(len(popt))
    
    #sys.stdout.write("\n --> punc: " + str(punc) +
    #                 "\n --> pcov: " + str(pcov) +
    #                 "\n")
    #sys.stdout.flush()

    for i in np.arange(0,len(popt)):
        punc[i] = np.sqrt(pcov[i,i])
    return popt, punc, rchi2, dof


def powerlaw_fit(x, y):
    #mod = lmfit.models.PowerLawModel()
    mod = lmfit.models.LinearModel()
    ly = np.log10(y)
    lx = np.log10(x)
    pars = mod.guess(ly, x=lx)
    out = mod.fit(ly, pars, x=lx)
    return 10**pars['intercept'].value, pars['slope'].value
    #return pars['amplitude'].value, pars['exponent'].value

