#!/usr/bin/env python
#### ** Modeled after (or copied from???) code by Kasey Russell **

from __future__ import division
    
import numpy as np
import scipy as sp
from pylab import ion
from scipy import signal as sig
from scipy import optimize as opt
from scipy.interpolate import interp1d
#from scipy.io import loadmat
import matplotlib as mpl
from matplotlib.mlab import *
from matplotlib.pyplot import *
from matplotlib.widgets import MultiCursor
from matplotlib.ticker import EngFormatter
import re
import os
import sys   
import csv   
import argparse
import pprint
from itertools import cycle
import time            
    
from peak_finding import find_peaks_cwt
from constants import pi
from smartFormat import smartFormat, simpleFormat
import timeAndFreq as tf

# TODO: Constrained optimization, especially keeping frequency of peak
#       and amplitude sign (PEAK vs. NULL) within desired ranges so as to
#       truly optimize fit and get a rational result out.


def complLorentzian(freq, x0, beta, gamma, phi0):
    Y = 1j*phi0 + \
            beta/(-freq**2 + 1j*gamma*freq + x0**2)
    return Y 


def complResErr(params, freq, cVal):
    x0, beta, gamma, phi0 = params
    Y = complLorentzian(freq, x0, beta, gamma, phi0)
    err = Y - cVal 
    return np.abs(err)


def realLorentzian(freq, x0, beta, gamma, y0):
    #Y = beta * (gamma/2)/((freq-x0)**2 + (gamma/2)**2)
    #Y = (gamma/2)/((freq-x0)**2 + (gamma/2)**2)
    #Y = beta/(1+((freq-x0)*gamma/2)**2) + y0
    Y = (beta)/((freq-x0)**2 + (gamma/2)**2) + y0
    return Y


def realGaussian(freq, x0, beta, gamma, y0):
    #-- Gamma is FWHM
    #Y = beta * np.exp((freq-x0)**2/(gamma**2/8/np.log(2)))
    Y = np.exp((freq-x0)**2/(gamma**2/8/np.log(2))) + y0
    return Y


def realResErr(params, freq, amplVal):
    x0, beta, gamma, y0 = params
    Y = realLorentzian(freq, x0, beta, gamma, y0)
    #Y = realGaussian(freq, x0, beta, gamma, y00)
    err = Y - amplVal
    return abs(err)


def fitLorentzian(extremumInd, xCoords, yData, f0, gamma0, n=1,
                  peak=True, compl=True):
    xCoords = xCoords.astype(np.float_)
    yData = yData.astype(np.float_)
    f0 = xCoords[extremumInd]
    gamma0 = 0.0001*f0
    #trialLorentzian = realLorentzian(xCoords, f0, 1, gamma0)
    #beta0 = np.abs(yData[extremumInd]) / max(trialLorentzian) 
    beta0 = yData[extremumInd]
    beta0 = max(yData)
    phi0 = 0
    y00 = 0
    print "initial parameters", f0, beta0, gamma0
    if compl:
        params = [f0, beta0, gamma0, phi0]
        optout = opt.leastsq(complResErr, params, args=(xCoords, yData),
                             full_output=True)
        return optout

    params = [f0, beta0, gamma0, y00]
    #optout = opt.leastsq(realResErr, params, args=(xCoords, yData),
    #                     full_output=True)
    optout = opt.curve_fit(realLorentzian, xCoords, yData, p0=params)

    return optout


def realLorentziansPD(x, paramsDicts):
    if isinstance(paramsDicts, dict):
        pd = paramsDicts
        return realLorentzian(x, pd['x0'], pd['beta'], pd['gamma'], pd['y0'])

    y = np.zeros_like(x)
    for pd in paramsDicts:
        y += realLorentzian(x, pd['x0'], pd['beta'], pd['gamma'], pd['y0'])
    return y


def realLorentziansPL(x, *args, **kwargs):
    nParams = 4
    paramsList = list(args)[1:]
    paramsDicts = []
    for n in range(int(len(paramsList)/nParams)):
        paramsDicts.append(
            {'x0':    paramsList[nParams*n],
             'beta':  paramsList[nParams*n+1],
             'gamma': paramsList[nParams*n+2],
             'y0':    paramsList[nParams*n+3]}
        )
    return realLorentziansPD(x, paramsDicts)


def realLorentziansTemp(x, x0, beta, gamma, y0=0.0):
    freq = x
    #y0 = 0.0
    #x0 = 6197.0
    print 'x0', x0, 'beta', beta, 'gamma', gamma, 'y0', y0
    Y = (beta*(gamma/2)**2)/((freq-x0)**2 + (gamma/2)**2) + y0
    return Y


def fitLorentzians(xCoords, yData, initialGuessDicts, compl=False):
    if compl:
        nParams = 5
    else:
        nParams = 4

    #-- Make sure data types are floats s.t. bug in scipy doesn't rear its
    #   ugly head
    xCoords = xCoords.astype(np.float_)
    yData = yData.astype(np.float_)

    #if isinstance(initialGuessDicts, dict):
    #    initialGuessDicts = [initialGuessDicts]

    ##-- Unpack dictionary parameters into a list
    #params = []
    #for igd in initialGuessDicts:
    #    params.extend([igd['x0'], igd['beta'], igd['gamma'], igd['y0']])

    params = (initialGuessDicts['x0'], initialGuessDicts['beta'],
              initialGuessDicts['gamma'], initialGuessDicts['y0'])
    
    print 'igparams', params 
    #if compl:
    #    params = [f0, beta0, gamma0, phi0]
    #    optout = opt.leastsq(complResErr, params, args=(xCoords, yData),
    #                         full_output=True)
    #    return optout

    optout = opt.curve_fit(realLorentziansTemp, xCoords, yData, p0=params)
    #optout = opt.curve_fit(realLorentziansPL, xCoords, yData, p0=params)
    print 'optout', optout

    ##-- Re-pack dictionary parameters into list of dictionaries
    #n = 0
    #paramsList = optout[0]
    #for igd in initialGuessDicts:
    #    igd.update(
    #        {'x0': paramsList[n*nParams],
    #         'beta':  paramsList[n*nParams+1],
    #         'gamma': paramsList[n*nParams+2],
    #         'y0':    paramsList[n*nParams+3]}
    #    )

    optout = list(optout)
    optout[0] = initialGuessDicts

    return optout
