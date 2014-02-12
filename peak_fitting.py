#!/usr/bin/env python

#### ** Some of below code is modeled after (or copied from???) code by Kasey
####    Russell

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


def structurallyDampedRes(params, angFreq):
    '''
    Model of a single mass-spring system with only structural (hysteretic)
    damping (i.e., no viscous damping)

    X-values are given by
        angFreq, omega

    Model parameters and the variables typically used for them are
        ampl0, A0 = F0/m
        resFreq, omega_0 = sqrt(k/m)
        qFactor, Q = 1/eta
    '''
    ampl0 = params['ampl0']
    resFreq = params['resFreq']
    qFactor = params['qFactor']

    #B = -2j*lossFactor*springConst*mass-2*springConst*mass+2*mass**2*angFreq**2
    #ampl = 4*force*mass**2 * (-1j*lossFactor*springConst -
    #                          springConst + mass*angFreq**2) / ( -B**2 )
    ampl = ampl0/(angFreq**2 - resFreq**2*(1-1j/qFactor))
    return ampl


def viscAndStructDampedRes(params, angFreq):
    '''
    Model of a single mass-spring-damper system with both viscous and
    structural (hysteretic) damping

    X-values are given by
        angFreq, omega

    Model parameters and the variables typically used for them are
        mass, m
        springConst, k
        lossFactor, eta
        viscousDamping, gamma
        force, F0

    '''
    mass = params['mass']
    springConst = params['springConst']
    lossFactor = params['lossFactor']
    viscousDamping = params['viscousDamping']
    force = params['force']

    A = viscousDamping*np.sqrt(
        viscousDamping**2 - 4j*lossFactor*springConst*mass - 4*springConst*mass)
    B = viscousDamping**2 - 2j*lossFactor*springConst*mass \
            - 2*springConst*mass + 2*mass**2*angFreq**2
    ampl = 4*force*mass**2 * ( -1j*lossFactor*springConst - springConst \
                              + mass*angFreq**2 - gamma*angFreq*(pi*1j/2) ) \
            / ( (A+B)*(A-B) )
    return ampl


def twoCoupledOscViscousDamping(params, omega):
    '''
    Model of two coupled mass-spring-damper systems, where there is no
    loss in the coupling term.

    X-values are given by 
        omega

    Model parameters are
        alpha0 -- nominal driving force
        r_alpha -- ratio of driving forces
        omega1 -- angular frequency of first resonance
        omega2 -- angular frequency of second resonance
        Q1 -- Q factor for first resonance
        Q2 -- Q factor for second resonance
        coupling -- strength of coupling between the two
        r_mass -- ratio of masses
    '''
    alpha0 = params['alpha0'].value
    r_alpha = params['r_alpha'].value
    omega1 = params['omega1'].value
    omega2 = params['omega2'].value
    Q1 = params['Q1'].value
    Q2 = params['Q2'].value
    coupling = params['coupling'].value
    r_mass = params['r_mass'].value
    #dc_offset = params['dc_offset'].value
   
    zeta1 = 1/(2*Q1)
    zeta2 = 1/(2*Q2)

    model = \
        (-( \
            ( \
                alpha0*( \
                    coupling*(-1+r_mass*r_alpha) \
                    + r_alpha*(-2*1j*zeta2*omega+omega**2-omega2**2) \
                ) \
            ) \
            / (  \
                (-2*1j* zeta1* omega + omega**2 - omega1**2) \
                * (-2*1j* zeta2 *omega + omega**2 - omega2**2) \
                + coupling*( \
                    omega*( \
                        -2*1j*(r_mass*zeta1+zeta2) \
                        + (1 + r_mass)*omega \
                    ) \
                    - r_mass*omega1**2 \
                    - omega2**2 \
                ) \
            ) \
        ) \
        + ( \
            1j*alpha0*( \
                coupling*(-1+r_mass*r_alpha) \
                + 2*1j*zeta1*omega - omega**2 + omega1**2 \
            ) \
        ) \
        / ( \
            (2*zeta1*omega + 1j*(omega - omega1)*(omega + omega1))  \
            * (-2*1j* zeta2*omega + omega**2 - omega2**2)  \
            + coupling*(  \
                omega*(2*(r_mass*zeta1 + zeta2) + 1j*(1+r_mass)*omega)  \
                - 1j*r_mass*omega1**2 - 1j*omega2**2  \
            )  \
          ))
    
    return model


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
