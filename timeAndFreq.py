#!/usr/bin/env python

from __future__ import division

import numpy as np
from scipy import signal as sig
import sys

#from constants import pi
#from smartFormat import *

def ADEV(y, dt, filterBW=None, confidenceInterval=1, noiseType=None,
         adevType='overlapped', nPts=0):
    '''Computes overlapped Allan deviation and confidence interval for
    discrete-sampled (but zero dead time) frequency measurements in numpy array
    'y' with sample period dt.

    Arguments:
    ----------
    y                   Sequence of frequency difference measurements in Hertz
                        (e.g., the difference between the oscillator under test
                        and a reference oscillator)

    dt                  Sampling period in seconds
    
    filterBW            Bandwidth of lowpass filter can be specified (in Hz) 
                        or else defaults to the Nyquist frequency, 1/(2*dt).

    confidenceInterval  Specifies the number of standard deviations at which to
                        place error bars about the sigma_y output.

    noiseType           The power-law noise type, where the following arguments
                        are valid: "random walk fm", "flicker fm", "white fm",
                        "flicker pm", or "white pm".
                        
                        If noiseType is not specified, default is "white fm".
                        
                        If an incorrect (or incomplete) description of the data
                        is due to the chosen noise type, this function makes
                        no attempt to correct the user or guess at the correct
                        data type. At least not yet.

    adevType            The type of Allan variance to compute: one of
                        'normal' or 'overlapped'

    nPts                Number of points to compute. If zero, compute ADEV for
                        all possible tau. If non-zero, compute this many points
                        spread out evenly on a logarithmic scale.

    Return values:
    --------------
    tau

    adev

    uncertainty

    '''
    #-- If no filter BW specified, assume filter used was Nyquist AA filter
    if filterBW == None:
        filterBW = 1/(2*dt)

    #-- Compute length of data and list of # of samples per averaging length
    #         N   = number of time difference data points avail
    #         M   = number of frequency diff data points avail for ADEV comp
    #               at a given tau
    #         m   = averaging factor
    #         tau = time corresponding to averaging factor
    M0 = np.float64(len(y))

    if nPts == 0:
        mList = np.int_(np.arange(1, np.floor(M0/2)))
    else:
        mList = np.int_(np.round(
            np.logspace(0, np.log10(np.floor(M0/2)),
                        num=nPts, endpoint=True, base=10.0) ))
        mList = np.unique(mList)

    tauList = mList * dt

    #-- Empty lists in which to put results
    avar = []

    if adevType == 'overlapped':
        for m in mList:
            #-- Create the Haar wavelet for the averaging length m
            filt = np.array([1]*m + [-1]*m, dtype=np.float64)
            
            #-- Convolve w/ data
            c = sig.fftconvolve(y, filt, mode='valid')

            #-- Allan variance is the sum-of-squares of differences
            avar.append( np.sum(c**2) / (2*m**2*(M0+1-2*m)) )

            #-- Compute variance-of-variance
            #var = [ np.mean(y[

    elif adevType == 'normal':
        for m in mList:
            M = np.floor(M0/m)
            #-- Create the Haar wavelet for the averaging length m
            filt = np.array([1]*m + [-1]*m, dtype=np.float64)
            
            #-- Convolve w/ data
            c = sig.fftconvolve(y, filt, mode='valid')

            #-- Pick only the non-overlapping convolution delays
            c = c[0:M0-m:m]

            #-- Allan variance is the sum-of-squares of differences
            avar.append( np.sum(c**2) / (2*m**2*(M-1)) )

            #-- Compute variance-of-variance
            #var = [ np.mean(y[

    avar = np.array(avar)
    #print avar
    adev = np.sqrt(avar)
    uncertainty = None

    return np.array(tauList), adev, uncertainty

if __name__ == "__main__":
    dt = 1
    tt = np.arange(0,9,dt)
    tx = np.array([0,43.6,89.7,121.6,163.7,208.4,248,289,319.8])
    ty = np.diff(tx)/1
    tau, adev, unc = ADEV(ty, dt)
    print 'overlapped adev:    ', adev
    tau, adev, unc = ADEV(ty, dt, adevType='normal')
    print 'non-overlapped adev:', adev
