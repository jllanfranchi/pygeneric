#!/usr/bin/env python

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
from matplotlib.widgets import Slider, Button, RadioButtons
from matplotlib.ticker import EngFormatter
import re
import os
import sys
import csv
import argparse
import pprint
from itertools import cycle
import time

#from peak_finding import find_peaks_cwt
from peak_fitting import fitLorentzians, realLorentziansPD, realLorentziansTemp
from constants import pi
from smartFormat import smartFormat, simpleFormat
import timeAndFreq as tf
from genericUtils import findFiles, wstdout, wstderr

#ion()

parser = argparse.ArgumentParser(
    description="Fit SR785 data acquired by sr785GrabSineSweep.py script in directories specified by user (or search from current working directory).")
parser.add_argument('searchDirs', metavar='dir', type=str, nargs='*',
                    help='Directory or directories in which to search for data',
                    default=os.getcwd())
parser.add_argument('-f', '--force_plot', action='store_true',
                    help='Forces plotting despite a "DO_NOT_PLOT" file in directory')
parser.add_argument('-s', '--solid_lines', action='store_true',
                    help='Plots all solid rather than cycled-style lines.')
args = parser.parse_args()
if isinstance(args.searchDirs, str):
    searchDirs = [args.searchDirs]
else:
    searchDirs = args.searchDirs


COLOR_SCHEME = 'k_on_w'
LINE_SCHEME = 'screen' #-- 'screen' or 'print'
PHASE_UNWR = False
FIND_PEAKS = False
PLOT = True
PLOT_PEAKS = False
FIND_LINEWIDTHS = False
PLOT_COMSOL_DATA = False
PLOT_LINEWIDTHS = False
ANNOTATE_Q = False

mpl.rcParams['font.size'] = 10

#cm = get_cmap('RdBu')
#cm = get_cmap('autumn')
#cm = get_cmap('Set1')
#cm = get_cmap('Paired')
#cm = get_cmap('Dark2')
#cm = get_cmap('Set3')
#cm = get_cmap('hsv')

#-- Light grey on dark grey
if COLOR_SCHEME == 'lg_on_dg':
    lightCol = (.7,.7,.7)
    darkCol0 = (.05,.05,.05)
    darkCol1 = (.15,.15,.15)
    majorGridLineCol = (0.5,0.5,0.5)
    minorGridLineCol = (0.2,0.2,0.2)
    cm = get_cmap('hsv')

#-- White on black
elif COLOR_SCHEME == 'w_on_k':
    lightCol = (1,1,1)
    darkCol0 = (0,0,0)
    darkCol1 = (0,0,0)
    majorGridLineCol = (0.5,0.5,0.5)
    minorGridLineCol = (0.1,0.1,0.1)
    cm = get_cmap('hsv')

#-- Black on white
#if COLOR_SCHEME == 'k_on_w':
else:
    lightCol = (0,0,0)
    darkCol0 = (1,1,1)
    darkCol1 = (1,1,1)
    majorGridLineCol = (0.4,0.4,0.4)
    minorGridLineCol = (0.9,0.9,0.9)
    #cm = get_cmap('hsv')
    cm = get_cmap('spectral')
    #cm = get_cmap('prism')
    #cm = get_cmap('Paired')
    #cm = get_cmap('Set1')

mpl.rcParams['text.color'] = lightCol
mpl.rcParams['axes.edgecolor'] = lightCol
mpl.rcParams['axes.labelcolor'] = lightCol
mpl.rcParams['xtick.color'] = lightCol
mpl.rcParams['ytick.color'] = lightCol
mpl.rcParams['grid.color'] = lightCol
mpl.rcParams['patch.edgecolor'] = lightCol

if args.solid_lines:
    lines = ["-"]
else:
    lines = ["-","--","-.",":"]
linecycler = cycle(lines)
if LINE_SCHEME == 'screen':
    stdLW = 1.0
else:
    stdLW = 0.5

fnameInfoRe = re.compile(r"SR785run([0-9]{4}).*SR785SineSweep_" +
                         r"([0-9]{4})-([0-9]{2})-([0-9]{2})T" +
                         r"([0-9]{2})([0-9]{2})")

fnames = []
for searchDir in searchDirs:
    fnames.extend([ f for f in findFiles(searchDir,
                                        r"^SR785SineSweep.*_data.csv$") ])
#-- Remove duplicates
fnames = set(fnames)
#-- Sort alphabetically by file name
fnames = list(fnames)
fnames.sort()

if not args.force_plot:
    toRemove = []
    for fname in fnames:
        root = fname[0]
        if os.path.exists(os.path.join(root, "DO_NOT_PLOT")):
            toRemove.append(fname)
    
    for fname in toRemove:
        fnames.remove(fname)

if len(fnames) == 0:
    raise Exception("NO FILES WERE FOUND TO PLOT.\n" +
                    "  Either none are present, or override DO_NOT_PLOT " +
                    "files with -f.")

pprint.pprint(fnames)

NUM_COLORS = int(round(len(fnames)*1.0))

fig2 = figure(2, figsize=(10.5,8), dpi=80, facecolor=darkCol1,
              edgecolor=lightCol, linewidth=0, frameon=True)
clf()

ax3 = fig2.add_subplot(211, axisbg=darkCol0)
ylabel("Gain (V/V)")

ax4 = fig2.add_subplot(212, axisbg=darkCol0, sharex=ax3)
ylabel("Phase (deg)")

ax3.set_color_cycle([cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS)])
ax4.set_color_cycle([cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS)])
xlabel("Frequency (Hz)")


fig1 = figure(1, figsize=(10.5,8), dpi=80, facecolor=darkCol1,
              edgecolor=lightCol, linewidth=0, frameon=True)
clf()

ax1 = fig1.add_subplot(211, axisbg=darkCol0)
ylabel("Magnitude (dB)")

ax2 = fig1.add_subplot(212, axisbg=darkCol0, sharex=ax1)
ylabel("Phase (deg)")

ax1.set_color_cycle([cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS)])
ax2.set_color_cycle([cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS)])
xlabel("Frequency (Hz)")

allData = []
for fname in fnames:
    fullFname = os.path.join(*fname)
    (run, year, month, day, hour, minute) = fnameInfoRe.findall(fullFname)[0]
    #label = month + "-" + day + "-" + year[-2:] + "  " + hour + ":" + minute + \
    #        str.rjust("r"+str(int(run)), 6, " ")
    label = year[-2:] + "-" + month + "-" + day + "  " + hour + ":" + minute + \
            str.rjust("r"+str(int(run)), 6, " ")
    rec = csv2rec(fullFname, comments="#", delimiter=',')
    allData.append( {'freq':rec['hz'],
                     'mag':rec['db'],
                     'phase':rec['deg'],
                     'label':label} )

fMin = 6100
fMax = 6300
fMin = 23000
fMax = 27000
freqData = []
magData = []
phaseData = []
for setN in range(len(allData)):
    data = allData[setN]
    
    #-- Compute unwrapped phase
    phaseUnwr = np.unwrap(data['phase'],discont=45)

    #-- Find indices in relevant freq range
    indices = find((data['freq'] >= fMin) * (data['freq'] <= fMax))

    #-- Extract relevant data
    freqData.extend(list(data['freq'][indices]))
    magData.extend(list(data['mag'][indices]))
    #phaseData.extend(list(data['phase'][indices]))
    phaseData.extend(list(phaseUnwr[indices]))

    #ax1.plot(data['freq'], data['mag'], linestyle, label=data['label'],
    #         linewidth=stdLW, marker='', markersize=0)
    #ax2.plot(data['freq'], data['phase'], linestyle,
    #         label=data['label'],
    #         linewidth=stdLW)

freqData = np.array(freqData)
magData = np.array(magData)
gainData = 10**(magData/20.0)
#gainData = gainData - min(gainData)
phaseData = np.array(phaseData)
xData = gainData*np.cos(phaseData*np.pi/180.)
yData = gainData*np.sin(phaseData*np.pi/180.)
complData = gainData*np.exp(1j*phaseData*np.pi/180.)

ax1.plot(freqData, magData, linestyle='-', linewidth=0.5,
         marker='o', markersize=6)
ax2.plot(freqData, phaseData, linestyle='-', linewidth=0.5,
         marker='o', markersize=6)

ax3.plot(freqData, gainData, linestyle='-', linewidth=0.5,
         marker='o', markersize=6)
ax4.plot(freqData, phaseData, linestyle='-', linewidth=0.5,
         marker='o', markersize=6)

nLor = 2
ig1 = {'x0': 6197, 'beta': 2.1, 'gamma': 20.36, 'y0': 0.1}
ig2 = {'x0': 6428, 'beta': -0.0005, 'gamma': 6428/2000, 'y0': 0.00}
ig = [ig1, ig2]
ig = ig1

#optout, opterr = fitLorentzians(freqData, gainData, ig, compl=False)
optout, opterr = fitLorentzians(freqData, gainData, ig, compl=False)
x0 = optout['x0']
beta = optout['beta']
gamma = optout['gamma']
y0 = optout['y0']
ax1.plot(freqData, 20*np.log10(realLorentziansTemp(freqData, x0,
                                                   beta, gamma, y0)), linestyle='-', linewidth=0.5,
         marker=None, markersize=2)
ax3.plot(freqData, (realLorentziansTemp(freqData, x0, beta, gamma, y0)), linestyle='-', linewidth=0.5,
         marker=None, markersize=2)

#f2 = figure(2)
#f2.clf()
#ax = subplot(111)
#subplots_adjust(bottom=0.4)
#l0, = plot(freqData, gainData, 'ko')
#l1, = plot(freqData, realLorentziansPD(freqData, ig1), 'b-')
#axcolor = 'lightgoldenrodyellow'
#ax_x01 = axes([0.25, 0.1, 0.65, 0.03], axisbg=axcolor)
#ax_b1 = axes([0.25, 0.15, 0.65, 0.03], axisbg=axcolor)
#ax_g1 = axes([0.25, 0.20, 0.65, 0.03], axisbg=axcolor)
#ax_y01 = axes([0.25, 0.25, 0.65, 0.03], axisbg=axcolor)
#
#
#sx01 = Slider(ax_x01, 'x01', 6190, 6200, valinit=6197)
#sb1  = Slider(ax_b1, 'b1', 0, 500, valinit=33)
#sg1  = Slider(ax_g1, 'g1', 1, 50, valinit=8.4)
#sy01 = Slider(ax_y01, 'y01', 0, 1, valinit=0)
#
#def update(val):
#    x01 = sx01.val
#    b1 = sb1.val
#    g1 = sg1.val
#    y01 = sy01.val
#    igx = {'x0':x01,'beta':b1,'gamma':g1,'y0':y01}
#    l1.set_ydata(realLorentziansPD(freqData, igx))
#    draw()
#
#sx01.on_changed(update)
#sb1.on_changed(update)
#sg1.on_changed(update)
#sy01.on_changed(update)

show()
