#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import with_statement

import os
import sys
import shutil
import traceback
import time
import re
import socket
import pprint
import numpy as np

from instrumentComms import InstrumentComms
from sr785Defs import *

def wstdout(s):
    sys.stdout.write(s)
    sys.stdout.flush()

def wstderr(s):
    sys.stderr.write(s)
    sys.stderr.flush()

PPstderr = pprint.PrettyPrinter(indent=2, width=80, depth=None,
                                stream=sys.stderr)

DEBUG = False

PAUSE_MEAS = False
WAIT_TO_COMPLETE = True

startTimeSec = time.time()

startTime = time.strftime("%Y-%m-%dT%H%M%S") + \
        "{0:+05d}".format(-int(round(time.timezone/3600)*100))

wstderr("\nScript start time: " + startTime + "\n")

#-- Instantiate FFT serial comms
SR785 = InstrumentComms(useSerial=True, useEthernet=False,
                      serCmdTerm='\r', serCmdPrefix='', argSep=',',
                      baud=19200, bytesize=8, parity='N', stopbits=1,
                      timeout=2, xonxoff=False, rtscts=False, dsrdtr=False)

#-- Find which serial port controls the SR785 (if this succeeds, comms work)
wstderr("\nFinding instrument...\n")
matches = SR785.findInstrument(
    model="SR785",
    searchSerial=True, searchEthernet=False, searchGPIB=False,
    idnQueryStr="OUTX RS232;*IDN?",
    debug=DEBUG)

if len(matches['serial']) == 0:
    raise Exception("Cannot find a serial port, or SR785 not attached to a serial port")

#-- Open the first matching serial port
SR785.openPort(matches['serial'][0])

idnStr = SR785.query("*IDN?")

# Log file...
homeDir = os.path.expanduser("~")
dataBaseDir = os.path.join(homeDir, "gibble", "data")
dataSubDir = os.path.join(dataBaseDir, time.strftime("%Y"),
                          time.strftime("%m-%b"),
                          time.strftime("%d"))

#-- Find existing run sub-directories and label this run accordingly
if os.path.isdir(dataSubDir):
    existingRunDirs = [ d for d in os.listdir(dataSubDir) if
                       os.path.isdir(os.path.join(dataSubDir,d)) ]
else:
    existingRunDirs = []
runDirRE = re.compile(r"^SR785run([0-9]+)")
runDirNumsTaken = [0]
for d in existingRunDirs:
    rd = runDirRE.findall(d)
    if len(rd) > 0:
        runDirNumsTaken.append(int(rd[0]))
runDirNum = max(runDirNumsTaken) + 1
runSubDir = "SR785run" + "{:04d}".format(runDirNum) + "-" + \
        time.strftime("%H%M")
dataDir = os.path.join(dataSubDir, runSubDir)
hostName = socket.gethostname()

filePrefix = "SR785SineSweep_"

baseFname = filePrefix + startTime

qualDataFname = os.path.join(dataDir, baseFname + "_data.csv")
qualDataFname0 = os.path.join(dataDir, baseFname + "_data_disp0.csv")
qualDataFname1 = os.path.join(dataDir, baseFname + "_data_disp1.csv")
qualMetaFname = os.path.join(dataDir, baseFname + "_meta.txt")

#-- Print out run number
wstdout("\nRUN " + "{:04d}".format(runDirNum) + "\n")

#-- Ask user for a description of the data
wstdout("\n" + "*"*80 + "\n")
description = raw_input(
    "Type a description of the data or <enter> to continue\n")
wstdout("-"*80 + "\n")

#-- Create directory if necessary (and if possible)
wstderr("\nCreating data directory structure...\n")
if os.path.exists(dataDir):
    createdDataDir = False
    if not os.path.isdir(dataDir):
        raise Exception("Wanted to create directory " + dataDir +
                        " but this path is a file.")
else:
    createdDataDir = True
    os.makedirs(dataDir, mode=0770)

wstderr("\nWriting meta data header...\n")
with file(qualMetaFname, 'w') as f:
    f.write("# dateTime = " + startTime + "\n")
    f.write("# description = " + description + "\n")
    f.write("# HOSTNAME = " + hostName + "\n")
    f.write("# dataPath = \"" + qualDataFname + "\"\n")
    #f.write("# dataPath0 = \"" + qualDataFname0 + "\"\n")
    #f.write("# dataPath1=\"" + qualDataFname1 + "\"\n")
    f.write("# metaPath = \"" + qualMetaFname + "\"\n")
    f.write("# IDN_SR785 = " + idnStr + "\n")
    #keys = metaDataDict.keys()
    #keys.sort()
    #for key in keys:
    #    f.write(key + "=" + str(metaDataDict[key]) + "\n")
    #f.write("completeConfig=" + completeConfigStr + "\n")

# To setup a measurement (will be useful for another piece of software):
#   1. Set Measurement Group
#   2. Set Measurement
#   3. Set View
#   4. Set Units
#   5. Set display scale and references

#-- Pause the measurement (has no effect if measurement is done)
swPausedMeas = False
instInitialStatus = interpretInstStatusWord(int(SR785.query("INST?")))
wstderr("\nInstrument status:\n" + PPstderr.pformat(instInitialStatus)+"\n\n")
#if instInitialStatus['STRT'] and not(instInitialStatus['PAUS']) and PAUSE_MEAS:
if instInitialStatus['STRT'] and not(instInitialStatus['PAUS']) and PAUSE_MEAS:
    wstderr("\nPausing current measurement...\n")
    SR785.tell("PAUS")
    swPausedMeas = True

measParams = {
    'display' : {},
    'frequency' : {},
    'sweptSineSrc' : {},
    'input' : {},
    'averaging' : {},
    'trigger' : {},
    'status' : instInitialStatus }

meas = {}

#-- Grab all meta-data pertinent to a swept sine measurement...
wstderr("\nQuerying all instrument parameters...\n")
DATA_TAKING_SUCCESS = True
try:
    #---------------------------------------------------------------------------
    # Display Setup
    #---------------------------------------------------------------------------
    measParams['display']['measGroup'] = measGroups[np.int(SR785.query("MGRP? 0"))]
    measParams['display']['measType0'] = measTypes[np.int(SR785.query("MEAS? 0"))]
    measParams['display']['measType1'] = measTypes[np.int(SR785.query("MEAS? 1"))]
    measParams['display']['view0']     = views[np.int(SR785.query("VIEW? 0"))]
    measParams['display']['view1']     = views[np.int(SR785.query("VIEW? 1"))]
    measParams['display']['unit0']     = SR785.query("UNIT? 0")
    measParams['display']['unit1']     = SR785.query("UNIT? 1")
    measParams['display']['undb0']     = undb[int(SR785.query("UNDB? 0"))]
    measParams['display']['undb1']     = undb[int(SR785.query("UNDB? 1"))]
    measParams['display']['unpk0']     = unpk[int(SR785.query("UNPK? 0"))]
    measParams['display']['unpk1']     = unpk[int(SR785.query("UNPK? 1"))]
    measParams['display']['psdu0']     = psdu[int(SR785.query("PSDU? 0"))]
    measParams['display']['psdu1']     = psdu[int(SR785.query("PSDU? 1"))]
    measParams['display']['unph0']     = unph[int(SR785.query("UNPH? 0"))]
    measParams['display']['unph1']     = unph[int(SR785.query("UNPH? 1"))]
    measParams['display']['dbmr']      = SR785.query("DBMR?")
    
    #---------------------------------------------------------------------------
    # Frequency
    #---------------------------------------------------------------------------
    if measParams['display']['measGroup'] == "Swept Sine":
        measParams['frequency']['start'] = np.float(SR785.query("SSTR? 0"))
        measParams['frequency']['stop'] = np.float(SR785.query("SSTP? 0"))
        measParams['frequency']['last'] = int(SR785.query("SSFR?"))
        measParams['frequency']['type'] = ssType[int(SR785.query("SSTY? 0"))]
        measParams['frequency']['autoRes'] = offOn[int(SR785.query("SARS? 0"))]
        measParams['frequency']['numPts'] = int(SR785.query("SNPS? 0"))
        if measParams['frequency']['autoRes'] == "On":
            measParams['frequency']['numSkipsMax']  = int(SR785.query("SSKP? 0")) 
            measParams['frequency']['fasterThresh'] = \
                    np.float(SR785.query("SFST? 0"))
            measParams['frequency']['slowerThresh'] = \
                    np.float(SR785.query("SSLO? 0"))

    #---------------------------------------------------------------------------
    # Swept Sine Source Parameters
    #---------------------------------------------------------------------------
    if measParams['display']['measGroup'] == "Swept Sine":
        measParams['sweptSineSrc']['autoRef'] = \
                autoRefSource[int(SR785.query("SSAL?"))]
        
        if measParams['sweptSineSrc']['autoRef'] == "Off":
            msg = SR785.query("SSAM?")
            y, i = msg.split(",")
            measParams['sweptSineSrc']['sineAmp'] = np.float(y)
            measParams['sweptSineSrc']['sineAmpUnits'] = ssUnits[int(i)]
        else:
            msg = SR785.query("SSRF?")
            y, i = msg.split(",")
            measParams['sweptSineSrc']['idealRefNumeric'] = np.float(y)
            measParams['sweptSineSrc']['idealRefUnits'] = ssUnits[int(i)]
            measParams['sweptSineSrc']['idealRefStr'] = \
                    y + " " + ssUnits[int(i)]
        
            measParams['sweptSineSrc']['ramp'] = \
                    offOn[int(SR785.query("SRMP?"))]
            if measParams['sweptSineSrc']['ramp'] == "On":
                measParams['sweptSineSrc']['rampRate'] = \
                        np.float(SR785.query("SRAT?"))
            measParams['sweptSineSrc']['refUpperLimit'] = \
                    np.float(SR785.query("SSUL?"))
            measParams['sweptSineSrc']['refLowerLimit'] = \
                    np.float(SR785.query("SSLL?"))
        
            msg = SR785.query("SMAX?")
            y, i = msg.split(",")
            measParams['sweptSineSrc']['maxNumeric'] = np.float(y)
            measParams['sweptSineSrc']['maxUnits'] = ssUnits[int(i)]
            measParams['sweptSineSrc']['maxStr'] = y + " " + ssUnits[int(i)]
        
            msg = SR785.query("SOFF?")
            y, i = msg.split(",")
            measParams['sweptSineSrc']['offsetNumeric'] = np.float(y)
            measParams['sweptSineSrc']['offsetUnits'] = ssUnits[int(i)]
            measParams['sweptSineSrc']['offsetStr'] = y + " " + ssUnits[int(i)]
    
    #---------------------------------------------------------------------------
    # Input Parameters
    #---------------------------------------------------------------------------
    measParams['input']['source'] = inputSources[int(SR785.query("ISRC?"))]
    measParams['input']['link'] = linkOptions[int(SR785.query("LINK?"))]
    measParams['input']['ch1Mode'] = inputModes[int(SR785.query("I1MD?"))]
    measParams['input']['ch2Mode'] = inputModes[int(SR785.query("I2MD?"))]
    
    measParams['input']['ch1Grounding'] = \
            groundingOpts[int(SR785.query("I1GD?"))]
    measParams['input']['ch2Grounding'] = \
            groundingOpts[int(SR785.query("I2GD?"))]
    
    measParams['input']['ch1Coupling'] = couplingOpts[int(SR785.query("I1CP?"))]
    measParams['input']['ch2Coupling'] = couplingOpts[int(SR785.query("I2CP?"))]
    
    msg = SR785.query("I1RG?")
    x, j = msg.split(",")
    measParams['input']['ch1RangeNumeric'] = np.float(x)
    measParams['input']['ch1RangeUnits'] = inputUnits[int(j)]
    measParams['input']['ch1RangeStr'] = x + " " + inputUnits[int(j)]
    
    msg = SR785.query("I2RG?")
    x, j = msg.split(",")
    measParams['input']['ch2RangeNumeric'] = np.float(x)
    measParams['input']['ch2RangeUnits'] = inputUnits[int(j)]
    measParams['input']['ch2RangeStr'] = x + " " + inputUnits[int(j)]
    
    measParams['input']['ch1AutoRanging'] = offOn[int(SR785.query("A1RG?"))]
    measParams['input']['ch2AutoRanging'] = offOn[int(SR785.query("A2RG?"))]
    
    if measParams['input']['ch1AutoRanging'] == "On":
        measParams['input']['ch1AutoRangeMode'] = \
                autoRangeModes[int(SR785.query("I1AR?"))]
    if measParams['input']['ch2AutoRanging'] == "On":
        measParams['input']['ch2AutoRangeMode'] = \
                autoRangeModes[int(SR785.query("I2AR?"))]
    
    measParams['input']['ch1AAFilt'] = offOn[int(SR785.query("I1AF?"))]
    measParams['input']['ch2AAFilt'] = offOn[int(SR785.query("I2AF?"))]
    
    measParams['input']['ch1A-WeightFilt'] = offOn[int(SR785.query("I1AW?"))]
    measParams['input']['ch2A-WeightFilt'] = offOn[int(SR785.query("I2AW?"))]
    
    measParams['input']['autoOffset'] = offOn[int(SR785.query("IAOM?"))]

    #---------------------------------------------------------------------------
    # Averaging Parameters
    #---------------------------------------------------------------------------
    if measParams['display']['measGroup'] == "Swept Sine":
        measParams['averaging']['settleTime'] = np.float(SR785.query("SSTM? 0"))
        measParams['averaging']['settleCycles'] = int(SR785.query("SSCY? 0"))
        
        measParams['averaging']['integrationTime'] = \
                np.float(SR785.query("SITM? 0"))
        measParams['averaging']['integrationCycles'] = \
                int(SR785.query("SICY? 0"))
    
    #---------------------------------------------------------------------------
    # Trigger Parameters
    #---------------------------------------------------------------------------
    measParams['trigger']['mode'] = triggerModes[int(SR785.query("TMOD?"))]
    measParams['trigger']['source'] = triggerSources[int(SR785.query("TSRC?"))]
    measParams['trigger']['sourceMode'] = \
            triggeredSourceModes[int(SR785.query("STMD?"))]
 
    #---------------------------------------------------------------------------
    # Wait for measurement to complete if user specified to do so
    #---------------------------------------------------------------------------
    if WAIT_TO_COMPLETE:
        waitStartTime = time.time()
        wstderr("Waiting for measurement to complete")
        N = measParams['frequency']['last'] + 1
        NTarget = measParams['frequency']['numPts']
        
        intentionallyPaused = instInitialStatus['PAUS'] or swPausedMeas
        if N < NTarget and not(intentionallyPaused):
            SR785.tell("CONT")

        while True:
            wstderr(".")
            #-- If the user has intentially paused intrument, don't wait
            #   for it to finish, because it might not ever! Note that the
            #   status word will NOT reliably report if the measurement is
            #   paused if it has been queried prior to the start of this
            #   software; in that case, we will force a "continue" command
            #   if the number of data points is less than the target number.
            if N >= NTarget:
                wstderr("\n --> Measurement completed.\n")
                break
            if intentionallyPaused:
                wstderr("\n --> It appears meas intentionally " +
                        "paused by user or this program; not waiting.\n")
                break
            time.sleep(5.0)
            measParams['frequency']['last'] = int(SR785.query("SSFR?"))
            N = measParams['frequency']['last'] + 1
        wstderr("\n")
        waitTime = time.time() - waitStartTime
        
    #---------------------------------------------------------------------------
    # Grab data from both channels,
    #---------------------------------------------------------------------------
    wstderr("Grabbing data from display 0...\n")
    meas['n0'] = int(SR785.query("DSPN? 0"))
    msg = SR785.query("DSPY? 0")
    strData = msg.split(",")
    meas['d0'] = msg
    
    wstderr("Grabbing data from display 1...\n")
    meas['n1'] = int(SR785.query("DSPN? 1"))
    msg = SR785.query("DSPY? 1")
    strData = msg.split(",")
    meas['d1'] = msg  

            
except Exception, err:
    DATA_TAKING_SUCCESS = False
    traceback.print_exc()
    try:
        if createdDataDir:
            shutil.rmtree(dataDir)
        else:
            os.remove(qualMetaFname)
    except:
        pass
    
finally:
    #-- If this program paused the measurement to record data, continue it now
    try:
        if swPausedMeas:
            wstderr("Resuming paused measurement...\n")
            SR785.tell("CONT")
    except:
        pass

#-- Save data to files
if DATA_TAKING_SUCCESS:
    #-- Number of data points taken
    N = measParams['frequency']['last']+1
    
    #-- Compute frequencies corresponding to data points; since data always
    #   comes back from the SR785 in ascending frequency order, regardless of
    #   how data was taken, re-define f0 and f1 if data taken hi-to-low
    f0 = min(measParams['frequency']['start'], measParams['frequency']['stop'])
    f1 = max(measParams['frequency']['start'], measParams['frequency']['stop'])
    nf = measParams['frequency']['numPts']

    freqs = [ format(freq, '0.6e') for freq in np.linspace(f0,f1,nf) ]

    wstderr("Writing metadata and data files...\n")
    with file(qualMetaFname, 'a') as f:
        keys0 = measParams.keys()
        keys0.sort()
        for k0 in keys0:
            keys1 = measParams[k0].keys()
            keys1.sort()
            for k1 in keys1:
                f.write("# " + k0 + "." + k1 + " = " +
                        str(measParams[k0][k1]) + "\n")
        f.write("# n0 = " + str(meas['n0']) + "\n")
        f.write("# n1 = " + str(meas['n1']) + "\n")
        f.flush()
        
    with file(qualDataFname, 'w') as f:
        data0 = meas['d0'].strip().split(',')
        data1 = meas['d1'].strip().split(',')
        data = [ ",".join([freq,d0,d1])+'\n' for (freq,d0,d1) in
                zip(freqs,data0,data1) ]
        f.write(",".join(["Hz",measParams['display']['unit0'],
                          measParams['display']['unit1']]) + "\n")
        f.writelines(data[0:N])
        f.flush()
        
    #with file(qualDataFname1, 'w') as f:
    #    data = meas['d1'].strip().split(',')
    #    data = [ d+'\n' for d in data ]
    #    f.writelines(data[0:N])
    #    f.flush()


    wstderr("\nNumber of data points recorded: " + str(N) + "\n")
    wstderr("\n")
    wstderr("  " + qualMetaFname + "\n")
    wstderr("  " + qualDataFname + "\n")
    #wstderr("  " + qualDataFname0 + "\n")
    #wstderr("  " + qualDataFname1 + "\n")
    wstderr("\n")

wstderr("Script run time: " + str(time.time()-startTimeSec) + " sec\n")
if WAIT_TO_COMPLETE:
    wstderr("Time spent waiting on measurement to complete: " + str(waitTime)
            + " sec\n\n")


##-- Clear status words
#SR785.talk("*CLS")
#
##-- Set the Standard Event enable register to catch EXE and CME
#SR785.talk("*ESE 48")
#
##-- Test comms
#idnStr = SR785.query("*IDN?")
#
##-- Reset to default state
#SR785.talk("*RST")
#
#
#SR785.talk("")
#SR785.talk("")
#
##-- Turn off key audible key clicks
#SR785.talk("KCLK OFF")
#
##-- Turn off alarm MESSAGES
#SR785.talk("ALRM OFF")
#
##-- Turn off alarm VOLUME
#SR785.talk("ALRT QUIET")
#
##-- Turn off "done" (either NOISY or OFF?)
#SR785.talk("ADON OFF")
#
##-- Turn off audible overload
#SR785.talk("AOVL OFF")

##-- Set frequency format to exact (vs. rounded)
#SR785.talk("FFMT EXACT")
#SR785.talk("FFMT ROUND")

