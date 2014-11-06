#!/usr/bin/env python

from __future__ import with_statement
from __future__ import division

import sys
import os
import time
import numpy as np
import re
import time
import csv
import socket
from matplotlib.mlab import *
from matplotlib.pyplot import *
from scipy import signal as sig
import matplotlib as mpl

from tek3kComms import Tek3kComms
from instrumentComms import InstrumentComms
from smartFormat import *

def wstdout(txt):
    sys.stdout.write(txt)
    sys.stdout.flush()

def wstderr(txt):
    sys.stderr.write(txt)
    sys.stderr.flush()

#==============================================================================
#-- Set some parameters of the system
#==============================================================================
#-- What anti-aliasing filters are applied at scope inputs?
errorChanAALPF = 1.6e3 # Hz
lockChanAALPF = 1.6e3 # Hz

#-- What is the desired loop bandwidth? (note that this can't be very high!)
loopBWDesired = 1.0 # Hz

#-- Define what scope channels are assigned to which function
#   (also valid to assign None to the channel for one or other lock)
scopeErrorChan = 1
scopeLock1Chan = 3
scopeLock2Chan = None

#-- Acceptable range of values to indicate lock...
lock1Range = [-0.001, 0.001]
lock2Range = [-0.001, 0.001]

#-- ... and acceptable RMS max to indicate lock
lock1RMS = [0.0001]
lock2RMS = [0.0001]

#==============================================================================
#-- Establish comms with scope
#==============================================================================
ipAddress = "128.118.112.2"
httpPort = 80

try:
    #-- Set up scope comms
    tekScope = Tek3kComms(useSerial=False, useEthernet=True,
                          ipAddress=ipAddress)
    scopeIDN = tekScope.query("*IDN?")
except:
    raise Exception("Expected Tek scope on Ethernet port. Aborting.")

#==============================================================================
#-- Establish comms with DDS
#==============================================================================
try:
    #-- Set up DDS comms
    DDS = InstrumentComms(useSerial=True, useEthernet=False, serCmdTerm='\n',
                          baud=115200, dsrdtr=True)
    DDS.simpleSerial()
    commsType = "serial"
    #-- Grab IDN string
    DDSIDN = DDS.query("*IDN?")
except:
    raise Exception("Expected DDS on serial port. Aborting.")

#==============================================================================
#-- Set scope channels, measurements
#==============================================================================

#==============================================================================
#-- Read initial DDS parameters
#==============================================================================

#==============================================================================
#-- Try to establish what the system gain is in V/Hz
#==============================================================================
freqScales = np.logspace(-6, +7, 14)

#==============================================================================
#-- Intialize control loop and vars by taking one sample
#==============================================================================
#-- Values for P, I, and D without the above process to estimate these
P = 100 # in units of Hz/V; crude measurement: 100 Hz |--> 100 mV
I = 15
D = 7
loopDelay = 0.3 # seconds
controlSign = -1
DCSetpoint = 0.0
gainEstimate = P

#-- Sampling window start time
sampTime11 = time.time()

#-- Grab sample from error measurement
sample = tekScope.query("MEASU:MEAS1:VAL?")

#-- Check if it is off-scale (high or low)
if sample == "9.9E37":
    raise Exception("Error signal is off-scale. Refusing to continue.")
else:
    sample = np.float64(sample)

error1 = sample - DCSetpoint

#-- Sampling window stop time
sampTime12 = time.time()

sampTime1 = np.mean([sampTime11,sampTime12])


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
runDirRE = re.compile(r"^PIDrun([0-9]+)")
runDirNumsTaken = [0]
for d in existingRunDirs:
    rd = runDirRE.findall(d)
    if len(rd) > 0:
        runDirNumsTaken.append(int(rd[0]))
runDirNum = max(runDirNumsTaken) + 1
runSubDir = "PIDrun" + '{:04d}'.format(runDirNum) + "-" + time.strftime("%H%M")
dataDir = os.path.join(dataSubDir, runSubDir)
hostName = socket.gethostbyaddr(socket.gethostname())[0]

filePrefix = "PID_"
timeNow = time.strftime("%Y-%m-%dT%H%M%S") + \
        "{0:+05d}".format(-int(round(time.timezone/3600)*100))

baseFname = filePrefix + timeNow

qualDataFname = os.path.join(dataDir, baseFname + "_data.csv")
qualMetaFname = os.path.join(dataDir, baseFname + "_meta.txt")

#description = raw_input(
#    "Type a description of the data or <enter> to continue\n")

description = "PID controller"

#-- Create directory if necessary (and if possible)
if os.path.exists(dataDir):
    if not os.path.isdir(dataDir):
        raise Exception("Wanted to create directory " + dataDir +
                        " but this path is a file.")
else:
    os.makedirs(dataDir, mode=0770)

with open(qualMetaFname, 'w') as f:
    f.write("dateTime=" + timeNow + "\r\n")
    f.write("description=" + description + "\r\n")
    f.write("HOSTNAME=" + hostName + "\r\n")
    f.write("dataPath=\"" + qualDataFname + "\"\r\n")
    f.write("metaPath=\"" + qualMetaFname + "\"\r\n")
    f.write("IDN_SCOPE=" + scopeIDN + "\r\n")
    f.write("IDN_DDS=" + DDSIDN + "\r\n")
    #keys = metaDataDict.keys()
    #keys.sort()
    #for key in keys:
    #    f.write(key + "=" + str(metaDataDict[key]) + "\r\n")
    #f.write("completeConfig=" + completeConfigStr + "\r\n")

integral = np.float64(0)
iterNum = np.uint64(1)
try:
    # TODO: does this actually tell you current freq when in sweep mode?
    # if so, great, that's the desired behavior.
    # TODO: Make sure that nothing I do changes the ref sync behavior
    #-- Grab initial DDS frequency
    freqStr = DDS.query("FREQ?")
    oldVal = np.float64(freqStr)
    #-- Set DDS in sine mode, if not already so
    DDS.tell("AM:STATe OFF;:FM:STATe OFF;:FSKey:STATe OFF;:SWEep:STATe OFF;:BURSt:STATe OFF")
    time.sleep(0.1)
    #DDS.tell("FUNC SIN")
    #DDS.tell("FREQ " + freqStr)
    with open(qualDataFname, 'w') as f:
        while True:
            c = csv.writer(f)
            
            #-- Beginning of time window 
            sampTime21 = time.time()
            
            #-- Grab sample from error measurement
            sample = tekScope.query("MEASU:MEAS1:VAL?")
            
            #-- Check if it is off-scale (high or low)
            if sample == "9.9E37":
                raise Exception(
                    "Error signal is off-scale. Refusing to continue.")
            else:
                sample = np.float64(sample)
            
            error2 = sample - DCSetpoint
    
            #-- End of time window 
            sampTime22 = time.time()
    
            sampTime2 = np.mean([sampTime21,sampTime22])
    
            dt = sampTime2 - sampTime1
    
            #-- Trapezoid area = (1/2)*dt*(error1+error2)
            integral += 0.5*dt*(error1+error2)
    
            #-- Compute derivative
            deriv = (error2-error1)/dt
    
            #-- Projected values by the time we can apply a control signal
            projectedValue = deriv*loopDelay + error2
            projectedIntegral = \
                    integral + 0.5*loopDelay*(error2+projectedValue)
    
            #-- Compute control signal and new value
            correction = P*projectedValue + I*projectedIntegral + D*deriv
            newVal = np.round(oldVal + controlSign*correction, 6)
    
            #-- Beginning of set time window 
            setTime1 = time.time()
    
            #-- Set value
            formattedNum = simpleFormat(newVal, sigFigs=14,
                                        keepAllSigFigs=False).upper()
            #newVal = np.float64(DDS.query("FREQ " + formattedNum + ";FREQ?"))
            DDS.tell("FREQ " + formattedNum)
    
            #-- End of set time window 
            setTime2 = time.time()
    
            setTime = np.mean([setTime1,setTime2])
    
            #-- Update estimate of loop delay
            loopDelay = 0.9*loopDelay+0.1*(setTime-sampTime2)
    
            #-- Update estimate of loop delay
            df = newVal - oldVal
            if iterNum == 1:
                meanDriftEst = df/dt
            meanDriftEst = 0.9*meanDriftEst + 0.1*(df/dt)
    
            #-- Write results
            row = [sampTime21,
                   sampTime22,
                   sampTime2,
                   setTime1,
                   setTime2,
                   setTime,
                   error2,
                   integral,
                   deriv,
                   correction,
                   newVal,
                   loopDelay,
                   gainEstimate,
                   meanDriftEst]
            c.writerow(row)
    
    
            if iterNum % 30 == 0:
                wstderr(time.strftime("%Y-%m-%d %H:%M:%S") + " est drift = " +
                        simpleFormat(meanDriftEst,
                                     sigFigs=5, keepAllSigFigs=True) +
                        " Hz/s\n")
            iterNum += 1
            #wstdout("sampTime1="+str(sampTime1)+"\n"+
            #        "sampTime21="+str(sampTime21)+"\n"+
            #        "sampTime22="+str(sampTime22)+"\n"+
            #        "sampTime2 ="+str(sampTime2)+"\n"+
            #        "dt="+str(dt)+"\n"+
            #        "error1="+str(error1)+"\n"+
            #        "error2="+str(error2)+"\n"+
            #        "integral="+str(integral)+"\n"+
            #        "deriv="+str(deriv)+"\n"+
            #        "projectedValue="+str(projectedValue)+"\n"+
            #        "projectedIntegral="+str(projectedIntegral)+"\n"+
            #        "correction="+str(correction)+"\n"+
            #        "oldVal="+str(oldVal)+"\n"+
            #        "newVal="+str(newVal)+"\n"+
            #        "loopDelay="+str(loopDelay)+"\n"+
            #        "-"*78 + "\n")
                    
            #-- Update 1's with 2's
            oldVal = newVal
            error1 = error2
            sampTime1 = sampTime2
except Exception, e:
    #-- Give control of DDS back to user!
    DDS.tell("SYSTem:LOCal")
    raise e
