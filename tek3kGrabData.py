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
import argparse
import matplotlib.pyplot as plt

from tek3kComms import Tek3kComms

def wstdout(txt):
    sys.stdout.write(txt)
    sys.stdout.flush()

def wstderr(txt):
    sys.stderr.write(txt)
    sys.stderr.flush()


def main(description=""):
    scriptStartTime = time.time()

    parser = argparse.ArgumentParser(description=
                                     "Simple data grab from TEK 3k scope")
    parser.add_argument('--plot', dest='showPlot', action='store_true',
                        help="display a plot of the data to the user")
    parser.add_argument('--dont_save_plot', dest='dontSavePlot',
                        action='store_true',
                        help="overrides default behavior which savea a plot " +
                        "of the data in the data destination dir; note " +
                        "that you must specify --show_plot in order to " +
                        "actually see the plots at the time the script is " +
                        "run.")
    
    args = parser.parse_args()
    wstderr("Initializing script & scope...")

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
    runDirRE = re.compile(r"^run([0-9]+)")
    runDirNumsTaken = [0]
    for d in existingRunDirs:
        rd = runDirRE.findall(d)
        if len(rd) > 0:
            runDirNumsTaken.append(int(rd[0]))
    runDirNum = max(runDirNumsTaken) + 1
    runSubDir = "run" + '{:04d}'.format(runDirNum)
    dataDir = os.path.join(dataSubDir, runSubDir)
    hostName = socket.gethostbyaddr(socket.gethostname())[0]
    filePrefix = "scope_"

    #commsType = "Ethernet"
    ipAddress = "128.118.112.2"
    httpPort = 80

    try:
        #-- Set up with serial port comms
        tekScope = Tek3kComms(useSerial=True)
        tekScope.simpleSerial()
        commsType = "serial"
    except:
        #-- If that failed, try to set up with Ethernet comms
        tekScope = Tek3kComms(useEthernet=True, ipAddress=ipAddress)
        commsType = "Ethernet"

    completeConfigStr = tekScope.query(
        "HEADer ON;VERBose ON;*LRN?;HEADer OFF;VERBose OFF")

    #-- Turn off headers
    tekScope.tell("HEADer OFF")

    if commsType == "serial":
        tekScope.tell("RS232:TRANsmit:TERMinator LF;RS232:HARDFlagging OFF")
        #tekScope.tell("RS232:HARDFlagging OFF")

    #-- TODO: the following is good sometimes, but bad if you want the user
    #         to be able to force a trigger, for example... and maybe the
    #         software should force a trigger?
    ##-- Lock scope
    #tekScope.tell("LOCk ALL")

    idnStr = tekScope.query("*IDN?")

    wstderr(" done.\nWaiting for acquisition sequence to complete...")
    try:
        #-- Grab original scope acquisition parameters
        origAcqMode = tekScope.query("ACQuire:MODe?")
        origAcqStopAfter = tekScope.query("ACQuire:STOPAfter?")
        origAcqState = tekScope.query("ACQuire:STATE?")

        #-- Wait until a full acquisition completes
        #tekScope.tell("WAI")
        #print "busy:", tekScope.query("BUSY?")

        ##-- Set acquisition state if necessary
        #if "0" not in origAcqState:
        #    #-- If it was running...
        #    if origAcqStopAfter[0:3].upper() != "SEQ":
        #        #-- Set scope to stop after the next sequence completes
        #        #   (useful for really long data sets, but could be undesirable
        #        #   in other situations, so ymmv)
        #        tekScope.tell("ACQuire:STOPAfter SEQuence")
        ##if origAcqStopAfter[0:5].upper() != "RUNST":
        ##    tekScope.tell("ACQuire:STOPAfter RUNSTop")
        ##if "0" not in origAcqState:
        ##    tekScope.tell("ACQuire:STATE 0")

        #-- TODO: The following, when "Auto" triggering is used, waits until
        #         *TWO* sequences come in, which can be bad if time scale is
        #         set too long. In this case, the program should compute how
        #         much time a single sequence will take, and wait that much
        #         time OR just force an acquisition "right now". If the
        #         scope isn't already triggered, should we have the software
        #         be able to force trigger, if user wishes so?
        #-- Wait for a full acquisition (or series, in case of ENV or AVE)
        if "0" not in origAcqState:
            nAcqExpected = 1
            averaging = False
            envelope = False
            if origAcqMode[0:3].upper() == "AVE":
                averaging = True
                numAvg = int(tekScope.query("ACQuire:NUMAVg?"))
                nAcqExpected = numAvg
            elif origAcqMode[0:3].upper() == "ENV":
                envelope = True
                numEnv = int(tekScope.query("ACQuire:NUMEnv?"))
                nAcqExpected = numEnv
            reportedNumAcq = []
            while True:
                nAcq = int(tekScope.query("ACQuire:NUMACq?"))
                if nAcq not in reportedNumAcq:
                    wstderr("\n  Number of acq reported: " + str(nAcq) + " (of "
                            + str(nAcqExpected) + ")")
                    reportedNumAcq.append(nAcq)
                if nAcq >= nAcqExpected:
                    break
            wstderr("\n")
            tekScope.tell("ACQuire:STATE 0")

        #-- Set filename (and hence timestamp) to now
        timeNow = time.strftime("%Y-%m-%dT%H%M%S") + \
                "{0:+05d}".format(-int(round(time.timezone/3600)*100))
        baseFname = filePrefix + timeNow + "_"

        #-- Set data fomratting
        if commsType == "Ethernet":
            #-- must do this until I figure out how to get binary data via HTTP
            dataType = "ASCII"
            tekScope.tell("DATa:ENCdg ASCII;DATa:WIDth 2")
        else:
            dataType = "RIBinary"
            tekScope.tell("DATa:ENCdg RIBinary;DATa:WIDth 2")

        #-- Find out what channels are active
        activeChannels = tekScope.getActiveChannels()

        wstderr(" done.\n\n")

        data = []
        for channel in activeChannels:
            #-- Retrieve data from scope
            wstderr("Grabbing " + channel + " data...")
            t0 = time.time()
            xVec, yVec, metaDataDict = tekScope.grabWaveform(channel)
            t1 = time.time()
            data.append(metaDataDict)
  
            qualDataFname = os.path.join(dataDir,
                                         baseFname + channel + "_data.csv")
            qualMetaFname = os.path.join(dataDir,
                                         baseFname + channel + "_meta.txt")
            qualPlotPDFFname = os.path.join(dataDir,
                                         baseFname + "plots.pdf")
            qualPlotPNGFname = os.path.join(dataDir,
                                         baseFname + "plots.png")
            wstderr("  saving data to\n\"" + qualDataFname + "\"\n" +
                    "  saving metadata to\n\"" + qualMetaFname + "\"\n" )

            #-- Create directory if necessary (and if possible)
            if os.path.exists(dataDir):
                if not os.path.isdir(dataDir):
                    raise Exception("Wanted to create directory " + dataDir +
                                    " but this path is a file.")
            else:
                os.makedirs(dataDir, mode=0770)

            with open(qualDataFname, 'w') as f:
                c = csv.writer(f)
                for n in range(len(xVec)):
                    c.writerow([xVec[n], yVec[n]])
            with open(qualMetaFname, 'w') as f:
                f.write("dateTime=" + timeNow + "\r\n")
                f.write("description=" + description + "\r\n")
                f.write("HOSTNAME=" + hostName + "\r\n")
                f.write("dataPath=\"" + qualDataFname + "\"\r\n")
                f.write("metaPath=\"" + qualMetaFname + "\"\r\n")
                f.write("IDN=" + idnStr + "\r\n")
                keys = metaDataDict.keys()
                keys.sort()
                for key in keys:
                    f.write(key + "=" + str(metaDataDict[key]) + "\r\n")
                f.write("completeConfig=" + completeConfigStr + "\r\n")
            wstderr("\n")

        #-- Reset to original run state
        #if origAcqStopAfter != "RUNST":
        #    tekScope.tell("ACQuire:STOPAfter " + origAcqStopAfter)
        if "0" not in origAcqState:
            tekScope.tell("ACQuire:STATE " + origAcqState)

        scriptEndTime = time.time()
        wstderr("Total script time: " +
                str(round((scriptEndTime-scriptStartTime)*10)/10) + " s.\n")
    finally:
        tekScope.tell("LOCk 0")

    if not args.dontSavePlot or args.showPlot:
        fig = tekScope.plotMultiChan()
        if not args.dontSavePlot:
            fig.savefig(qualPlotPDFFname, format='pdf')
            fig.savefig(qualPlotPNGFname, format='png')
        if args.showPlot:
            plt.show()

if __name__ == "__main__":
    description = raw_input(
        "Type a description of the data or <enter> to continue\n")
    main(description)
