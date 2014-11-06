#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import with_statement
from __future__ import division

import sys
import os
import time
import glob
import serial
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import struct
import re
import time
import urllib2
import urllib
import urlparse
from smartFormat import *

from instrumentComms import *

if os.name is 'nt':
    import scanwin32

if os.name is 'posix':
    pass

def wstdout(s):
    sys.stdout.write(s)
    sys.stdout.flush()

def wstderr(s):
    sys.stderr.write(s)
    sys.stderr.flush()

#===============================================================================
# Module exception definitions
#===============================================================================

#===============================================================================
# Conversions convenient for module
#===============================================================================
def interpretWFMPREString(s):
    sDict = {}
    s = s.split(";")
    sDict = {
        'byteNum'   : int(s[0]), # number of bytes per datum
        'bitNum'    : int(s[1]), # number of bits per datum
        'encoding'  : s[2],  # one of ASC or BIN
        'binFmt'    : s[3],   # one of RI or RP
        'byteOrder' : s[4], # one of MSB, LSB
        'numPts'    : int(s[5]), # number of data points
        'WFID'      : s[6], # string describing scope params
        'ptFmt'     : s[7], # one of ENV or Y (?)
        'xIncr'     : np.float64(s[8]), # x increment (floating point value)
        'xPtOffset' : np.float64(s[9]),
        'xZero'     : np.float64(s[10]),
        'xUnit'     : s[11].replace('"', ''),
        'yMult'     : np.float64(s[12]),
        'yZero'     : np.float64(s[13]),
        'yOffset'   : np.float64(s[14]),
        'yUnit'     : s[15].replace('"', '')  
    }
    return sDict

def interpretRawData(bd, **kwargs):
    encoding = kwargs['encoding']
    binFmt = kwargs['binFmt']
    byteNum = kwargs['byteNum']
    byteOrder = kwargs['byteOrder']
    yOffset = kwargs['yOffset']
    numPts = kwargs['numPts']
    yZero = kwargs['yZero']
    yMult = kwargs['yMult']
    xZero = kwargs['xZero']
    xIncr = kwargs['xIncr']
    xPtOffset = kwargs['xPtOffset']

    if encoding == 'BIN':
        if byteOrder == "MSB":
            fmt0 = ">"
        else:
            fmt0 = "<"

        if (binFmt == "RP" or binFmt == "SRP") and byteNum == 2:
            fmt1 = "h"
        elif (binFmt == "RI" or binFmt == "SRI") and byteNum == 2:
            fmt1 = "H"
        elif (binFmt == "RP" or binFmt == "SRP") and byteNum == 1:
            fmt1 = "b"
        elif (binFmt == "RI" or binFmt == "SRI") and byteNum == 1:
            fmt1 = "B"
        
        rawSamples = np.array([ struct.unpack(fmt0+fmt1, bd[n:n+byteNum])[0]
                    for n in range(0,len(bd),byteNum) ],
                           dtype=np.float64)

    if encoding == 'ASC':
        rawSamples = np.array(bd.split(','), dtype=np.float64)

    samples = yZero + yMult*(rawSamples-yOffset)

    t = xZero + xIncr*(np.arange(0,numPts)-xPtOffset)

    return t, samples

class Tek3kComms(InstrumentComms):
    def __init__(self, useSerial=True, useEthernet=True,
                 ipAddress="0.0.0.0", ethPort=80,
                 serialPortName=None,
                 baud=38400, bytesize=8, parity='N', stopbits=1,
                 timeout=2, xonxoff=False, rtscts=False, dsrdtr=True,
                 serCmdPrefix="", serCmdTerm='\n', serIdnCmd='*IDN?',
                 ethCmdPrefix='COMMAND=',ethCmdTerm='\r\ngpibsend=Send\r\n',
                 ethDataRE=re.compile('.*<TEXTAREA.*>(.*)</TEXTAREA>'),
                 ethIdnCmd='*IDN?',
                 argSep=""):
        super(Tek3kComms, self).__init__(
            useSerial=useSerial, useEthernet=useEthernet,
            ipAddress=ipAddress, ethPort=ethPort,
            serialPortName=serialPortName,
            baud=baud, bytesize=bytesize, parity=parity, stopbits=stopbits,
            timeout=timeout, xonxoff=xonxoff, rtscts=rtscts, dsrdtr=dsrdtr,
            serCmdPrefix=serCmdPrefix, serCmdTerm=serCmdTerm,
            serIdnCmd=serIdnCmd,
            ethCmdPrefix=ethCmdPrefix, ethCmdTerm=ethCmdTerm,
            ethDataRE=ethDataRE, ethIdnCmd=ethIdnCmd,
            argSep=argSep)

    def __ethernetQuery(self, command, returnResponse=True, upperCaseOnly=True):
        url = 'http://' + self.scopeIP + '/Comm.html' #+ str(self.scopePort)
        
        if upperCaseOnly:
            command2 = self.lowerRegex.sub('', command)
        else:
            command2 = command

        #wstderr("\""+ command2 +"\"\n")

        httpPostSendStr = self.ETHCMD_PREFIX + command2 + self.ETHCMD_TERM
        
        fullRequest = urllib2.Request(url, httpPostSendStr)
        cnxn = urllib2.urlopen(fullRequest)
        httpPostReturnStr = cnxn.read()

        #wstdout('-'*80 + '\n' + str(httpPostSendStr) + '\n')
        #wstdout(str(httpPostReturnStr) + '\n' )

        if returnResponse:
            response = self.ETHDATA_RE.findall(httpPostReturnStr)[0]
            #wstdout(str(response) + '-'*80 + '\n')
            return response
        else:#
            pass
            #wstdout('-'*80 + '\n')

    def __ethernetBinaryQuery(self, command, returnResponse=True):
        """??? -- Haven't figured out how to do binary data via HTTP post;
        possibly if I used a different command altogether (see web
        interface, as they have a data download function there)"""
        url = 'http://' + self.scopeIP + '/Comm.html' #+ str(self.scopePort)
        s = self.ETHCMD_PREFIX + command + self.ETHCMD_TERM
        httpPostSendStr = s
        
        fullRequest = urllib2.Request(url, httpPostSendStr)
        cnxn = urllib2.urlopen(fullRequest)
        httpPostReturnStr = cnxn.read()
        #httpPostReturnStr += cnxn.read()
        #httpPostReturnStr += cnxn.read()

        #wstdout('-'*80 + '\n' + str(httpPostSendStr) + '\n')
        #wstdout(str(httpPostReturnStr) + '\n' )

        if returnResponse:
            response = self.ETHDATA_RE.findall(httpPostReturnStr)[0]
            #wstdout(str(response) + '-'*80 + '\n')
            return response
        else:
            pass
            #wstdout('-'*80 + '\n')

    def getActiveChannels(self):
        selected = self.query("SELect?").split(';')
        self.config['activeChannels'] = []
        if "CH" in selected:
            #-- This means verbose mode is on
            rxEnabled = re.compile(r"(CH|REF|MATH)([1-4]) [1]",
                                   flags=re.IGNORECASE)
            for s in selected:
                r = rxEnabled.findall(line)
                if len(r) > 0:
                    self.config['activeChannels'].append( ''.join(r[0]) )
        else:
            nCh = 4
            nMath = 1
            nRef = 4
            for n in range(nCh):
                if selected[n] == '1':
                    self.config['activeChannels'].append('CH' + str(n+1))
            for n in range(nMath):
                if selected[nCh+n] == '1':
                    self.config['activeChannels'].append('MATH')
            for n in range(nRef):
                if selected[nCh+nMath+n] == '1':
                    self.config['activeChannels'].append('REF' + str(n))
        return self.config['activeChannels']

    def grabWaveform(self, channel):
        self.tell("DATa:SOUrce " + channel)
        waveformMetaData = self.query("WFMPre?")
        waveform = self.binQuery("CURVe?")
        metaDataDict = interpretWFMPREString(waveformMetaData)
        xVec, yVec = interpretRawData(waveform, **metaDataDict)
        self.data[channel] = {}
        self.data[channel]['rawdata'] = waveform
        self.data[channel]['x'] = xVec
        self.data[channel]['y'] = yVec
        #-- Now append ALL the metadata to the channel's dictionary
        self.data[channel].update(metaDataDict)
        return xVec, yVec, metaDataDict

    def plotMultiChan(self, channels=None):
        mpl.rcParams['font.size'] = 7
        if channels == None:
            channels = self.config['activeChannels']
        if not (isinstance(channels, list) or isinstance(channels, tuple)):
            channels = [channels]

        nChannels = len(channels)
        fig = plt.figure(figsize=(8,nChannels*2),dpi=80)
        #ax = plt.axes(axisbg='k')
        #colors = { \
        #    'CH1' : (1,1,.1),
        #    'CH2' : (.1,1,1),
        #    'CH3' : (1,.1,1),
        #    'CH4' : (.1,1,.1),
        #    'MATH': (1,.05,.05),
        #    'MATH1': 'r',
        #    'REF1': 'w',
        #    'REF2': (.5,.5,.5),
        #    'REF3': (.8,.4,.3),
        #    'REF4': (.3,.4,.8) }
        colors = { \
            'CH1' : (.6,.6,0),
            'CH2' : (0,0.6,0.6),
            'CH3' : (0.8,0.2,0.5),
            'CH4' : (0,0.6,0),
            'MATH': 'k',
            'MATH1': 'k',
            'REF1': (.6,.6,.2),
            'REF2': (.3,.3,.3),
            'REF3': (.6,.4,.3),
            'REF4': (.3,.4,.6) }
        linestyles = { \
            'CH1' : '-',
            'CH2' : '--',
            'CH3' : '-.',
            'CH4' : ':',
            'MATH': '-',
            'MATH1': 'r',
            'REF1': 'w',
            'REF2': (.5,.5,.5),
            'REF3': (.8,.4,.3),
            'REF4': (.3,.4,.8) }
        allAxes = []
        n = 0
        for channel in channels:
            try:
                x = self.data[channel]['x']
                y = self.data[channel]['y']
            except:
                self.retrieveSingleWaveform(channel)

            #print channel, len(x), len(y), x[0], x[-1], y[0], y[-1]
            #print x[0:10], y[0:10]
            #print x[-11:-1], y[-11:-1]
            xUnit = self.data[channel]['xUnit']
            yUnit = self.data[channel]['yUnit']
            if n == 0:
                ax = fig.add_subplot(nChannels,1,n+1, axisbg='w')
            else:
                ax = fig.add_subplot(nChannels,1,n+1, axisbg='w',
                                     sharex=allAxes[0])
            allAxes.append(ax)

            ax.plot(x,y, color=colors[channel.upper()], label=channel,
                   linewidth=0.75)
            plt.xlabel(xUnit)
            plt.ylabel(yUnit)
            ax.hold(True)
            n += 1
            plt.title(channel)
            ax.grid(True, color=(.25,.25,.25))
            plt.xlim((min(x), max(x)))
        ax.hold(False)
        if nChannels == 1:
            fig.subplots_adjust(top=0.85, bottom=0.2)
        elif nChannels == 2:
            fig.subplots_adjust(hspace=0.4, top=0.915)
        elif nChannels == 3:
            fig.subplots_adjust(hspace=0.425, top=0.925)
        elif nChannels == 4:
            fig.subplots_adjust(hspace=0.445, top=0.945)
        elif nChannels == 5:
            fig.subplots_adjust(hspace=0.485, top=0.945, bottom=0.07)
        elif nChannels == 6:
            fig.subplots_adjust(hspace=0.7, top=0.945, bottom=0.06)
        elif nChannels == 7:
            fig.subplots_adjust(hspace=0.8, top=0.955, bottom=0.06)
        elif nChannels == 8:
            fig.subplots_adjust(hspace=0.8, top=0.955, bottom=0.06)

        return fig

    def plotChannel(self, channel):
        try:
            x = self.data[channel]['x']
            y = self.data[channel]['y']
        except:
            self.retrieveSingleWaveform(channel)

        xUnit = self.data[channel]['xUnit']
        yUnit = self.data[channel]['yUnit']
        figure()
        plot(x,y)
        title(channel + " data")
        xlabel(xUnit)
        ylabel(yUnit)

    def testPlot(self, numWaveforms=2):
        wfRef = ['CH1','CH2','CH3','CH4','MATH','REF1','REF2','REF3','REF4']
        waveforms = wfRef[0:numWaveforms]
        self.config['activeChannels'] = waveforms
        for wf in waveforms:
            x = np.arange(-5,-5+.001*10000,.001)
            y = np.random.standard_normal(10000) + (np.random.rand()-.5)*10
            self.data[wf] = {}
            self.data[wf]['x'] = x
            self.data[wf]['y'] = y
            self.data[wf]['xUnit'] = 's'
            self.data[wf]['yUnit'] = 'V'
        self.plotMultiChan(waveforms)


if __name__ == "__main__":
    tk = Tek3kComms()
    tk.testPlot(4)
    plt.show()
