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
import urllib2
import urlparse
import pprint

from smartFormat import smartFormat, simpleFormat

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

PPstderr = pprint.PrettyPrinter(indent=2, width=80, depth=None, stream=sys.stderr)
PPstdout = pprint.PrettyPrinter(indent=2, width=80, depth=None, stream=sys.stderr)

#===============================================================================
# Module exception definitions
#===============================================================================
class CommProtocolError(Exception):
    pass

class MethodCallError(CommProtocolError):
    pass

class InvalidArgumentError(CommProtocolError):
    pass

class ArgumentConversionError(InvalidArgumentError):
    pass

class ArgumentLengthError(InvalidArgumentError):
    pass

class UnexptectedArgumentError(InvalidArgumentError):
    pass

class InvalidResponseError(CommProtocolError):
    pass

class ResponseConversionError(InvalidResponseError):
    pass

class ResponseLengthError(InvalidResponseError):
    pass

class UnexptectedResponseError(InvalidResponseError):
    pass

class UnsupportedOSError(CommProtocolError):
    pass

class InvalidCommandError(CommProtocolError):
    pass

class ChecksumMismatchError(CommProtocolError):
    pass

class SerialPortError(CommProtocolError):
    pass


class InstrumentComms(object):
    def __init__(self, useSerial=True, useEthernet=False,
                 ipAddress=None, ethPort=80,
                 serialPortName=None,
                 baud=19200, bytesize=8, parity='N', stopbits=1,
                 timeout=2, xonxoff=False, rtscts=False, dsrdtr=False,
                 serCmdPrefix="", serCmdTerm='\r', serIdnCmd='*IDN?',
                 ethCmdPrefix='COMMAND=', ethCmdTerm='\r\ngpibsend=Send\r\n',
                 ethDataRE=re.compile('.*<TEXTAREA.*>(.*)</TEXTAREA>'),
                 ethIdnCmd='*IDN?',
                 argSep=""):
        
        self.config = {}
        self.data = {}

        self.lowerRegex = re.compile("[a-z]")

        self.useSerial = useSerial
        self.useEthernet = useEthernet

        if self.useEthernet == True and ipAddress == None:
            raise CommProtocolError("IP address must be supplied")
        else:
            self.ipAddress = ipAddress
        
        if self.useEthernet == True and ethPort == None:
            raise CommProtocolError("Ethernet port must be supplied")
        else:
            self.ethPort = ethPort

        self.controllerSPName = serialPortName
        if self.useSerial:
            self.serialPort = serial.Serial()
        else:
            self.serialPort = None
        # Common baud rates: 115200, 38400, 19200, 9600, 4800, 2400, 1200
        self.BAUDRATE = baud
        self.BYTESIZE = bytesize
        self.PARITY = parity
        self.STOPBITS = stopbits
        self.SERIALTIMEOUT = timeout
        self.XONXOFF = xonxoff
        self.RTSCTS = rtscts
        self.DSRDTR = dsrdtr
        self.SERCMD_TERM = serCmdTerm
        self.SERCMD_TERM_LEN = len(self.SERCMD_TERM)
        self.SERCMD_PREFIX = serCmdPrefix
        
        self.ETHCMD_PREFIX = ethCmdPrefix
        self.ETHCMD_TERM = ethCmdTerm
        self.ETHDATA_RE = ethDataRE
   
        self.CMD_ARG_SEP = argSep

    def scanSerialPorts(self, debug=False):
        if not self.useSerial:
            raise Exception("Serial port operations not allowed")

        if self.serialPort.isOpen():
            self.serialPort.flushInput()
            self.serialPort.flushOutput()
            self.serialPort.close()

        if os.name is 'nt':
            possibleSerPorts = [cp[1] for cp in sorted(scanwin32.comports())]

        elif os.name is 'posix':
            possibleSerPorts = glob.glob('/dev/ttyS*') + \
                    glob.glob('/dev/ttyUSB*') + glob.glob('/dev/ttyACM*')

        else:
            raise UnsupportedOSError("Operating system " + os.name + 
                                     " unsupported.")
       
        serPorts = []
        for psp in possibleSerPorts:
            try:
                #-- Open port with the settings we will use, to make sure these
                #   work on the given port
                self.openPort(psp)
                sp = self.serialPort
                spDict = {}
                spDict['name'] = sp.portstr
                spDict['supportedBaudrates'] = sp.getSupportedBaudrates()
                spDict['supportedByteSizes'] = sp.getSupportedByteSizes()
                spDict['supportedParities'] = sp.getSupportedParities()
                spDict['supportedStopbits'] = sp.getSupportedStopbits()
                self.__closeSerialPort()
                serPorts.append(spDict)
            except: # serial.Serial.SerialException, err:
                #wstderr('ERROR: %s\n' % str(err))
                pass
   
        self.serPorts = serPorts

        if debug:
            wstderr("\nD0 --> serPorts:\n"+PPstderr.pformat(serPorts)+"\n\n")

        return self.serPorts

    def simpleSerial(self, debug=False):
        allSP = self.scanSerialPorts(debug)
        self.controllerSPName = allSP[0]['name']
        self.openPort()

    def findInstrument(self, mfr=None, model=None, idnStr=None,
                       ethMAC=None, ethHostname=None,
                       searchSerial=True, searchEthernet=True,
                       searchGPIB=False, idnQueryStr="*IDN?",
                       debug=False):
        instrumentMatches = {'serial':[], 'ethernet':[], 'GPIB':[]}
        if searchSerial:
            #-- Search serial ports for the instrument
            allSP = self.scanSerialPorts(debug)
            for spNum in range(len(allSP)):
                isMatch = True
                self.openPort(allSP[spNum]['name'])
                if mfr or model or idnStr:
                    try:
                        idn = self.query(idnQueryStr)
                    except:
                        isMatch = False
                        continue
                    if mfr and not np.any([mfr.upper() in s.upper() for s in
                                           idn.split(",")]):
                        isMatch = False
                    if model and not np.any([model.upper() in s.upper() for s in
                                             idn.split(",")]):
                        isMatch = False
                    if idnStr and not np.any([idnStr.upper() in s.upper()
                                              for s in idn.split(",")]):
                        isMatch = False

                if ethMAC:
                    try:
                        MACAddy = self.query("ETHERnet:ENETADDress?")
                    except:
                        isMatch = False
                        continue

                if ethHostname:
                    try:
                        hostname = self.query("ETHERnet:NAME?")
                    except:
                        continue
                    if mfr and not np.any([mfr.upper() in s.upper() for s in
                                           idn.split(",")]):
                        isMatch = False

                if isMatch:
                    instrumentMatches['serial'].append(self.controllerSPName)

        if searchEthernet:
            pass
        
        if searchGPIB:
            pass

        if debug:
            wstderr("\nD0 --> instrumentMatches:\n" +
                    PPstderr.pformat(instrumentMatches) + "\n\n")

        return instrumentMatches

    def __openSerialPort(self, serPortName=None):
        if not self.useSerial:
            raise Exception("Serial port operations not allowed")

        self.__closeSerialPort()
        if serPortName == None:
            serPortName = self.controllerSPName
        if self.controllerSPName == None:
            raise SerialPortError("No valid serial port found")
        self.serialPort = serial.Serial(
            serPortName, baudrate=self.BAUDRATE, bytesize=self.BYTESIZE,
            parity=self.PARITY, stopbits=self.STOPBITS,
            timeout=self.SERIALTIMEOUT, xonxoff=self.XONXOFF,
            rtscts=self.RTSCTS, writeTimeout=self.SERIALTIMEOUT,
            dsrdtr=self.DSRDTR, interCharTimeout=self.SERIALTIMEOUT
        )
        self.serialPort.flushInput()
        self.serialPort.flushOutput()
        return self.serialPort

    def __closeSerialPort(self):
        if self.serialPort.isOpen():
            self.serialPort.close()
        self.serialPortOpen = False

    def __serialSendCommand(self, command, upperCaseOnly=True):
        if not self.isPortOpen():
            self.openPort()

        if upperCaseOnly:
            command2 = self.lowerRegex.sub('', command)
        else:
            command2 = command

        #wstderr("\""+ command2 +"\"\n")

        fullString = self.SERCMD_PREFIX + command2 + self.SERCMD_TERM

        #-- Send command
        
        #print "="*80
        #print "Sending..."
        #print "command:     '" + command + "' translated to '" + command2 + "'"
        #print ""
        #sys.stdout.flush()

        #-- Send first char, wait for 1 second, then send remaining chars
        self.serialPort.write(fullString)
        self.serialPort.flush()

    def __serialReceiveASCIIResponse(self, timeout=0):
        t0 = time.time()
        if not self.serialPort.isOpen():
            self.openPort()

        # DEBUG
        #print "Receiving..."
        #sys.stdout.flush()
        try:
            response = ''
            char = ''
            while True:
                char = self.serialPort.read(1)
                if char == "":
                    raise Exception("Serial port timeout exceeded")
                response += char
                # DEBUG
                #sys.stdout.write('\n"' + str(response) + '"\n"')
                #sys.stdout.flush()
                if response[-self.SERCMD_TERM_LEN:] == self.SERCMD_TERM:
                    response = response[0:-self.SERCMD_TERM_LEN]
                    # DEBUG
                    #sys.stdout.write('breaking!')
                    #sys.stdout.flush()
                    break
                if (timeout > 0) and (time.time()-t0 > timeout):
                    raise Exception("Response timeout exceeded")
        except:
            # DEBUG
            #print "ERROR getting response"
            #sys.stdout.flush()
            raise Exception("ERROR getting response")
        if len(response) < 1000:
            pass
            # DEBUG
            #print 'response:    >>', response
            #sys.stdout.flush()
        else:
            pass
            # DEBUG
            #print 'response[0:500]', response[0:500]
            #sys.stdout.flush()
        # DEBUG
        #print 'response len >>', len(response)
        #sys.stdout.flush()

        return response

    def __serialReceiveBinaryResponse(self, timeout=0):
        t0 = time.time()
        if not self.isPortOpen():
            self.openPort()

        try:
            response = ''
            char = ''

            #-- Try twice to receive initial char: should be '#'
            goodToGo = False
            for n in range(2):
                if self.serialPort.read(1) == '#':
                    goodToGo = True
                    break
            if not goodToGo:
                raise Exception(
                    "ERROR -- initial char in binary receive not '#'")

            #-- Receive char that indicates how many ascii characters to get
            #   that will tell the length of the binary data to be transferred
            dataLengthNumberLength = int(self.serialPort.read(1))
            #print dataLengthNumberLength
            lengthStr = ''
            for n in range(dataLengthNumberLength):
                lengthStr += self.serialPort.read(1)
            dataLength = int(lengthStr)
            #print dataLength
            data = ''
            bytesToRead = dataLength
            n = 0
            while True:
                data += self.serialPort.read(1)
                n += 1
                if n == bytesToRead:
                    break
            #-- Read the command termination character or sequence
            terminator = ''
            for n in range(self.SERCMD_TERM_LEN):
                terminator += self.serialPort.read(1)

            if (timeout > 0) and (time.time()-t0 > timeout):
                raise Exception("Response timeout exceeded")
        except:
            raise Exception("ERROR getting response")
        if len(data) < 500:
            pass
        else:
            pass

        return data

    def __ethernetQuery(self, command, returnResponse=True, upperCaseOnly=True,
                        timeout=0):
        # TODO: Implement timeout
        url = 'http://' + self.ipAddress + '/Comm.html' #+ str(self.ethPort)
        
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

    def __ethernetBinaryQuery(self, command, returnResponse=True, timeout=0):
        """??? -- Haven't figured out how to do binary data via HTTP post;
        possibly if I used a different command altogether (see web
        interface, as they have a data download function there)"""
        # TODO: Implement timeout
        url = 'http://' + self.ipAddress + '/Comm.html' #+ str(self.ethPort)
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

    def receiveBinaryResponse(self, timeout=0):
        if self.useSerial:
            return self.__serialReceiveBinaryResponse(timeout=timeout)
        else:
            return self.__receiveEtherentResponse(timeout=timeout)

    def isPortOpen(self):
        if self.useEthernet:
            return True
        else:
            return self.serialPort.isOpen()

    def openPort(self, spName=None):
        if spName != None:
            self.controllerSPName = spName

        if self.useEthernet:
            return
        self.__openSerialPort()

    def closePort(self):
        if self.useEthernet:
            return
        self.__closeSerialPort()

    def tell(self, command):
        if self.useSerial:
            self.__serialSendCommand(command)
        else:
            self.__ethernetQuery(command, returnResponse=False)

    def query(self, command, timeout=0):
        if self.useSerial:
            self.__serialSendCommand(command)
            return self.__serialReceiveASCIIResponse(timeout=timeout)
        else:
            return self.__ethernetQuery(command, timeout=timeout)

    def listen(self, timeout=0):
        if self.useSerial:
            return self.__serialReceiveASCIIResponse(timeout=timeout)
        else:
            raise Exception("Cannot just listen while using Ethernet comms")

    def binQuery(self, command, timeout=0):
        if self.useSerial:
            self.__serialSendCommand(command)
            return self.__serialReceiveBinaryResponse(timeout=timeout)
        else:
            return self.__ethernetQuery(command, timeout=timeout)

    def binListen(self, timeout=0):
        if self.useSerial:
            return self.receiveBinaryResponse(timeout=timeout)
        else:
            raise Exception("Cannot just listen while using Ethernet comms")


if __name__ == "__main__":
    tk = InstrumentComms()
    #tk.testPlot(4)
    #plt.show()
