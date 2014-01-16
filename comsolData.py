#!/usr/bin/env python

#
# NOTE: In order to get good results from Comsol, particularly from
# integration, you must evalute on gauss points. For 2D axial symmetry, the
# meshvol reported  produced is the area of the element in the plane; in order
# to get a volume with which to scale a quantity, you must multiply this
# meshvol by 2*pi*R where R is the radius of the corresponding gauss point.
#
# I tested the above by simply looking at a rectangle in a 2D axial symm
# model, and in order to convert the meshvol to the actual volume, I
# merely had to multiply each meshvol value by the corresponding R times
# 2*pi, then sum all these points, which yielded the "theoretical" volume
# of the cylinder to the precision of the reported values (diff = 0).
#
# NOTE: The PTB paper shows in Table 2 a comparison of strain energies for
# an all-ULE cavity, which I've tried to replicate. I did not include the
# coating in my FEM analysis, but that doesn't contribute significantly.
# 
#   (1) it is to be noted that they arrive at their HEATMAP image (Fig 2)
#   using (probably) a 2 mm beam-waist size and by normalizing the strain
#   energy by a factor such that the maximum is 1.
#
#   (2) the actual tabulated values (Table 2) use the 240 um beam waist size as
#   quoted in Table 1.
#
#   Source: "Thermal noise in optical cavities revisited"
#   T. Kessler, T. Legero, U. Sterr
#   J. Opt. Soc. Am. B / Vol. 29, No. 1, January 2012
#
# With the above considerations, I was able to replicate their Fig. 2 and
# their Spacer and Subtrate tabulated values in Table 2.


from __future__ import division
from __future__ import with_statement

import numpy as np
import re
import os

from genericUtils import wstdout, wstderr


class ComsolData: #(OrderedDict):
    def __init__(self, *arg, **kwarg):
        #super(ComsolData, self).__init__(*arg, **kwarg)
        self.variables = {}
        self.re_modelName = re.compile(r"\% Model[:,]\s*(\S[\S ]*)")
        self.re_dimensions = re.compile(r"\% Dimension[:,]\s*(\S*)")
        self.re_nodes = re.compile(r"\% Nodes[:,]\s*(\S*)")

        #-- The following regex was designed to parse the entire header at
        #   once, but fails to capture the initial coordinate letters, e.g., R
        #   and Z or X and Y ...
        #self.re_header = re.compile(r"(\S+)(?: \((\S*)\)){0,1}(?: @ (t|freq|Eigenfrequency)=([0-9.e+-]+)){0,1}\s*", re.IGNORECASE)
        self.re_header = re.compile(r"([0-9a-zA-Z.()_+-]+)(?: \((\S*)\)){0,1}(?: @ (t|freq|Eigenfrequency)=([0-9.e+-]+)){0,1}(?:,){0,1}\s*", re.IGNORECASE)
        
        #-- This one only works after splitting the header row at commas
        #re_header = re.compile(r"(?: @ (t|freq|Eigenfrequency)=([0-9.e+-]+)){0,1}\s*[,]{0,1}", re.IGNORECASE+re.DEBUG)
        self.re_units = re.compile(r"\% .*unit[:,]\s*(\S*)")
        self.re_data = re.compile(r"([0-9a-zA-Z.()_+-]+)[,]{0,1}[\s]*")
 
    def importSpreadsheetData(self, fname, numLines=-1, header=True):
        modelNameLine = 1
        comsolVersionLine = 2
        dateLine = 3
        dimensionLine = 4
        nodesLine = 5
        numExpressionsLine = 6
        descriptionLine = 7
        unitsLine = 8
        headerLine = 9
        firstDataLine = 10
        
        self.varNames = []
        lineN = 0
        with open(fname, 'r') as f:
            while True:
                lineN += 1
        
                lineStr = f.readline()
                if len(lineStr) == 0:
                    break
       
                if lineN == dimensionLine:
                    numDimensions = int(self.re_dimensions.findall(lineStr)[0])
                    #print "numDimensions", numDimensions

                if lineN == nodesLine:
                    numDataLines = int(self.re_nodes.findall(lineStr)[0])
                    #print "numDataLines", numDataLines

                if lineN == unitsLine:
                    self.coordUnits = self.re_units.findall(lineStr)[0]
                    #print "self.coordUnits", self.coordUnits
        
                if lineN == headerLine:
                    #print lineStr[2:].strip().split(',')
                    self.varTuples = self.re_header.findall(lineStr[2:])
                    #print self.varTuples[0]
                    #print self.varTuples[1]
                    #print self.varTuples[2]
                    #print self.varTuples[-1]
                    for varTuple in self.varTuples:
                        varName = varTuple[0].replace(".", "_")
                        units = varTuple[1]
                        freqTimeType = varTuple[2]
                        freqTime = varTuple[3]

                        if len(freqTimeType) > 0:
                            freqTimeSpecd = True
                        else:
                            freqTimeSpecd = False

                        if not self.variables.has_key(varName):
                            if freqTimeSpecd:
                                self.variables[varName] = {'units':units,
                                                           'val':[],
                                                           'ft':freqTimeType,
                                                           freqTimeType:[],
                                                           'dim':2}
                            else:
                                self.variables[varName] = {'units':units,
                                                           'val':[],
                                                           'dim':1}

                        if self.variables[varName]['dim'] == 2:
                            ft = self.variables[varName]['ft']
                            self.variables[varName][ft].append(np.float64(freqTime))
                        
                        self.varNames.append(varName)
        
                if lineN == firstDataLine:
                    self.uniqueVarNames = list(set(self.varNames))
                    for varName in self.uniqueVarNames:
                        self.variables[varName]['val'] = []

                if lineN >= firstDataLine:
                    n = lineN - firstDataLine
                    for varName in self.uniqueVarNames:
                        self.variables[varName]['val'].append([])
                    valStrings = self.re_data.findall(lineStr)
                    for (varName, valString) in zip(self.varNames, valStrings):
                        try:
                            self.variables[varName]['val'][n].append(np.float64(valString))
                        except:
                            #print "'" + valString + "'"
                            self.variables[varName]['val'][n].append(complex(valString.replace("i","j")))

                if numLines >= 0 and lineN-firstDataLine+1 >= numLines:
                    break
        

        #-- Break *single-valued* datum out of list
        #    -or-
        #   stuff multi-valued data into numpy arrays
        #   And in either case, if there's a freq or time associated, make that list an NP array
        for varName in self.varNames: 
            self.variables[varName]['val'] = np.array(self.variables[varName]['val'])
            sp = self.variables[varName]['val'].shape
            if sp == (1,1):
                self.variables[varName]['val'] = self.variables[varName]['val'][0,0]
            
            if self.variables[varName].has_key('ft'):
                ft = self.variables[varName]['ft']
                self.variables[varName][ft] = np.array(self.variables[varName][ft])

            #elif sp[0] == 1:
            #    self.variables[varName]['val'] = self.variables[varName]['val'][0,:]
      
        #try:
        #    len(self.variables['R']['val'])
        #except:
        #    pass
        #else:
        #    self.coords = zip(self.variables['R']['val'], self.variables['Z']['val'])

    def appendData(self, src):
        n = 0
        for varName in np.unique(src.varNames):
            if varName in ['R', 'Z', 'X', 'Y']:
                continue
            if self.variables[varName].has_key('ft'):
                ft = self.variables[varName]['ft']
                self.variables[varName][ft] = np.concatenate(
                    (self.variables[varName][ft], src.variables[varName][ft]) )
            self.variables[varName]['val'] = np.concatenate(
                (self.variables[varName]['val'],
                 src.variables[varName]['val']), axis=1 )

    #def writeSpreadsheet(self, fname):
    #    with open(fname, 'w') as f:


def concatenateData(dataFnames, numLines=-1):
    if isinstance(dataFnames, str):
        dataFnames = [dataFnames]
    n = 0
    for fname in dataFnames:
        #wstdout( " Loading data from  " + fname + "\n" )
        if n == 0:
            comDat = ComsolData()
            comDat.importSpreadsheetData(os.path.join( os.path.expanduser("~"), fname ), numLines=numLines)
            if len(dataFnames) == 1:
                break
        else:
            tempComDat = ComsolData()
            tempComDat.importSpreadsheetData(os.path.join( os.path.expanduser("~"), fname ), numLines=numLines)
            comDat.appendData(tempComDat)
        n += 1
    return comDat


if __name__ == "__main__":
    headerTest = []
    #-- Comsol v4.3
    headerTest.append("% R                       Z                        w (m) @ freq=1                                    w (m) @ freq=20                                   w (m) @ freq=40                                   w (m) @ freq=60                                   w (m) @ freq=80                                   w (m) @ freq=100                                  w (m) @ freq=120                                  w (m) @ freq=140                                  w (m) @ freq=160")
    #-- Comsol v4.4
    headerTest.append("% R,Z,solid.omega (rad/s) @ Eigenfrequency=1,solid.omega (rad/s) @ Eigenfrequency=2,solid.omega (rad/s) @ Eigenfrequency=3,solid.omega (rad/s) @ Eigenfrequency=4,solid.omega (rad/s) @ Eigenfrequency=5,solid.omega (rad/s) @ Eigenfrequency=6,solid.omega (rad/s) @ Eigenfrequency=7,solid.omega (rad/s) @ Eigenfrequency=8,solid.omega (rad/s) @ Eigenfrequency=9,solid.omega (rad/s) @ Eigenfrequency=10,solid.omega (rad/s) @ Eigenfrequency=11,solid.omega (rad/s) @ Eigenfrequency=12")

    dataTest = []
    #-- Comsol v4.3
    dataTest.append("0                         -2.0320000000000002E-5   3.163184658904194E-8-5.253257571105478E-12i       3.1631848087487895E-8-5.2532620879589966E-12i     3.16318525942391E-8-5.253275673017112E-12i        3.163186010598492E-8-5.25329831660938E-12i        3.1631870623465245E-8-5.253330021474611E-12i      3.163188414771718E-8-5.253370791450161E-12i       3.163190068007449E-8-5.2534206314766505E-12i      3.1631920222168254E-8-5.2534795475923665E-12i     3.1631942775926096E-8-5.253547546939175E-12i      3.163196834357526E-8-5.2536246377639015E-12i      3.1631996927641414E-8-5.253710829427027E-12i      3.1632028530949193E-8-5.2538061323995324E-12i     3.163206315662417E-8-5.253910558267347E-12i       3.1632100808093434E-8-5.254024119742534E-12i")
    #-- Comsol v4.4
    dataTest.append("0,-1.0E-6,40217.86220191298-24.1687515218126i,40217.86220191298+24.1687515218126i,112699.22619363452-69.58419851241112i,112699.22619363452+69.58419851241112i,146594.47787933613-94.7889630852653i,146594.47787933613+94.7889630852653i,154652.19752320193-110.38330785740403i,154652.19752320193+110.38330785740403i,180043.40522204724-108.3307881079505i,180043.40522204724+108.3307881079505i,182426.34065921057-206.1517842909712i,182426.34065921057+206.1517842909712i,192912.50407993805-158.3260394512793i,192912.50407993805+158.3260394512793i")

    comDat = ComsolData()
    for header in headerTest:
        hTuple = comDat.re_header.findall(header[1:].strip())
        print "hTuple:", hTuple, "\n"

    comDat = ComsolData()
    for data in dataTest:
        dTuple = comDat.re_data.findall(data)
        print "dTuple:", dTuple, "\n"

    dataFnames = ["gibble/laser_cavity/comsol/resulting_data/" +
                        "comsol_cav1a_params_02_epo02_epop2_2mil_r5.2_R12.7.txt",
                  "gibble/laser_cavity/comsol/freqresp_8mil_0.05loss_epo_6-6.8kHz_mirdve.txt"
                 ]
    
    wstdout( "="*79 + "\n" )
    comDat = ComsolData()
    #print dataFnames
    for fname in dataFnames:
        print fname
        wstdout( " Loading data from  " + fname + "\n" )
        comDat.importSpreadsheetData(os.path.join( os.path.expanduser("~"), fname ))


    dataFnames = [
        "gibble/laser_cavity/comsol/fresp_0.8mil_0.05loss_1-6.3_6.7-50_20step_pgaus_mesh5_1.5.txt",
        "gibble/laser_cavity/comsol/fresp_0.8mil_0.05loss_6.3-6.7kHz_0.5Hzstep_pgaus_mesh5_1.5.txt",
        "gibble/laser_cavity/comsol/fresp_0.8mil_0.05loss_50-100k_20step_pgaus_mesh5_1.5.txt",
        "gibble/laser_cavity/comsol/fresp_0.8mil_0.05loss_100-150k_20step_pgaus_mesh5_1.5.txt"
    ]
    
    comDat = concatenateData(dataFnames)
