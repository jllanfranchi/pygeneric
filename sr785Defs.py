#!/usr/bin/env python


#-- Translation dictionaries
measGroups = {
    0: "FFT",
    1: "Correlation",
    2: "Octave",
    3: "Swept Sine",
    4: "Order",
    5: "Time/Histogram" }

measTypes = {
    # FFT Group
    0:  "FFT 1",
    1:  "FFT 2",
    2:  "Power Spectrum 1",
    3:  "Power Spectrum 2",
    4:  "Time 1",
    5:  "Time 2",
    6:  "Windowed Time 1",
    7:  "Windowed Time 2",
    8:  "Orbit",
    9:  "Coherence",
    10: "Cross Spectrum",
    11: "Frequency Response",
    12: "Capture Buffer 1",
    13: "Capture Buffer 2",
    14: "FFT User Function 1",
    15: "FFT User Function 2",
    16: "FFT User Function 3",
    17: "FFT User Function 4",
    18: "FFT User Function 5",
    # Correlation group
    19: "Auto Correlation 1",
    20: "Auto Correlation 2",
    21: "Cross Correlation",
    22: "Time 1",
    23: "Time 2",
    24: "Windowed Time 1",
    25: "Windowed Time 2",
    26: "Capture Buffer 1",
    27: "Capture Buffer 2",
    28: "Correlation Function 1",
    29: "Correlation Function 2",
    30: "Correlation Function 3",
    31: "Correlation Function 4",
    32: "Correlation Function 5",
    # Octave Group
    33: "Octave 1",
    34: "Octave 2",
    35: "Capture 1",
    36: "Capture 2",
    37: "Octave User Function 1",
    38: "Octave User Function 2",
    39: "Octave User Function 3",
    40: "Octave User Function 4",
    41: "Octave User Function 5",
    # Swept Sine Group
    42: "Spectrum 1",
    43: "Spectrum 2",
    44: "Normalized Variance 1",
    45: "Normalized Variance 2",
    46: "Cross Spectrum",
    47: "Frequency Response",
    48: "Swept Sine User Function 1",
    49: "Swept Sine User Function 2",
    50: "Swept Sine User Function 3",
    51: "Swept Sine User Function 4",
    52: "Swept Sine User Function 5",
    # Order Group
    53: "Linear Spectrum 1",
    54: "Linear Spectrum 2",
    55: "Power Spectrum 1",
    56: "Power Spectrum 2",
    57: "Time 1",
    58: "Time 2",
    59: "Windowed Time 1",
    60: "Windowed Time 2",
    61: "RPM Profile",
    62: "Orbit",
    63: "Track 1",
    64: "Track 2",
    65: "Capture Buffer 1",
    66: "Capture Buffer 2",
    67: "Order User Function 1",
    68: "Order User Function 2",
    69: "Order User Function 3",
    70: "Order User Function 4",
    71: "Order User Function 5",
    # Time/Histogram Group
    72: "Histogram 1",
    73: "Histogram 2",
    74: "PDF 1",
    75: "PDF 2",
    76: "CDF 1",
    77: "CDF 2",
    78: "Time 1",
    79: "TIme 2",
    80: "Capture Buffer 1",
    81: "Capture Buffer 2",
    82: "Histogram User Function 1",
    83: "Histogram User Function 2",
    84: "Histogram User Function 3",
    85: "Histogram User Function 4",
    86: "Histogram User Function 5" }

views = {
    0: "Log Magnitude",
    1: "Linear Magnitude",
    2: "Magnitude Squared",
    3: "Real Part",
    4: "Imaginary Part",
    5: "Phase",
    6: "Unwrapped Phase",
    7: "Nyquist",
    8: "Nichols" }

offOn = {
    0: "Off",
    1: "On" }

autoRefSource = {
    0: "Off",
    1: "Channel 1",
    2: "Channel 2" }

ssUnits = {
    0: "mV",
    1: "V",
    2: "dBVpk" }

ssType = {
    0: "Linear",
    1: "Logarithmic" }

inputSources = {
    0: "Analog",
    1: "Capture" }

linkOptions = {
    0: "Independent Channels",
    1: "Dual Channels" }

inputModes = {
    0: "A (single-ended)",
    1: "A-B (differential)" }

groundingOpts = {
    0: "Float",
    1: "Ground" }

couplingOpts = {
    0: "DC",
    1: "AC",
    2: "ICP" }

inputUnits = {
    0:  "dBVpk",
    1:  "dBVpp",
    2:  "dBVrms",
    3:  "Vpk",
    4:  "Vpp",
    5:  "Vrms",
    6:  "dBEUpk",
    7:  "dBEUpp",
    8:  "dBEUrms",
    9:  "EUpk",
    10: "EUpp",
    11: "EUrms" }
    
undb = {
    0: "Off",
    1: "On",
    2: "dBm",
    3: "dBspl" }

unpk = {
    0: "Off",
    1: "pk",
    2: "rms",
    3: "pp" }
    
psdu = {
    0: "Off",
    1: "On" }

unph = {
    0: "Degrees",
    1: "Radians" }

autoRangeModes = {
    0: "Up Only",
    1: "Tracking" }

triggerModes = {
    0: "Auto Arm",
    1: "Manual Arm",
    2: "RPM Arm",
    3: "Time Arm" }

triggerSources = {
    0: "Continuous",
    1: "Ch1",
    2: "Ch2",
    3: "External",
    4: "External TTL",
    5: "Source",
    6: "Manual" }

triggerLevelUnits = {
    0: "%",
    1: "V",
    2: "mV",
    3: "ChU" }

triggerSlopes = {
    0: "Rising",
    1: "Falling" }

triggeredSourceModes = {
    0: "1-Shot",
    1: "Continuous" }

triggerRPMArmingDeltaSenses = {
    0: "Absolute Change",
    1: "Increasing RPM",
    2: "Decreasing RPM" }


#-------------------------------------------------------------------------------
# Status word definitions
#-------------------------------------------------------------------------------
#-- Serial poll status word; queried with  *STB? or serial poll; set with *SRE
serialPollStatusWord = {
    0: 'INST',
    1: 'DISP',
    2: 'INPT',
    3: 'IERR',
    4: 'MAV',
    5: 'ESB',
    6: 'SRQ',
    7: 'IFC' }

#-- Standard Event status; queried with *ESR?; set with *ESE (Only for GPIB?)
standardEventStatusWord = {
    2: 'QRY',
    3: 'DDE',
    4: 'EXE',
    5: 'CME',
    6: 'URQ',
    7: 'PON' }

#-- Instrument status; queried with INST?; set with INSE
instrumentStatusWord = {
    0: 'TRIG',
    1: 'DISK',
    2: 'OUTP',
    3: 'TACH',
    4: 'CAPT',
    5: 'PAUS',
    6: 'STRT',
    7: 'PLBK',
    8: 'PREV' }

#-- Display status; queried with DSPS?; set with DSPE
displayStatusWord = {
    0:  'NEWA',
    1:  'AVGA',
    2:  'STLA',
    3:  'LIMA',
    4:  'SSA',
    5:  'WFA',
    6:  'WFD',
    # 7 is unusued
    8:  'NEWB',
    9:  'AVGB',
    10: 'STLB',
    11: 'LIMB',
    12: 'SSB',
    13: 'WFB',
    14: 'WFB' }

#-- Input status; queried with INPS?; set with INPE
inputStatusWord = {
    0:  'LOW1',
    1:  'HLF1',
    2:  'OVL1',
    3:  'HIV1',
    4:  'HIV1',
    # 5-7 are unusued
    8:  'LOW2',
    9:  'HLF2',
    10: 'OVL2',
    11: 'HIV2',
    12: 'ARG2' }

#-- Error status; queried with ERRS?; set with ERRE
errorStatusWord = {
    0:  'OUTE',
    1:  'DSKE',
    2:  'FLTE',
    3:  'RAME',
    4:  'ROME',
    5:  'VIDE',
    6:  'HELPE',
    7:  'DSDE',
    8:  'DSPE',
    9:  'DSRE',
    10: 'CAL0',
    11: 'CAL1',
    12: 'CAL2' }

def interpretInstStatusWord(sw):
    sw = int(sw)
    # TODO: Are the bit orders correct? (i.e., is bit 0 in the 1's position?)
    status = {}
    for (bitNum, name) in instrumentStatusWord.iteritems():
        if (sw & 2**bitNum):
            status.update( {name: True} )
        else:
            status.update( {name: False} )

    #status = {
    #    'TRIG': False,
    #    'DISK': False,
    #    'OUTP': False,
    #    'TACH': False,
    #    'CAPT': False,
    #    'PAUS': False,
    #    'STRT': False,
    #    'PLBK': False,
    #    'PREV': False }

    #if sw & 1:
    #    status['TRIG'] = True
    #if sw & 2:
    #    status['DISK'] = True
    #if sw & 4:
    #    status['OUTP'] = True
    #if sw & 8:
    #    status['TACH'] = True
    #if sw & 16:
    #    status['CAPT'] = True
    #if sw & 32:
    #    status['PAUS'] = True
    #if sw & 64:
    #    status['STRT'] = True
    #if sw & 128:
    #    status['PLBK'] = True
    #if sw & 256:
    #    status['PREV'] = True

    return status
