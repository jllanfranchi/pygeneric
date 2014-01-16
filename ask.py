#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import with_statement
from __future__ import division

import sys

from instrumentComms import InstrumentComms

DEBUG = False

def wstdout(txt):
    sys.stdout.write(txt)
    sys.stdout.flush()

def wstderr(txt):
    sys.stderr.write(txt)
    sys.stderr.flush()

if __name__ == "__main__":
    ipAddress = "128.118.112.2"
    httpPort = 80

    try:
        #-- Set up with serial port comms
        instrument = InstrumentComms(useSerial=True, baud=19200)
        instrument.simpleSerial(debug=DEBUG)
        commsType = "serial"
    except:
        #-- If that failed, try to set up with Ethernet comms
        instrument = InstrumentComms(useEthernet=True, ipAddress=ipAddress)
        commsType = "Ethernet"

    wstdout("\"" + instrument.query(sys.argv[1]) + "\"")