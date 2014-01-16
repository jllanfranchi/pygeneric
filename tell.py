#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import with_statement
from __future__ import division

import sys

from instrumentComms import InstrumentComms

def wstdout(txt):
    sys.stdout.write(txt)
    sys.stdout.flush()

def wstderr(txt):
    sys.stderr.write(txt)
    sys.stderr.flush()

if __name__ == "__main__":
    #commsType = "Ethernet"
    ipAddress = "128.118.112.2"
    httpPort = 80

    try:
        #-- Set up with serial port comms
        instrument = InstrumentComms(useSerial=True)
        instrument.simpleSerial()
        commsType = "serial"
    except:
        #-- If that failed, try to set up with Ethernet comms
        instrument = InstrumentComms(useEthernet=True, ipAddress=ipAddress)
        commsType = "Ethernet"

    instrument.tell(sys.argv[1])