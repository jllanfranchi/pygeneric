#!/usr/bin/env python
"""
Copied in large part from David Müller,
http://www.d-mueller.de/blog/dealing-with-trusted-timestamps-in-php-rfc-3161/
"""

import os
import sys
import subprocess

class TrustedTimestamps:
    def __init__(self):
        pass

    def createRequestFile(self, hashText):
        subprocess.check_output("openssl ts -query -digest " + hashText + "", stderr=subprocess.STDOUT)
