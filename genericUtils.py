# -*- coding: iso-8859-15 -*-

import re, os, sys, time
import numpy as np


def timediffstamp(dt_sec, hms_always=False):
    if dt_sec < 0:
        sign = '-'
        dt_sec = -dt_sec
    else:
        sign = ''

    r = dt_sec % 3600
    h = int((dt_sec - r)/3600)
    s = r % 60
    m = int((r - s)/60)
    strdt = ''
    if hms_always or h != 0:
        strdt += format(h, '02d') + ':'
    if hms_always or h != 0 or m != 0:
        strdt += format(m, '02d') + ':'

    if float(s) == int(s):
        s = int(s)
        if len(strdt) > 0:
            s_fmt = '02d'
        else:
            s_fmt = 'd'
    else:
        s = np.round(s, 3)
        if len(strdt) > 0:
            s_fmt = '06.3f'
        else:
            s_fmt = '.3f'
    if len(strdt) > 0:
        strdt += format(s, s_fmt)
    else:
        strdt += format(s, s_fmt) + ' sec'
    
    return sign + strdt


def timestamp(d=True, t=True, tz=True, utc=False, winsafe=False):
    '''Simple utility to print out a time, date, or time&date stamp,
    with some reconfigurability for commonly-used options. Default is in
    ISO8601 format without colons separating hours, min, and sec to avoid
    file naming issues.

    Options:
        d          print date (default: True)
        t          print time (default: True)
        tz         print timezone offset from UTC (default: True)
        utc        print time/date in UTC (default: False)
        winsafe    omit colons between hours/minutes (default: True)

    '''
    if utc:
        timeTuple = time.gmtime()
    else:
        timeTuple = time.localtime()

    dts = ""
    if d:
        dts += time.strftime("%Y-%m-%d", timeTuple)
        if t:
            dts += "T"
    if t:
        if winsafe:
            dts += time.strftime("%H%M%S", timeTuple)
        else:
            dts += time.strftime("%H:%M:%S", timeTuple)

        if tz:
            if utc:
                if winsafe:
                    dts += time.strftime("+0000")
                else:
                    dts += time.strftime("+00:00")
            else:
                offset = time.strftime("%z")
                if not winsafe:
                    offset = offset[:-2:] + ":" + offset[-2::]
                dts += offset
    return dts

#-- Credit to http://nedbatchelder.com/blog/200712.html#e20071211T054956
#   for the original code and to
#   http://personal.inet.fi/cool/operator/Human%20Sort.py
#   for the internationalized version below
#numeric_rex = re.compile(r'([0-9]+)')
#def numericSortFn(s):
#    
#
## The code extended with suitable renamings:
#spec_dict = {'Å':'A', 'Ä':'A'}
#
#def spec_order(s):
#    return ''.join([spec_dict.get(ch, ch) for ch in s])
#    
#def trynum(s):
#    try:
#        return float(s)
#    except:
#        return spec_order(s)
#
#def alphanum_key(s):
#    """ Turn a string into a list of string and number chunks.
#        "z23a" -> ["z", 23, "a"]
#    """
#    return [ trynum(c) for c in re.split('([0-9]+\.?[0-9]*)', s) ]
#
#def sort_nicely(l):
#    """ Sort the given list in the way that humans expect.
#    """
#    l.sort(key=alphanum_key)

#-- See http://nedbatchelder.com/blog/200712/human_sorting.html#comments, comment by "Andre Bogus"
def nsort(l):
    return sorted(l, key=lambda a:zip(re.split("(\\d+)", a)[0::2], map(int, re.split("(\\d+)", a)[1::2])))

#-- ... and comment by "Py User":
#def nsort_ci(l) return sorted(l, key=lambda a.lower()):zip(re.split("(\\d+)", a)[0::2], map(int, re.split("(\\d+)", a)[1::2]))) 


#-- Recursive w/ ordering reference: http://stackoverflow.com/questions/18282370/python-os-walk-what-order
def findFiles(rootpath, regex=None, recurse=True, dir_sorter=nsort, file_sorter=nsort):
    if isinstance(regex, (str, unicode)):
        regex = re.compile(regex)
    
    if regex is None:
        def validfilefunc(fname):
            return True, None
    else:
        def validfilefunc(fname):
            match = regex.match(fname)
            if match and (len(match.groups()) == regex.groups):
                return True, match
            return False, None

    if recurse:
        for root, dirs, files in os.walk(rootpath):
            for dirname in dir_sorter(dirs):
                fulldirpath = os.path.join(root, dirname)
                for basename in file_sorter(os.listdir(fulldirpath)):
                    fullfilepath = os.path.join(fulldirpath, basename)
                    if os.path.isfile(fullfilepath):
                        isValid, match = validfilefunc(basename)
                        if isValid:
                            yield fullfilepath, basename, match
    else:
        for basename in file_sorter(os.listdir(rootpath)):
            fullfilepath = os.path.join(rootpath, basename)
            #if os.path.isfile(fullfilepath):
            isValid, match = validfilefunc(basename)
            if isValid:
                yield fullfilepath, basename, match

def wstdout(x):
    sys.stdout.write(x)
    sys.stdout.flush()


def wstderr(x):
    sys.stderr.write(x)
    sys.stderr.flush()
