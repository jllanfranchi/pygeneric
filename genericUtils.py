# -*- coding: iso-8859-15 -*-
"""
My collected generic utilities (whether coded by me or others)
"""


from __future__ import absolute_import, division

import copy
import cPickle
import functools
import grp
import inspect
import itertools
import multiprocessing
import os
import re
import sys
import time

import numpy as np

try:
    from pisa.utils.log import logging
except ImportError:
    class Logging(object):
        def __init__(self):
            pass

        @staticmethod
        def info(s):
            sys.stdout.write('[ Info    ] %s\n' % s)
            sys.stdout.flush()

        @staticmethod
        def warn(s):
            sys.stderr.write('[ Warning ] %s\n' % s)
            sys.stderr.flush()

        @staticmethod
        def error(s):
            sys.stderr.write('[ Error   ] %s\n' % s)
            sys.stderr.flush()
    logging = Logging()

try:
    import pisa.utils.fileio as fileio
except ImportError:
    pass


__all__ = ['chown_and_chmod', 'SIGMA_OR_PCT_RE', 'sigmaOrPct2ConfIntvl',
           'sigmaOrPct2chi2', 'sigma2confIntvl', 'pct2sigma', 'pct2confIntvl',
           'confIntvl2chi2', 'sigma2chi2', 'pct2chi2', 'jeffreys_interval',
           'trace', 'my_hash', 'cmp_to_key', 'genericTester', 'DictDiffer',
           'expand', 'absPath', 'mkdir', 'timediffstamp', 'timestamp',
           'nsort', 'findFiles', 'wstdout', 'wstderr', 'memoize_volatile',
           'func_memoize_persistent', 'NUMBER_RESTR', 'NUMBER_RE',
           'HRGROUP_RESTR', 'HRGROUP_RE', 'num2floatOrInt', 'isint',
           'hrgroup2list', 'WS_RE', 'hrlist2list', 'hrlol2lol', 'list2hrlist',
           'hrbool2bool', 'two_bad_seeds', 'n_bad_seeds', 'samplesFilename',
           'sampleHypercube', 'linExtrap', 'rangeBelowThresh',
           'test_rangeBelowThresh', 'home', 'pdSafe', 'makeFuncMappable',
           'applyParallel']


def chown_and_chmod(f, mode, uid=-1, group=-1):
    """Change group and permissions of a file handle or file path

    Parameters
    ----------
    f : file handle or string
    mode : binary
    uid : None or int
        None or -1 does not set uid
    group : None, int, or string
        If None or -1, do not set gid. If int or string, set appropriate gid:
        string is interpreted as group name, which must be converted by OS to
        GID, while int is interpreted directly as a GID.

    """
    if group is None:
        gid = -1
    elif isinstance(group, basestring):
        gid = get_gid(group)
    elif isinstance(group, int):
        gid = group
    else:
        raise TypeError('Invalid `group`: %s, type %s' % (group, type(group)))

    if uid is None:
        uid = -1

    if isinstance(f, file):
        os.fchown(f.fileno(), uid, gid)
        os.fchmod(f.fileno(), mode)
    elif isinstance(f, basestring):
        os.chown(f, uid, gid)
        os.chmod(f, mode)
    else:
        raise TypeError('Unhandled type for arg `f`: %s' % type(f))


SIGMA_OR_PCT_RE = re.compile(
    r'(?P<val>[0-9]+)(?P<unit>sigma|sig|pct|percent|%)'
)
def sigmaOrPct2ConfIntvl(s):
    md = SIGMA_OR_PCT_RE.match(s.lower()).groupdict()
    if md['unit'] in ['pct', 'percent', '%']:
        return pct2confIntvl(float(md['val'])/100.)
    if md['unit'] in ['sig', 'sigma']:
        return sigma2confIntvl(float(md['val']))
    raise ValueError('Could not parse string into sigma or percent: "%s"' % s)


def sigmaOrPct2chi2(s, dof):
    md = SIGMA_OR_PCT_RE.match(s.lower()).groupdict()
    if md['unit'] in ['pct', 'percent', '%']:
        return pct2chi2(float(md['val'])/100., dof=dof)
    if md['unit'] in ['sig', 'sigma']:
        return sigma2chi2(float(md['val']), dof=dof)
    raise ValueError('Could not parse string into sigma or percent: "%s"' % s)


def sigma2confIntvl(s):
    from scipy import stats
    if hasattr(s, '__len__'):
        s = np.array(s)
    return stats.chi2.cdf(s**2, 1)


def pct2sigma(pct):
    from scipy import stats
    if hasattr(pct, '__len__'):
        pct = np.array(pct)
    return np.sqrt(stats.chi2.ppf(pct, 1))


def pct2confIntvl(pct):
    return sigma2confIntvl(pct2sigma(pct))


def confIntvl2chi2(ci, dof):
    from scipy import stats
    if hasattr(ci, '__len__'):
        ci = np.array(ci)
    return stats.chi2.ppf(ci, dof)


def sigma2chi2(sigma, dof):
    return sigma**2 #confIntvl2chi2(sigma2confIntvl(sigma), dof)


def pct2chi2(pct, dof):
    return confIntvl2chi2(pct2confIntvl(pct), dof)


def jeffreys_interval(x_successes, n_trials, conf):
    """Compute and return the Jeffreys interval.

    Parameters
    ----------
    x_successes : numeric
        Number of successes
    n_trials
        Number of trials
    conf
        Confidence at which to compute the interval, e.g. 0.682689 for 1-sigma.
        Cutoff is applied at 0.5 due to possible buggy behavior at lower
        values.

    Returns
    -------
    lower_bound, upper_bound

    Notes
    -----
    For details, see following Wikipedia entry (and contained references):
    https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval

    At present, picking low `conf` might yield erroneous results
    (i.e., an interval that does not include x_successes/n_trials).

    """
    from scipy import stats
    assert conf > 0.5, 'Due to lack of understanding, `conf`' \
            ' is currently limited to be greater than 0.5.'
    lower_bound, upper_bound = stats.beta.interval(
        conf,
        x_successes + 0.5,
        n_trials - x_successes + 0.5
    )
    if x_successes == 0:
        lower_bound = 0
    if x_successes == n_trials:
        upper_bound = n_trials
    return lower_bound, upper_bound


# NOTE: See pyDOE for regular Latin hypercube sampling

#def orthogonalSample(dims, divs, subdivs, seed=1439):
#    """
#    dims
#        Number of dimensions of parameter hypercube
#
#    divs
#        Number of divisions to divide each parameter into; it is guaranteed
#        that each of the resulting subspaces will receive one sample
#
#        N_subspaces = divs**dims
#
#    subdivs
#        Number of subdivisions to divide the division into (for each parameter)
#
#        N_bins = (subdivs*divs)**dims
#        N_samples = N_subspaces
#
#    seed
#        Random seed to set prior to generating the samples
#    """
#    np.random.seed(seed)


def trace(frame, event, arg): # pylint: disable=unused-argument
    wstderr("%s, %s:%d\n" % (event, frame.f_code.co_filename, frame.f_lineno))
    return trace


def my_hash(s):
    import xxhash
    return xxhash.xxh64(s).hexdigest()


def cmp_to_key(mycmp):
    """Convert a cmp= function into a key= function.
    wiki.python.org/moin/HowTo/Sorting"""
    class K(object):
        def __init__(self, obj, *args): # pylint: disable=unused-argument
            self.obj = obj
        def __lt__(self, other):
            return mycmp(self.obj, other.obj) < 0
        def __gt__(self, other):
            return mycmp(self.obj, other.obj) > 0
        def __eq__(self, other):
            return mycmp(self.obj, other.obj) == 0
        def __le__(self, other):
            return mycmp(self.obj, other.obj) <= 0
        def __ge__(self, other):
            return mycmp(self.obj, other.obj) >= 0
        def __ne__(self, other):
            return mycmp(self.obj, other.obj) != 0
    return K


def genericTester(cases, transform):
    passed = []
    for case in cases:
        result = transform(case['input'])
        passfail = (result == case['output'])
        if not passfail:
            wstdout('Failure: input\n' + str(case['input']) +
                    '\nexpected to yield\n' + str(case['output']) +
                    '\nbut got\n' + str(result) + '\n')
        passed.append(passfail)
    assert np.all(passed)


class DictDiffer(object):
    """Dictionary difference calculator

    Originally posted as:
    http://stackoverflow.com/questions/1165352/fast-comparison-between-two-python-dictionary/1165552#1165552

    Calculate the difference between two dictionaries as:
    (1) items added
    (2) items removed
    (3) keys same in both but changed values
    (4) keys same in both and unchanged values
    """
    def __init__(self, current_dict, past_dict):
        self.current_dict, self.past_dict = current_dict, past_dict
        self.set_current = set(current_dict.keys())
        self.set_past = set(past_dict.keys())
        self.intersect = self.set_current.intersection(self.set_past)

    def added(self):
        return self.set_current - self.intersect

    def removed(self):
        return self.set_past - self.intersect

    def changed(self):
        return set(o for o in self.intersect
                   if self.past_dict[o] != self.current_dict[o])

    def unchanged(self):
        return set(o for o in self.intersect
                   if self.past_dict[o] == self.current_dict[o])


def expand(path):
    """Shortcut to expand user and (shell) vars in a pathname"""
    return os.path.expandvars(os.path.expanduser(path))


def absPath(path):
    return os.path.abspath(os.path.expandvars(os.path.expanduser(path)))


def path_components(path):
    reversed_comp = []
    while True:
        parts = os.path.split(path)
        reversed_comp.append(parts[1])
        if parts[0] == '':
            break
        if parts[0] == '/':
            reversed_comp.append(parts[0])
            break
        path = parts[0]
    return reversed_comp[::-1]


def get_gid(group):
    if isinstance(group, int):
        return group
    if isinstance(group, basestring):
        return grp.getgrnam(group).gr_gid


def mkdir(d, mode=0o2777, group=None, warn=True):
    """Only set mode and group for dirs created by this function"""
    d = expand(d)
    gid = None
    if group is not None:
        gid = get_gid(group)

    if warn and os.path.isdir(d):
        logging.warn('Directory already exists: "%s"', d)
        return

    dirs = path_components(d)
    fullpath = ''
    for d in dirs:
        fullpath = os.path.join(fullpath, d)
        if os.path.isdir(fullpath):
            continue
        os.mkdir(fullpath, mode)
        if gid is not None:
            os.chown(fullpath, -1, gid)


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
        if strdt:
            s_fmt = '02d'
        else:
            s_fmt = 'd'
    else:
        s = np.round(s, 3)
        if strdt:
            s_fmt = '06.3f'
        else:
            s_fmt = '.3f'
    if strdt:
        strdt += format(s, s_fmt)
    else:
        strdt += format(s, s_fmt) + ' sec'

    return sign + strdt


def timestamp(at=None, d=True, t=True, tz=True, utc=False, winsafe=False,
              t_sep='T'):
    """Simple utility to print out a time, date, or time & date stamp,
    with some reconfigurability for commonly-used options.

    Default is in ISO8601 format in local time. Use winsafe mode to remove
    colons separating hours, min, and sec to avoid file naming issues.

    Parameters
    ----------
    at : None or int
        Time in seconds for which to get the timestamp. If None, current time
        is used.
    d
        print date (default: True)
    t
        print time (default: True)
    tz
        print timezone offset from UTC (default: True)
    utc
        print time/date in UTC (default: False)
    winsafe
        omit colons between hours/minutes (default: False)
    t_sep
        Separator between date and time (default: "T")

    """
    if at is None:
        at = time.time()

    if utc:
        timeTuple = time.gmtime(at)
    else:
        timeTuple = time.localtime(at)

    dts = ""
    if d:
        dts += time.strftime("%Y-%m-%d", timeTuple)
        if t:
            dts += t_sep
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
                    offset = offset[:-2] + ":" + offset[-2:]
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

def nsort(l):
    """See http://nedbatchelder.com/blog/200712/human_sorting.html#comments,
    comment by "Andre Bogus"

    """
    return sorted(
        l,
        key=lambda a: zip(re.split("(\\d+)", a)[0::2],
                          [int(x) for x in re.split("(\\d+)", a)[1::2]])
    )

#-- ... and comment by "Py User":
#def nsort_ci(l) return sorted(l, key=lambda a.lower()):zip(re.split("(\\d+)", a)[0::2], map(int, re.split("(\\d+)", a)[1::2])))


def findFiles(root, regex=None, fname=None, recurse=True, dir_sorter=nsort,
              file_sorter=nsort):
    """Recursive w/ ordering code thanks to
    http://stackoverflow.com/questions/18282370/python-os-walk-what-order"""
    if isinstance(regex, basestring):
        regex = re.compile(regex)

    if regex is None:
        if fname is None:
            def validfilefunc(fn): # pylint: disable=unused-argument
                return True, None
        else:
            def validfilefunc(fn):
                if fn == fname:
                    return True, None
                return False, None
    else:
        def validfilefunc(fn):
            match = regex.match(fn)
            if match and (len(match.groups()) == regex.groups):
                return True, match
            return False, None

    if recurse:
        for rootdir, dirs, files in os.walk(root):
            for basename in file_sorter(files):
                fullfilepath = os.path.join(root, basename)
                isValid, match = validfilefunc(basename)
                if isValid:
                    yield fullfilepath, basename, match
            for dirname in dir_sorter(dirs):
                fulldirpath = os.path.join(rootdir, dirname)
                for basename in file_sorter(os.listdir(fulldirpath)):
                    fullfilepath = os.path.join(fulldirpath, basename)
                    if os.path.isfile(fullfilepath):
                        isValid, match = validfilefunc(basename)
                        if isValid:
                            yield fullfilepath, basename, match
    else:
        for basename in file_sorter(os.listdir(root)):
            fullfilepath = os.path.join(root, basename)
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


def memoize_volatile(obj):
    cache = obj.cache = {}

    @functools.wraps(obj)
    def memoizer(*args, **kwargs):
        key = str(args) + str(kwargs)
        if key not in cache:
            cache[key] = obj(*args, **kwargs)
        return cache[key]
    return memoizer


# TODO: Make serialization method, compression, etc. optional
# TODO: Add .pkl{protocol ver} extension to pickled files (is this even a good
#       idea?)
def func_memoize_persistent(diskcache_dir=None,
                            diskcache_dir_envvar='PYTHON_CACHE',
                            diskcache_enabled=True, memcache_enabled=True):
    """
    1. Assume any ACTUALLY important arguments are defined with names (i.e.,
       NOT gotten by the function via *args or **kwargs). Hash will be based
       ONLY upon values passed into the name-specified arguments (found via
       inspect.getargspec(f).args).

       NOTE: A "hash key" is named via the following convention:
            (func name)_(func src hash)_(named args hash)
       which has the weakenesses that
       a. Differently-named functions with same functionality will hash
          differently (but this adds a touch of human-readability)
       b. Functions that behave identically but have superficial source-code
          differences will hash differently
       c. Only named arguments are hashed
       d. Hashing of args is via cPickle binary string; objects or sub-objects
          that don't hash nicely will
       e. Memory issues or speed issues could arise if arguments are large
          objects (even if passed by reference, since cPickle will serielize
          the entire object if possible)

    2. Look in memory cache for the hash key; if it's there, simply return (a
       deepcopy of) the value there

    3. If hash key is not in memory cache, look in the first-specified
       directory among {diskcache_dir, $PYTHON_CACHE, $PWD} for a file named
       with the hash key
            diskcache_dir/(hash key)
       If this exists, load the result from the file, populate the hash/return
       value to the local cache, and return this.
       NOTE: If $PYTHON_CACHE is not specified, creates a .pycache directory in
       the current-working directory

    4. If the hash key exists neither in the memory cache nor in the disk cache
       dir, run the function with all arguments (not just named args, so
       including *args and **kwargs), and then store the result in the memory
       cache AND in a file named with the hash key in the first-specified dir
       among {diskcache_dir, $PYTHON_CACHE, $PWD}).
    """
    import jsonpickle

    DCD = diskcache_dir
    CACHE_VARNAME = diskcache_dir_envvar
    DC_ENABLED = diskcache_enabled
    MC_ENABLED = memcache_enabled
    def decorator(func):
        if not DC_ENABLED and not MC_ENABLED:
            return func

        # Create memory cache as a dictionary
        memcache = func.memcache = {}
        diskcache_enabled = func.diskcache_enabled = DC_ENABLED
        memcache_enabled = func.memcache_enabled = MC_ENABLED

        # Define path to, and create if necessary, cache directory on disk
        if diskcache_enabled:
            if DCD:
                diskcache_dir = func.diskcache_dir = DCD
            else:
                diskcache_dir = func.diskcache_dir = os.path.expandvars(
                    '$'+CACHE_VARNAME
                )
                if diskcache_dir == '$'+CACHE_VARNAME:
                    # Revert to .pycache directory in local dir
                    cwd = os.getcwd()
                    diskcache_dir = func.diskcache_dir = os.path.join(
                        cwd, '.pycache'
                    )
            wstderr('Using dir \'' + os.path.abspath(diskcache_dir)
                    + '\' for caching results to disk.')
            if not os.path.exists(diskcache_dir):
                wstderr(' Dir does not exist. Creating... ')
                try:
                    os.makedirs(diskcache_dir)
                except OSError as err:
                    if err.errno != 17:
                        wstderr(' failed.\n')
                        raise err
                else:
                    wstderr(' success.')
            wstderr('\n')
            if not os.path.isdir(diskcache_dir):
                wstderr('Cache path \'' + diskcache_dir
                        + '\' does not point to a valid directory. Disk'
                        ' caching disabled.\n')
                diskcache_enabled = func.diskcache_enabled = False

        # Retrieve info about func & its args
        func_name = func.func_name
        func_src = inspect.getsource(func)

        func_hash = func.func_hash = '_'.join((func_name, my_hash(func_src)))
        argspec = func.argspec = inspect.getargspec(func)

        del func_name, func_src

        @functools.wraps(func)
        def memoizer(*args, **kwargs):
            force_execution = False
            if (kwargs.has_key('MEMO_FORCE_EXECUTION') and
                    kwargs['MEMO_FORCE_EXECUTION']):
                force_execution = True

            #
            # Stringify only the args defined by name in the function's arg
            # spec...
            #

            # Populate default arguments & their values. May be overwritten
            # below.
            named_args = {}
            if argspec.defaults:
                named_args = {arg: dflt for arg, dflt in
                              zip(argspec.args[-len(argspec.defaults):],
                                  argspec.defaults)}

            tmp_argspec_args = copy.deepcopy(argspec.args)
            argsspecd_n = 0
            for arg in args:
                if not tmp_argspec_args:
                    break
                refarg = tmp_argspec_args.pop(0)
                named_args[refarg] = arg
                argsspecd_n += 1
            for refarg in tmp_argspec_args:
                if refarg in kwargs:
                    #print '    populating refarg: ', refarg
                    named_args[refarg] = kwargs[refarg]
                    argsspecd_n += 1
            #print ' ** argsspecd_n:', argsspecd_n
            # TODO: set_encoder_options('simplejson', sort_keys=True, indent=2)
            # ... or use faster backend, like ujson? but does that sort keys?

            requires_recompute = False
            args_bstr = b''
            for arg in sorted(named_args.keys()):
                try:
                    arg_bstr = cPickle.dumps(named_args[arg],
                                             protocol=cPickle.HIGHEST_PROTOCOL)
                except:
                    wstderr(' ** failed to hash argument ' +  arg
                            + ' via cPickle.dumps; trying jsonpickle' + '\n')
                    try:
                        arg_bstr = jsonpickle.encode(named_args,
                                                     unpicklable=False)
                    except:
                        wstderr(
                            ' ** failed to hash argument ' +  arg
                            + ' via jsonpickle; forcing recomputation of'
                            ' function ' + func_hash + '\n'
                        )
                        requires_recompute = True
                        break
                args_bstr += arg + arg_bstr

            if requires_recompute:
                return func(*args, **kwargs)

            del named_args
            arg_hash = my_hash(args_bstr)
            #print 'func_hash:', func_hash, 'arg_hash:', arg_hash
            del args_bstr

            key = '_'.join((func_hash, arg_hash))
            fpath = os.path.join(diskcache_dir, key)

            if (not(memcache_enabled)
                    or (memcache_enabled and (key not in memcache))
                    or force_execution):
                #... need to check disk cache, or re-run the function
                in_diskcache = False
                if not force_execution and diskcache_enabled:
                    if os.path.exists(fpath):
                        f = file(fpath, 'rb')
                        try:
                            ret = cPickle.load(f)
                        except:
                            pass
                        else:
                            in_diskcache = True
                        finally:
                            f.close()
                if not in_diskcache:
                    ret = func(*args, **kwargs)
                    if diskcache_enabled:
                        with file(fpath, 'wb') as f:
                            cPickle.dump(ret, f,
                                         protocol=cPickle.HIGHEST_PROTOCOL)
                if memcache_enabled:
                    memcache[key] = ret
            else:
                ret = memcache[key]
            return copy.deepcopy(ret)
        return memoizer
    return decorator


# This regex matches signed, unsigned, and scientific-notation (e.g. "1e10")
# numbers.
NUMBER_RESTR = r'((?:-|\+){0,1}[0-9.]+(?:e(?:-|\+)[0-9.]+){0,1})'
NUMBER_RE = re.compile(NUMBER_RESTR, re.IGNORECASE)

# This regex
# The starting number
# Optional range, e.g., --10 (which means "to negative 10"); in my
# interpretation, the "to" number should be *INCLUDED* in the list
# If there's a range, optional stepsize, e.g., --10 (which means "to negative
# 10")
HRGROUP_RESTR = (
    NUMBER_RESTR +
    r'(?:-' + NUMBER_RESTR +
    r'(?:\:' + NUMBER_RESTR + r'){0,1}' +
    r'){0,1}'
)
HRGROUP_RE = re.compile(HRGROUP_RESTR, re.IGNORECASE)


def num2floatOrInt(num):
    try:
        if int(num) == float(num):
            return int(num)
    except:
        pass
    return float(num)


def isint(num):
    """Test whether a number is *functionally* an integer"""
    try:
        int(num) == float(num)
    except ValueError:
        return False
    return True


def hrgroup2list(hrgroup):
    # Strip all whitespace from the group string
    hrgroup = ''.join(hrgroup.split())
    if (hrgroup is None) or (hrgroup == ''):
        return []
    numstrs = HRGROUP_RE.match(hrgroup).groups()
    range_start = num2floatOrInt(numstrs[0])
    # If no range is specified, just return the number
    if numstrs[1] is None:
        return [range_start]
    range_stop = num2floatOrInt(numstrs[1])
    step = 1
    if numstrs[2] is not None:
        step = num2floatOrInt(numstrs[2])
    all_ints = isint(range_start) and isint(step)
    # Make an *INCLUSIVE* list
    lst = np.arange(range_start, range_stop+step, step)
    if all_ints:
        lst = [int(item) for item in lst]
    return lst


WS_RE = re.compile(r'\s')
def hrlist2list(hrlst):
    groups = re.split(r'[,; _]+', WS_RE.sub('', hrlst))
    lst = []
    if not groups:
        return lst
    for g in groups:
        lst.extend(hrgroup2list(g))
    return lst


def hrlol2lol(hrlol):
    supergroups = re.split(r'[;]+', hrlol)
    return [hrlist2list(group) for group in supergroups]


# Below is adapted by me to make scientific notation work correctly from Scott
# B's adaptation to Python 2 of Rik Poggi's answer to his question:
# stackoverflow.com/questions/9847601/convert-list-of-numbers-to-string-ranges
def hrlist_formatter(start, end, step):
    if step == 1:
        return '{}-{}'.format(start, end)
    return '{}-{}:{}'.format(start, end, step)


def list2hrlist(lst):
    if np.isscalar(lst):
        lst = [lst]
    lst = sorted(lst)
    tol = np.finfo(np.float).resolution
    n = len(lst)
    result = []
    scan = 0
    while n - scan > 2:
        step = lst[scan + 1] - lst[scan]
        if not np.isclose(lst[scan + 2] - lst[scan + 1], step, rtol=tol):
            result.append(str(lst[scan]))
            scan += 1
            continue

        for j in xrange(scan+2, n-1):
            if not np.isclose(lst[j+1] - lst[j], step, rtol=tol):
                result.append(hrlist_formatter(lst[scan], lst[j], step))
                scan = j+1
                break
        else:
            result.append(hrlist_formatter(lst[scan], lst[-1], step))
            return ','.join(result)

    if n - scan == 1:
        result.append(str(lst[scan]))
    elif n - scan == 2:
        result.append(','.join(itertools.imap(str, lst[scan:])))

    return ','.join(result)


def hrbool2bool(s):
    s = str(s).strip()
    if s.lower() in ['t', 'true', '1', 'yes', 'one']:
        return True
    elif s.lower() in ['f', 'false', '0', 'no', 'zero']:
        return False
    raise ValueError('Could not parse input into bool: ' + s)


def two_bad_seeds(badseed1, badseed2):
    """badseed1 >= 0; badseed2 >= 1"""

    # init generator with bad seed
    np.random.seed(badseed1)
    # blow through some states to increase entropy
    np.random.randint(-1e9, 1e9, 1e5)
    # grab a good seed from a randomly-generated integer
    goodseed1 = np.random.randint(0, 2**63-1, 1)
    #print goodseed1
    # seed the generator with the good seed
    np.random.seed(goodseed1)
    # blow through some states
    np.random.randint(-1e9, 1e9, 1e5)
    # pick the final good seed from the badseed2-nd number generated
    goodseed2 = np.random.randint(0, 2**63-1, badseed2)
    #print goodseed2
    goodseed2 = goodseed2[-1]
    # set the state of the generator
    np.random.seed(goodseed2)
    # blow through some states
    np.random.randint(-1e9, 1e9, 1e5)
    # Now you're ready to go!
    return np.random.get_state()


def n_bad_seeds(*args):
    """All seeds must be integers in the range [0, 2**32)"""
    np.random.seed(args[0])
    for _, badseed in enumerate(args):
        next_seed_set = np.random.randint(0, 2**32, badseed+1)
        # init generator with bad seed
        np.random.seed(next_seed_set[badseed])
        # blow through some states to increase entropy
        np.random.randint(-1e9, 1e9, 1e5)
        # grab a good seed (the next randomly-generated integer)
        goodseed = np.random.randint(0, 2**32, 1)
        # seed the generator with the good seed
        np.random.seed(goodseed)
        # blow through some states to increase entropy
        np.random.randint(-1e9, 1e9, 1e5)
    return np.random.get_state()


def samplesFilename(n_dim, n_samp, rand_set_id=0, crit='m', iterations=5,
                    prefix=None, suffix=None, extn='.pkl'):
    if isinstance(crit, basestring):
        crit = crit.lower().strip()
    if (crit is None) or crit == '':
        crit = None
        crit_lab = 'randomized'
        iter_lab = ''
    elif crit in ['c', 'center']:
        crit = 'c'
        crit_lab = 'center'
        iter_lab = ''
    elif crit in ['m', 'maximin']:
        crit = 'm'
        crit_lab = 'maximin'
        iter_lab = '_%diter' % iterations
    elif crit in ['cm', 'centermaximin']:
        crit = 'cm'
        crit_lab = 'centermaximin'
        iter_lab = '_%diter' % iterations
    elif crit in ['corr', 'correlate']:
        crit_lab = 'corr'
        iter_lab = '_%diter' % iterations
    else:
        raise ValueError('Unrecognized crit for pyDOE.lhs: "%s"' % (crit,))
    fname = 'samps_%dD_%s%s_%dsamples_setnum%d' % \
            (n_dim, crit_lab, iter_lab, n_samp, rand_set_id)
    if prefix:
        fname = prefix + '_' + fname
    if suffix:
        fname = fname + '_' + suffix
    return fname + extn


# Retrieve random sampling params in range [0, 1], 3-dim parameter cube
def sampleHypercube(n_dim, n_samp, rand_set_id=0, crit='m', iterations=5,
                    rdata_dir='~/cowen/data/random'):
    """Load (if file exists) or generate samples from within hypercube using
    Latin hypercube sampling

    Requires pyDOE to generate new samples.
    """
    fname = samplesFilename(n_dim=n_dim,
                            n_samp=n_samp,
                            rand_set_id=rand_set_id,
                            crit=crit,
                            iterations=iterations)
    rdata_dir = os.path.expandvars(os.path.expanduser(rdata_dir))
    fpath = os.path.join(rdata_dir, fname)

    if os.path.exists(fpath):
        samps = fileio.from_file(fpath)
    else:
        logging.info('File not found. Generating new set of samples & saving'
                     ' result to "%s"', fpath)
        import pyDOE
        mkdir(rdata_dir)
        # Set a deterministic random state based upon the critical hypercube
        # sampling parameters specified
        n_bad_seeds(n_dim, n_samp, rand_set_id)
        samps = pyDOE.lhs(n=n_dim, samples=n_samp, criterion=crit,
                          iterations=iterations)
        fileio.to_file(samps, fpath)
    return samps


def linExtrap(x, y, xmin, xmax, const_low=False, const_high=False):
    from scipy import interpolate
    x = np.array(x)
    y = np.array(y)
    sort_ind = np.argsort(x)
    x = x[sort_ind]
    y = y[sort_ind]
    """
    set x0 = 0 and y0 = 0, then y' = m*x'; y' = y-y0, x' = x-x0
    y-y0 = m*(x-x0)
    y = m*(x-x0) + y0
    y = (y1-y0)/(x1-x0) * (x-x0) + y0
    """
    if xmin < min(x):
        if const_low:
            y_new = y[0]
        else:
            y_new = (y[1]-y[0])/(x[1]-x[0]) * (xmin-x[0]) + y[0]
        x = np.concatenate(([xmin], x))
        y = np.concatenate(([y_new], y))
    if xmax > max(x):
        if const_high:
            y_new = y[-1]
        else:
            y_new = (y[-1]-y[-2])/(x[-1]-x[-2]) * (xmax-x[-2]) + y[-2]
        x = np.concatenate((x, [xmax]))
        y = np.concatenate((y, [y_new]))
    return interpolate.interp1d(x=x, y=y, kind='linear', copy=False,
                                bounds_error=True, fill_value=np.nan,
                                assume_sorted=True)


def rangeBelowThresh(x, y, y_thresh):
    """Simplistic function that linearly interpolates between points to find
    x-bounds of regions where y drops below y_thresh.

    Note: this is not robust to things like repeated zeros; y data must
    straddle zero OR start or end below zero in order for this to give sensible
    results.
    """
    # Convert to numpy arrays
    x = np.array(x)
    y = np.array(y)

    # Algo only works if data is sorted (ascending) in x
    sortind = np.argsort(x)
    x = x[sortind]
    y = y[sortind]

    boolind = y <= y_thresh
    dbi = np.diff(np.concatenate(([0], boolind, [0])))
    left_inds = np.where(dbi > 0)[0]
    right_inds = np.where(dbi < 0)[0] - 1
    ranges = []
    for (left_n, right_n) in zip(left_inds, right_inds):
        if left_n == 0:
            x_left = x[left_n]
        else:
            x0 = x[left_n-1]
            y0 = y[left_n-1]
            x1 = x[left_n]
            y1 = y[left_n]
            x_left = (y_thresh - y0) * (x1-x0)/(y1-y0) + x0

        if right_n == len(x)-1:
            x_right = x[right_n]
        else:
            x0 = x[right_n]
            y0 = y[right_n]
            x1 = x[right_n+1]
            y1 = y[right_n+1]
            x_right = (y_thresh - y0) * (x1-x0)/(y1-y0) + x0
        ranges.append((x_left, x_right))

    return ranges


def test_rangeBelowThresh():
    # pylint: disable=bad-whitespace
    x = [0, 1, 2, 3, 4, 5, 6]
    y = [3, 1,-1,-2,-1, 1,-1]
    assert rangeBelowThresh(x, y, y_thresh=0) == [(1.5,4.5), (5.5,6)]

    x = [ 0, 1, 2, 3, 4, 5, 6]
    y = [-1, 1,-1,-2,-1, 1,-1]
    assert rangeBelowThresh(x, y, y_thresh=0) == [(0,0.5), (1.5,4.5), (5.5,6)]

    x = [ 0, 1, 2, 3, 4, 5, 6]
    y = [ 1, 1, 1, 2, 1, 1, 1]
    assert rangeBelowThresh(x, y, y_thresh=0) == []


def home(d=None):
    if d is None:
        return os.path.expanduser('~')
    return os.path.join(os.path.expanduser('~'), d)


def pdSafe(s):
    s = s.translate(None, '\\/ ?!@#$%^&*()-+=\`~|][{}<>,')
    s = s.replace('.', '_')
    return s


def makeFuncMappable(func, *args, **kwargs):
    """Generally, `map` doesn't take scalar arguments (Pandas `map` is more
    restrictive, but even Python's `map` is restricted to all arguments having
    same length as the first argument -- which is the iterable being mapped).

    This function returns a version of the passed `func` that only takes one
    argument, running the original `func` with that argument and all other args
    and kwargs specified to makeFuncMappable.
    """
    def mappableFunc(arg0):
        return func(arg0, *args, **kwargs)
    return mappableFunc


def applyParallel(grouped_df, func, cpucount=None, chunksize=None):
    """User Pietro Battiston's solution from
    http://stackoverflow.com/questions/26187759/parallelize-apply-after-pandas-groupby
    """
    import pandas as pd
    if cpucount is None:
        cpucount = multiprocessing.cpu_count()
    #with multiprocessing.Pool(cpucount) as pool:
    pool = multiprocessing.Pool(cpucount)
    ret_list = pool.imap(func, [group for _, group in grouped_df],
                         chunksize=chunksize)
    return pd.concat(ret_list)
