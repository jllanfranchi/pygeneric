# -*- coding: iso-8859-15 -*-

import numpy as np
import pandas as pd
import multiprocessing
import pathos.multiprocessing as multi


def pdSafe(s):
    '''Transform name into Pandas-safe name (i.e., dot-notation-accessible).'''
    s = s.translate(None, '\\/ ?!@#$%^&*()-+=\`~|][{}<>,')
    s = s.replace('.', '_')
    return s


#def applyParallel(groupedDF, func, cpucount=None, chunksize=None):
#    '''User Pietro Battiston's solution from
#    http://stackoverflow.com/questions/26187759/parallelize-apply-after-pandas-groupby
#    '''
#    if cpucount is None:
#        cpucount = multiprocessing.cpu_count()
#    # Python 3 only?: with multiprocessing.Pool(cpucount) as pool:
#    pool = multiprocessing.Pool(cpucount)
#    #try:
#    ret_list = pool.map(func, [group for name, group in groupedDF], chunksize=chunksize)
#    #except:
#    #    pool.terminate()
#    #    raise
#    return pd.concat(ret_list)


def applyParallel(groupedDF, func, cpucount=None, chunksize=None, nice=False):
    '''Combination of user Pietro Battiston's solution from
    http://stackoverflow.com/a/29281494
    and Mike McKerns answer at http://stackoverflow.com/a/21345423

    This requires that the func return a DataFrame (or indexed Series, if
    that's even possible?)
    '''
    if cpucount is None:
        if nice:
            # Be nice, taking over most but not all of available physical CPU's
            cpucount = int(np.floor(multiprocessing.cpu_count()*0.85))
        else:
            # Be greedy, trying to take over *all* available CPU's
            cpucount = int(np.ceil(multiprocessing.cpu_count()))
    with multi.ProcessingPool(cpucount) as pool:
        pool = multi.ProcessingPool(cpucount)
        ret_list = pool.imap(func,
                            [group for name, group in groupedDF],
                            chunksize=chunksize)
    #if isinstance(ret_list[0], pd.Series):
    #    outDF = pd.concat(ret_list, axis=1).T
    #    return outDF
    #    #outDF.index
    #else:
    return pd.concat(ret_list)


def applymapParallel(groupedDF, func, cpucount=None, chunksize=None):
    '''Combination of user Pietro Battiston's solution from
    http://stackoverflow.com/a/29281494
    and Mike McKerns answer at http://stackoverflow.com/a/21345423

    This differs from applyParallel in that the func should only return a
    scalar or vector (unindexed) result.

    TODO: make this work (only roughed out thus far!)
    '''
    raise NotImplementedError('This function has yet to be fully fleshed out.')

    def metafunc(func):
        def dropinfunc(idx_grp):
            return {idx_grp[0]: func(idx_grp[1])}

    if cpucount is None:
        #cpucount = int(np.ceil(multiprocessing.cpu_count()/2.0*0.75))
        cpucount = int(np.ceil(multiprocessing.cpu_count()))
    with multi.ProcessingPool(cpucount) as pool:
        pool = multi.ProcessingPool(cpucount)
        # The following MUST be one of the ordered pool methods (either `map`
        # or `imap`), # or else reattaching an index will be arbitrary
        ret_list = pool.map(dropinfunc, [idx_grp for idx_grp in groupedDF], chunksize=chunksize)
    if isinstance(ret_list[0], pd.Series):
        outDF = pd.concat(ret_list, axis=1).T
        # TODO: give it an index!
    elif isinstance(ret_list[0], pd.DataFrame):
        return pd.concat(ret_list)
