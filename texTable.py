#!/usr/bin/env python


import numpy as np
from smartFormat import smartFormat


def texTable(array, headings=None, tableFormat=None, numFormat=[(1,5)],
             precision=[5], nanSub=['---'], inftyThresh=[np.infty]):

    if not isinstance(array, np.ndarray):
        array = np.array(array)

    nRows = np.shape(array)[0]
    print nRows
    try:
        nCols = np.shape(array)[1]
    except:
        nCols = nRows
        nRows = 1

    #-- Begin environment and specify centering, etc. for each col
    table = r"\begin{tabular}{"
    if tableFormat != None:
        table += tableFormat
    else:
        table += "c"*nCols
    table += r"}" + "\n"

    #-- Add headings if present
    if headings != None:
        table += r" & ".join(headings)
    else:
        table += r" & ".join([r"\ " for n in range(nCols)])

    #-- Add horizontal line
    table += r" \\ \midrule"

    #-- Add table entries
    for rowN in range(nRows):
        table += "\n" + r" & ".join([smartFormat(array[rowN,colN])
                              for colN in range(nCols)])
        table += r"\\ \addlinespace[5pt]"

    if headings == None:
        table += r" \midrule" + "\n"
    else:
        table += "\n"

    #-- Close out environment
    table += r"\end{tabular}" + "\n"

    return table
    #if len(numFormat) == 1:
    #    numFormat = numFormat*nCols
    #if len(precision) == 1:
    #    precision = precision*nCols
    #if len(nanSub) == 1:
    #    nanSub = nanSub*nCols
    #if len(inftyThresh) == 1:
    #    inftyThresh = inftyThresh*nCols

    #for colNum in range(nCols):
    #    vals = array[:,colNum]

    #    allInts = True
    #    for val in vals:
    #        allInts = allInts and isInt(val)

    #    maxEl = np.max(vals)
    #    minEl = np.min(vals)

    #    #-- Compute minimum resolution for discerning elements
    #    v2 = vals.copy()
    #    v2.sort()
    #    aDiff = np.abs(np.diff(v2))
    #    aDiff = aDiff[aDiff>0]
    #    if len(aDiff) > 0:
    #        res = np.min(aDiff[aDiff >
    #                            (10**-(precision+np.floor(np.log10(minEl)))])
    #    else:
    #        res = 0
    #    
    #    #-- Multiplicative dynamic range
    #    MDR = np.ceil(np.log10(maxEl)-np.log10(minEl))

    #    #-- Additive dynamic range
    #    ADR = np.ceil(np.log10(maxEl-minEl))
    #    
    #    dynamicRange = np.ceil(np.log10(maxEl)-np.log10(minEl))
    #    if dynamicRange <= precision[colNum]:
    #        fixExp = True
    #        fixedExp = np.floor(np.log10(minEl))
    #    else:
    #        fixExp = False

if __name__ == "__main__":
    header = r"""\documentclass{article}
    \usepackage{amssymb}
    \usepackage{amsmath}
    \usepackage{booktabs}
    \usepackage[letterpaper,landscape]{geometry}

    \begin{document}{
    \begin{centering}
    """

    a = np.array([np.logspace(-4,10,10),np.logspace(-4,10,10)*10])
    headings = ['Col ' + str(n) for n in range(np.size(a,1))]
    body = texTable(a, headings=headings)
    footer = r"""
    \end{centering}
    }\end{document}"""

    print header, body, footer
