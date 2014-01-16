import numpy as np

def customTicks(data,
                forceMin=np.nan, forceMax=np.nan,
                forceMinTick=np.nan, forceMaxTick=np.nan,
                desiredNumTicks=5,
                displayFirst=True, displayLast=True,
                multipleCandidates=[1,2,2.5,5]):
    
    multipleCandidates = np.array(multipleCandidates)
    
    if np.isnan(forceMaxTick):
        maximum = np.max(data)
    else:
        maximum = forceMaxTick

    if np.isnan(forceMinTick):
        minimum = np.min(data)
    else:
        minimum = forceMinTick


    ts_exact = (maximum-minimum)/desiredNumTicks
    if ts_exact == 0:
        ts_exact = 1
        desiredNumTicks = 1
    ts_nearest = np.round(ts_exact,-int(np.floor(np.log10(ts_exact))))
    ts_tenPower = np.floor(np.log10(ts_nearest))
    ts_nTens = ts_nearest/10**ts_tenPower
    multInd = np.argmin(np.abs(ts_nTens - multipleCandidates))
    multiple = multipleCandidates[multInd]
    tickSpacing = multiple*10**ts_tenPower
    
    minimum = np.floor(minimum/tickSpacing)*tickSpacing
    maximum = np.ceil(maximum/tickSpacing)*tickSpacing

    tickValues = np.arange(minimum, maximum+tickSpacing/10, tickSpacing)
    if (not displayFirst) and (len(tickValues) > 1):
        tickValues = tickValues[1:]
    if (not displayLast) and (len(tickValues) > 1):
        tickValues = tickValues[:-1]
    #numTicks = len(tickValues)

    if not np.isnan(forceMin):
        minimum = forceMin
        tickValues = [tv for tv in tickValues if tv >= forceMin]
    if not np.isnan(forceMax):
        maximum = forceMax
        tickValues = [tv for tv in tickValues if tv <= forceMax]

    return minimum, maximum, tickValues 

