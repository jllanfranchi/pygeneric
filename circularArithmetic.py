#!/usr/bin/env python

import numpy as np

#def monotonicUnwrap(data, jump, dataDir=+1):
#    if dataDir > 0:
#        jump = np.abs(jump)
#        return np.concatenate([np.array([data[0]]),
#                               data[1:]+jump*np.cumsum(np.diff(data)<0)])
#    else:
#        jump = -np.abs(jump)
#        return np.concatenate([np.array([data[0]]),
#                               data[1:]+jump*np.cumsum(np.diff(data)>0)])


def circularDifference(a, b, deg=False):
    if deg:
        return ((a-b+180.0) % (2*180.0)) - 180.0
    return ((a-b+np.pi) % (2*np.pi)) - np.pi


def monotonicUnwrap(data, jump, dataDir=+1, returnIndOnly=False):
    if not returnIndOnly:
        if dataDir > 0:
            jump = np.abs(jump)
            return np.concatenate([np.array([data[0]]),
                                   data[1:]+jump*np.cumsum(np.diff(data)<0)])
        else:
            jump = -np.abs(jump)
            return np.concatenate([np.array([data[0]]),
                                   data[1:]+jump*np.cumsum(np.diff(data)>0)])
    else:
        if dataDir > 0:
            dy = np.concatenate([[0],np.array(np.diff(data))])
            rawInds = np.arange(0,len(data),dtype=int)
            flipInds = np.concatenate([[0],np.where(dy < 0)[0]])
            c0 = np.zeros(len(data))
            c0[flipInds[1:]] = np.diff(flipInds)
            c = np.cumsum(c0)
            reformattedInds = rawInds - c
            return reformattedInds
        else:
            dy = np.concatenate([[0],np.array(np.diff(data))])
            rawInds = np.arange(0,len(data),dtype=int)
            flipInds = np.concatenate([[0],np.where(dy > 0)[0]])
            c0 = np.zeros(len(data))
            c0[flipInds[1:]] = np.diff(flipInds)
            c = np.cumsum(c0)
            reformattedInds = rawInds - c
            return reformattedInds

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    ddUp = np.random.randint(-2,50,100)
    dUp = np.cumsum(ddUp)
    dDown = np.random.randint(-100,0,100)
    dUpUW = monotonicUnwrap(dUp, jump=100, dataDir=+1)
    dDownUW = monotonicUnwrap(dDown, jump=100, dataDir=-1)
    indUpUW = monotonicUnwrap(dUp, jump=100, dataDir=+1, returnIndOnly=True)
    indDownUW = monotonicUnwrap(dDown, jump=100, dataDir=-1, returnIndOnly=True)

    fig = plt.figure(200)
    fig.clf()
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212, sharex=ax1)
    ax1.plot(dUp,'b-o')
    ax1.plot(dUpUW,'g-o')
    ax2.plot(indUpUW,'r-o')
    
    fig = plt.figure(201)
    fig.clf()
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212, sharex=ax1)
    ax1.plot(dDown,'b-o')
    ax1.plot(dDownUW,'g-o')
    ax2.plot(indDownUW,'r-o')
    plt.draw()
    plt.show()
