import numpy as np
from pylab import *

'''
Use the following as:
#mpl.rc('axes', color_cycle=colorCycleOrthog)
'''
colorCycleOrthog = [
    '#000000', #Black
    '#FFB300', #Vivid Yellow
    '#803E75', #Strong Purple
    '#FF6800', #Vivid Orange
    '#A6BDD7', #Very Light Blue
    '#C10020', #Vivid Red
    '#CEA262', #Grayish Yellow
    '#817066', #Medium Gray

    #The following will not be good for people with defective color vision
    '#007D34', #Vivid Green
    '#F6768E', #Strong Purplish Pink
    '#00538A', #Strong Blue
    '#53377A', #Strong Violet
    '#FF8E00', #Vivid Orange Yellow
    '#B32851', #Strong Purplish Red
    '#F4C800', #Vivid Greenish Yellow
    '#7F180D', #Strong Reddish Brown
    '#93AA00', #Vivid Yellowish Green
    '#593315', #Deep Yellowish Brown
    '#F13A13', #Vivid Reddish Orange
    '#FF7A5C', #Strong Yellowish Pink
]

colorCycleRainbow = [
    '#8000FF',
    '#4E4DFC',
    '#1C93F3',
    '#17CBE4',
    '#48F1D0',
    '#7AFFB7',
    '#ACF59A',
    '#DED579',
    '#FFA055',
    '#FF5C2F',
    '#FF1008']



def generateColorCycle(cmap, nColors):
    stepSize = int(np.floor(cmap.N/nColors))
    colorCycle = [ "#%0.2X%0.2X%0.2X" % tuple(np.round(c[0:3]*255))
         for c in cmap(range(0,nColors*stepSize+1,stepSize)) ]
    return colorCycle

def peakInfo(xdata, ydata):
    maxvalind = np.argmax(ydata)
    maxx = xdata[maxvalind]
    #halfPower = 
    halfInd = find(ydata < halfPower)

def onpick_peakfind(event):
    '''Use this by:
        >> fig = figure(1)
        >> ax = axis(111)
        >> line, = ax.plot(x,y, picker=5)
        >> fig.canvas.mpl_connect('pick_event', onpick_peakfind)

    '''
    print event, event.canvas

    thisline = event.artist
    vis = thisline.get_visible()
    #-- This function doesn't handle the lines in the legend
    fig = event.canvas.figure
    #leg = fig.
    #print leg.__dict__
    #for child in  leg.get_children():
    #    print "child:", child
    #if thisline in leg.get_lines():
    #    return
    #-- If the line has been made invisible, ignore it (return from function)
    if not vis:
        return
    c = thisline.get_color()
    ls = thisline.get_linestyle()
    lw = thisline.get_linewidth()
    mk = thisline.get_marker()
    mkec = thisline.get_markeredgecolor()
    mkew = thisline.get_markeredgewidth()
    mkfc = thisline.get_markerfacecolor()
    mksz = thisline.get_markersize()

    xdata = thisline.get_xdata()
    ydata = thisline.get_ydata()
    label = thisline.get_label()
    freqrangeind = event.ind
    
    #print 'onpick points:', zip(xdata[ind], ydata[ind])

    #print freqrangeind
    #print ""
    #print ydata[freqrangeind]
    minvalind = np.argmin(ydata[freqrangeind])
    maxvalind = np.argmax(ydata[freqrangeind])

    minval = ydata[freqrangeind][minvalind]
    minx = xdata[freqrangeind][minvalind]

    maxval = ydata[freqrangeind][maxvalind]
    maxx = xdata[freqrangeind][maxvalind]
  
    print ''
    print label
    print 'min:', minval, 'at', minx, 'max:', maxval, 'at', maxx

    halfInd = -1
    maxInd = -1
    try:
        maxInd = find(ydata[freqrangeind] == maxval)
        maxInd = maxInd[0]
        #print maxInd
        maxInd = freqrangeind[0] + maxInd
        halfPower = maxval-10*np.log10(2)
        #quarterind = find(ydata[freqrangeind] < maxval-10*np.log10(4))
        halfInd = find(ydata < halfPower)

        inddiff = halfInd - maxInd
        upperInd = min(halfInd[find(inddiff > 0)])
        lowerInd = max(halfInd[find(inddiff < 0)])

        #print lowerInd, maxInd, upperInd

        yLower = ydata[lowerInd:maxInd+1]
        xLower = xdata[lowerInd:maxInd+1]
        dyLower = max(yLower)-min(yLower)

        yUpper = ydata[maxInd:upperInd+1]
        xUpper = xdata[maxInd:upperInd+1]
        dyUpper = max(yUpper)-min(yUpper)
        
        figure(999)
        clf()
 
        #print ls, lw, mk, mkfc, mksz
        #print l
        #print l.get_markerfacecolor()
        #print l.get_color()
        #l.set_color(c)
        #l.set_linestyle(ls)
        #l.set_linewidth(lw)
        #l.set_marker(mk)
        #l.set_markeredgecolor(mkec)
        #l.set_markeredgewidth(mkew)
        #l.set_markerfacecolor(mkfc)
        #l.set_markersize(mksz)

        peakPlotTitle = title(label, fontsize=14)

        interpKind = 'linear'
        interpLower = interp1d( yLower, xLower, kind=interpKind )
        interpUpper = interp1d( np.flipud(yUpper), np.flipud(xUpper), kind=interpKind )

        lowerHalfPowerFreq = interpLower(halfPower)
        upperHalfPowerFreq = interpUpper(halfPower)

        iyLower = np.arange(min(yLower), max(yLower), dyLower/40)
        ixLower = interpLower(iyLower)

        iyUpper = np.arange(max(yUpper), min(yUpper), -dyUpper/40)
        ixUpper = interpUpper(iyUpper)

        delta_f = upperHalfPowerFreq - lowerHalfPowerFreq
        f0 = xdata[maxInd]
        Q = f0/delta_f
        print 'f0:', f0, 'delta_f:', delta_f, 'pkval:', ydata[maxInd], 'Q:', Q, 'eta:', 1/Q
        
        plot( np.concatenate((ixLower, ixUpper)), np.concatenate((iyLower, iyUpper)), 'b.-', alpha=0.2, linewidth=8 )
        plot( [lowerHalfPowerFreq, upperHalfPowerFreq], [halfPower]*2, 'c-', linewidth=15, alpha=0.25 )
        l, = plot( np.concatenate((xLower, xUpper)),
                  np.concatenate((yLower, yUpper)),
                  color=c, linestyle=ls, linewidth=3, marker=mk,
                  markerfacecolor=mkfc, markersize=mksz, markeredgewidth=mkew,
                  markeredgecolor=mkec )
        text((lowerHalfPowerFreq+upperHalfPowerFreq)/2, halfPower,
                 "FWHM = " + lowPrec(delta_f) + ", Q = " + lowPrec(Q) + r", $\eta$ = " + lowPrec(1/Q),
                 horizontalalignment='center', verticalalignment='center', fontsize=12 )
        draw()
    except:
        pass
    #    raise()
    #    print "failed to find/fit peak", halfInd, maxInd


def onpickLegend_toggle(event):
    try:
        # on the pick event, find the orig line corresponding to the
        # legend proxy line, and toggle the visibility
        legline = event.artist
        origline = lined[legline]
        vis = not origline.get_visible()
        origline.set_visible(vis)
        # Change the alpha on the line in the legend so we can see what lines
        # have been toggled
        if vis:
            legline.set_alpha(1.0)
        else:
            legline.set_alpha(0.2)
        f4.canvas.draw()
    except:
        pass


def complexPlot(f, data, plot_kwargs=None, fig_kwargs=None, label=None,
                title=None, xlabel=None,
                magPlot=True, phasePlot=True, realPlot=True, imagPlot=True,
                magScale='log', phaseScale='deg', realScale='linear',
                imagScale='linear', freqScale='log', fignum=301):

    nPlots = magPlot + phasePlot + realPlot + imagPlot

    #plt.close(fignum)
    if fig_kwargs == None:
        fig = plt.figure(fignum, figsize=(7,2.00*nPlots))
    else:
        fig = plt.figure(fignum, **fig_kwargs)
  
    if plot_kwargs == None:
        plot_kwargs = [{}]*nPlots
    elif isinstance(plot_kwargs, dict):
        plot_kwargs = [plot_kwargs]*nPlots

    #-- Stack plots directly on top of one another
    #plt.subplots_adjust(hspace=0.001)
    
    #fig.clf()
    
    plotN = 0
    axesList = []
    xticklabels = []
    magSq = (np.abs(data))**2
    if magPlot:
        plotN += 1
        kwargs = plot_kwargs.pop(0)
        ax = fig.add_subplot(nPlots, 1, plotN)
        ax.plot(f, magSq, label=label, **kwargs)
        ax.set_ylabel("Mag squared")
        ax.grid(b=True)
        ax.set_yscale(magScale)
        axesList.append(ax)
        if plotN < nPlots:
            xticklabels += ax.get_xticklabels()
        ax.set_xscale(freqScale)
        ax.set_xlim(min(f), max(f))
        if plotN == 1 and title != None:
            ax.set_title(title)
        if label != None:
            ax.legend(loc='best')
    
    if phasePlot:
        plotN += 1
        if plotN == 1:
            sharex = None
        else:
            sharex = axesList[0]
        kwargs = plot_kwargs.pop(0)
        phi = np.arctan2(np.imag(data), np.real(data))
        if phaseScale == 'deg':
            phaseUnits = r"deg"
            phi = phi*180/np.pi
        else:
            phaseUnits = r"rad"
        ax = fig.add_subplot(nPlots, 1, plotN, sharex=sharex)
        ax.plot(f, phi, label=label, **kwargs)
        ax.set_ylabel(r"Phase (" + phaseUnits + r")")
        ax.grid(b=True)
        axesList.append(ax)
        if plotN < nPlots:
            xticklabels += ax.get_xticklabels()
        ax.set_xscale(freqScale)
        ax.set_xlim(min(f), max(f))
        if plotN == 1 and title != None:
            ax.set_title(title)
        if label != None:
            ax.legend(loc='best')
    
    if realPlot:
        plotN += 1
        if plotN == 1:
            sharex = None
        else:
            sharex = axesList[0]
        kwargs = plot_kwargs.pop(0)
        ax = fig.add_subplot(nPlots, 1, plotN, sharex=sharex)
        ax.plot(f, np.real(data), label=label, **kwargs)
        ax.set_ylabel("Real")
        ax.grid(b=True)
        axesList.append(ax)
        if plotN < nPlots:
            xticklabels += ax.get_xticklabels()
        ax.set_xscale(freqScale)
        ax.set_yscale(realScale)
        ax.set_xlim(min(f), max(f))
        if plotN == 1 and title != None:
            ax.set_title(title)
        if label != None:
            ax.legend(loc='best')
     
    if imagPlot:
        plotN += 1
        if plotN == 1:
            sharex = None
        else:
            sharex = axesList[0]
        kwargs = plot_kwargs.pop(0)
        ax = fig.add_subplot(nPlots, 1, plotN, sharex=sharex)
        ax.plot(f, np.imag(data), label=label, **kwargs)
        ax.set_ylabel("Imaginary")
        ax.grid(b=True)
        axesList.append(ax)
        if plotN < nPlots:
            xticklabels += ax.get_xticklabels()
        ax.set_xscale(freqScale)
        ax.set_yscale(imagScale)
        ax.set_xlim(min(f), max(f))
        if plotN == 1 and title != None:
            ax.set_title(title)
        if label != None:
            ax.legend(loc='best')
     
    ax.set_xscale(freqScale)
    if xlabel != None:
        ax.set_xlabel(xlabel)

    #plt.setp(xticklabels, visible=False)

    #fig.tight_layout()

    return fig, axesList
