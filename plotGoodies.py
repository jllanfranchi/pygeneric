from __future__ import division

import colorsys
import numpy as np
from pylab import *
from fractions import Fraction
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, LogNorm, PowerNorm

from smartFormat import smartFormat, TeX, numFmt, numFmt2

MARKERS = ['.', 'v', 'o', '*', '+', 'D', '^', 's', 'p', 'x', '<', '>', 'h', 'H', 'd', '|', '_']

DARK_BLUE = (0.0, 0.0, 0.7)
DARK_RED =  (0.7, 0.0, 0.0)

LIGHT_BLUE = (0.4, 0.4, 0.8)
LIGHT_RED =  (0.8, 0.4, 0.4)


'''
Use the following as:
#mpl.rc('axes', color_cycle=colorCycleOrthog)
source: http://stackoverflow.com/questions/470690/how-to-automatically-generate-n-distinct-colors
... but modified somewhat from that!
'''
colorCycleOrthog = (
    '#000000', #  0  Black
    '#803E75', #  2  Strong Purple
    '#FF6800', #  3  Vivid Orange
    '#8A9DD7', #  4  Very Light Blue
    '#FFB300', #  1  Vivid Yellow
    '#C10020', #  5  Vivid Red
    '#CEA262', #  6  Grayish Yellow
    '#817066', #  7  Medium Gray

    #The following will not be good for people with defective color vision
    '#007D34', #  8  Vivid Green
    '#F6768E', #  9  Strong Purplish Pink
    '#00538A', # 10  Strong Blue
    '#93AA00', # 11  Vivid Yellowish Green
    '#593315', # 12  Deep Yellowish Brown
    '#F14AD3', # 13  PINK/Magenta!  (used to be: #F13A13, Vivid Reddish Orange
    '#53377A', # 14  Strong Violet
    '#FF8E00', # 15  Vivid Orange Yellow
    '#54BF00', # 16  Vivid Greenish Yellow
    '#0000A5', # 17  BLUE!
    '#7F180D', # 18  Strong Reddish Brown

    #'#F13A13', # 13  Vivid Reddish Orange
    #'#B32851', # 16  Strong Purplish Red
    #'#FF7A5C', # 19  Strong Yellowish Pink
)

def invertColor(c):
    r, g, b, a = mpl.colors.colorConverter.to_rgba(c)
    if len(c) == 3:
        return (1-r, 1-g, 1-b)
    return (1-r, 1-g, 1-b, a)
    #if isinstance(c, basestring):
    #    c = c.replace('#', '')
    #    r, g, b = (int(c[2*i:2*i+2], 16) for i in range(3))
    #    ri = 255-r
    #    gi = 255-g
    #    bi = 255-b
    #    return '#%02x%02x%02x'%(ri,gi,bi)

def hsvaFact(c, hf=1.0, sf=1.0, vf=1.0, af=1.0, clip=True):
    r, g, b, a = mpl.colors.colorConverter.to_rgba(c)
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    ri, gi, bi = colorsys.hsv_to_rgb(h*hf, s*sf, v*vf)
    if clip:
        # Clip all values to range [0,1]
        result = (np.clip(ri,0,1), np.clip(gi,0,1), np.clip(bi,0,1),
                  np.clip(a*af,0,1))
    else:
        # Rescale to fit largest within [0,1]; if all of r,g,b fit in this
        # range, do nothing
        maxval = max(ri, gi, bi)
        # Scale colors if one exceeds range
        if maxval > 1:
            ri /= maxval
            gi /= maxval
            bi /= maxval
        # Clip alpha to range [0,1]
        alpha = np.clip(a*af)
        result = (ri, gi, bi, alpha)
    return result

colorCycleRainbow = (
    '#FF1008',
    '#FF5C2F',
    '#FFA055',
    '#DED579',
    '#ACF59A',
    '#7AFFB7',
    '#48F1D0',
    '#17CBE4',
    '#1C93F3',
    '#4E4DFC',
    '#8000FF',
)

human_safe = ListedColormap(colorCycleOrthog, name='human_safe')
my_rainbow = ListedColormap(colorCycleRainbow, name='my_rainbow')


def grayify_cmap(cmap):
    """Return a grayscale version of the colormap
    From: https://jakevdp.github.io/blog/2014/10/16/how-bad-is-your-colormap/"""
    cmap = plt.cm.get_cmap(cmap)
    colors = cmap(np.arange(cmap.N))
    
    # convert RGBA to perceived greyscale luminance
    # cf. http://alienryderflex.com/hsp.html
    RGB_weight = [0.299, 0.587, 0.114]
    luminance = np.sqrt(np.dot(colors[:, :3] ** 2, RGB_weight))
    colors[:, :3] = luminance[:, np.newaxis]
   
    if isinstance(cmap, LinearSegmentedColormap): 
        return cmap.from_list(cmap.name + "_grayscale", colors, cmap.N)
    elif isinstance(cmap, ListedColormap): 
        return ListedColormap(colors=colors, name=cmap.name + "_grayscale")


def show_colormap(cmap):
    """From: https://jakevdp.github.io/blog/2014/10/16/how-bad-is-your-colormap/"""
    im = np.outer(np.ones(100), np.arange(1000))
    fig, ax = plt.subplots(2, figsize=(6, 1.5),
                           subplot_kw=dict(xticks=[], yticks=[]))
    fig.subplots_adjust(hspace=0.1)
    ax[0].imshow(im, cmap=cmap)
    ax[1].imshow(im, cmap=grayify_cmap(cmap))


def plotColorCycle(color_cycle=colorCycleOrthog):
    N = len(color_cycle)
    x = np.linspace(0,2*np.pi,100)
    f = plt.figure(333)
    clf()
    ax = f.add_subplot(111)
    [ax.plot(x,np.cos(x-2*pi/N*n), lw=3, label=format(n,'2d')+': '+color_cycle[n][1:],color=color_cycle[n]) for n in range(N)]
    plt.legend(loc='center right')
    ax.set_xlim([0, 8.2])
    ax.set_ylim([-1.1, 1.1])
    plt.tight_layout()


def plotDefaults():
    plt.ion()
    mpl.rc('font', **{'family':'serif', 'weight':'normal', 'size': 16})
    mpl.rc('axes', color_cycle=human_safe.colors)
    #generateColorCycle(n_colors=6)


def generateColorCycle(cmap=mpl.cm.brg, n_colors=8, set_it=True):
    cmap_indices = np.array(
        np.round(np.arange(0,n_colors)*(cmap.N-1)/(n_colors-1)),
        dtype=int)
    step_size = int(np.floor(cmap.N/n_colors))
    color_cycle = [ "#%0.2X%0.2X%0.2X" % tuple(np.round(c[0:3]*255))
         for c in cmap(cmap_indices) ]
    if set_it:
        mpl.rc('axes', color_cycle=color_cycle)
    return color_cycle


def rugplot(a, y0, dy, ax, **kwargs):
    return ax.plot([a,a], [y0, y0+dy], **kwargs)


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
        plt.draw()
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
                squareMag=True,
                magScale='log', phaseScale='deg', realScale='linear',
                imagScale='linear', freqScale='log', unwrapPhase=False,
                fignum=301):

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
        if squareMag:
            M = magSq
            ylab = r"Mag$^2$"
        else:
            M = np.sqrt(magSq)
            ylab = r"Mag"
        plotN += 1
        kwargs = plot_kwargs.pop(0)
        ax = fig.add_subplot(nPlots, 1, plotN)
        ax.plot(f, M, label=label, **kwargs)
        ax.set_ylabel(ylab)
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
        if unwrapPhase:
            phi = np.unwrap(phi) #, np.pi*(1-1/10))
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


def plotMatrix(tuplesDict, labelsList):
    """From: 
    http://fromthepantothefire.com/matplotlib/rock_paper_scissors.py"""
    # list of string labels for rows/columns and
    # data in dictionary of tuples of these labels (row_label,col_label)
    
    # Map text labels to index used on plot
    # this is convenient if you want to reorganize the display order
    # just update the labelsList order.
    labelNameToIndex = {}
    for i,lab in enumerate(labelsList):
        labelNameToIndex[lab] = i

    # number of rows and columns
    numLabels = len(labelsList)

    #create a list of data points
    xyz = []
    for t in tuplesDict:
        x = labelNameToIndex[t[1]]
        # y values are reversed so output oriented the way I
        # think about matrices (0,0) in upper left.
        y = numLabels -1 - labelNameToIndex[t[0]]
        
        # extract value and color
        (z,c) = tuplesDict[t]

        xyz.append( (x,y,z,c))

    for x,y,z,c in xyz:
        plt.scatter([x],[y], s= [z], color = c, alpha = 0.8)

    tickLocations = range(numLabels)
    plt.xticks(tickLocations, labelsList, rotation = 90)
    # reverse the labels for y axis to match the data
    plt.yticks(tickLocations, labelsList[::-1]) 
    # set the axis 1 beyond the data so it looks good.
    plt.axis([-1, numLabels, -1, numLabels])


def removeBorder(axes=None, top=False, right=False, left=True, bottom=True):
    """
    Minimize chartjunk by stripping out unnecessary plot borders and axis ticks
    
    The top/right/left/bottom keywords toggle whether the corresponding plot border is drawn

    from ChrisBeaumont,
        https://github.com/cs109/content/blob/master/README.md
    """
    ax = axes or plt.gca()
    ax.spines['top'].set_visible(top)
    ax.spines['right'].set_visible(right)
    ax.spines['left'].set_visible(left)
    ax.spines['bottom'].set_visible(bottom)
    
    #turn off all ticks
    ax.yaxis.set_ticks_position('none')
    ax.xaxis.set_ticks_position('none')
    
    #now re-enable visibles
    if top:
        ax.xaxis.tick_top()
    if bottom:
        ax.xaxis.tick_bottom()
    if left:
        ax.yaxis.tick_left()
    if right:
        ax.yaxis.tick_right()


def findRenderer(fig):
    '''From http://stackoverflow.com/questions/22667224/matplotlib-get-text-bounding-box-independent-of-backend'''
    if hasattr(fig.canvas, "get_renderer"):
        #Some backends, such as TkAgg, have the get_renderer method, which 
        #makes this easy.
        renderer = fig.canvas.get_renderer()
    else:
        #Other backends do not have the get_renderer method, so we have a work 
        #around to find the renderer.  Print the figure to a temporary file 
        #object, and then grab the renderer that was used.
        #(I stole this trick from the matplotlib backend_bases.py 
        #print_figure() method.)
        import io
        fig.canvas.print_pdf(io.BytesIO())
        renderer = fig._cachedRenderer
    return(renderer)


class ScaledMaxNLocator(mpl.ticker.MaxNLocator):
    def __init__(self, scale, *args, **kwargs):
        super(ScaledMaxNLocator, self).__init__(**kwargs)
        self.scale = scale

    def __call__(self):
        vmin, vmax = self.axis.get_view_interval()
        #print self.scale, vmin, vmax, [float(tl)/float(self.scale) for tl in self.tick_values(vmin*self.scale, vmax*self.scale)]
        return [float(tl)/float(self.scale) for tl in self.tick_values(vmin*self.scale, vmax*self.scale)]


def logticks_format(value, index):
    """
    By Francesco Montesano
    http://stackoverflow.com/questions/19239297/matplotlib-bad-ticks-labels-for-loglog-twin-axis
    This function decompose value in base*10^{exp} and return a latex string.
    If 0<=value<99: return the value as it is.
    if 0.1<value<0: returns as it is rounded to the first decimal
    otherwise returns $base*10^{exp}$
    I've designed the function to be use with values for which the decomposition
    returns integers
    
    Use as:
        import matplotlib.ticker as ticker
        subs = [1., 3., 6.]
        ax.xaxis.set_minor_locator(ticker.LogLocator(subs=subs))
        ax.xaxis.set_major_formatter(ticker.NullFormatter())
        ax.xaxis.set_minor_formatter(ticker.FuncFormatter(ticks_format))
    """
    exp = np.floor(np.log10(value))
    base = value/10**exp
    if exp == 0 or exp == 1:
        return '${0:d}$'.format(int(value))
    if exp == -1:
        return '${0:.1f}$'.format(value)
    else:
        if base == 1:
            return '$10^{{{0:d}}}$'.format(int(exp))
        return '${0:d}\\times10^{{{1:d}}}$'.format(int(base), int(exp))

def smartticks_format(**kwargs):
    sfmtargs = dict(sciThresh=[4,4],
                    sigFigs=3,
                    keepAllSigFigs=False)
    if not kwargs is None:
        sfmtargs.update(kwargs)
    def smart_ticks_formatter(value, index):
        return smartFormat(value, **sfmtargs)
    return smart_ticks_formatter


def fmtstr_format(fmt):
    def fixed_ticks_formatter(value, index):
        return TeX(format(value, fmt))
    return fixed_ticks_formatter


def fractticks_format(DENOM_LIMIT):
    def fract_ticks_formatter(value, index):
        f = Fraction(value).limit_denominator(DENOM_LIMIT)
        if f.denominator == 1:
            return r'$' + format(f.numerator,'d') + r'$'
        return r'$' + format(f.numerator,'d') + r'/' + format(f.denominator,'d') + r'$'
    return fract_ticks_formatter


def maskZeros(H):
    return H == 0
