from __future__ import division
from __future__ import with_statement

import numpy as np
from pylab import ion
import matplotlib as mpl
from matplotlib.path import Path
from matplotlib import pyplot as plt
from matplotlib import animation
from scipy.optimize import curve_fit
from scipy.weave import inline, converters

import sys
import time
import cPickle as pickle

from JSAnimation import IPython_display, HTMLWriter

from smartFormat import smartFormat
from plotGoodies import plotDefaults

plotDefaults()
