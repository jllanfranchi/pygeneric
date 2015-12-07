#!/usr/bin/env python

import numpy as np
import constants as const
import matplotlib._cntr as cntr
import matplotlib.path as mplPath


def cart2pol(x, y=None):
    if y is None:
        X = x[:,0]
        Y = x[:,1]
    else:
        X = x
        Y = y
        
    return np.sqrt(X**2 + Y**2), np.arctan2(Y,X)


def sph2cart(r, theta, phi):
    '''
    Arguments
    ---------
    r
        distance from origin
    theta
        inclination/zenith/co-latitude angle (out/down from the +z-axis)
    phi
        azimuth/longitude angle (positive is counter-clockwise about equator)

    Returns
    -------
    x, y, z 
        Cartesian coordinate(s)
    '''
    sintheta = np.sin(theta)
    return r*sintheta*np.cos(phi), r*sintheta*np.sin(phi), r*np.cos(theta)


def pol2cart(r, theta=None):
    if theta is None:
        R = r[:,0]
        THETA = r[:,1]
    else:
        R = r
        THETA = theta
    
    return R*np.cos(THETA), R*np.sin(THETA)


def polyArea(vert):
    '''Vertices must be an N_vert x 2 Numpy array; first vertex need not be
    the same as the last index (i.e., the algorithm "closes" the polygon for
    you).

    See http://paulbourke.net/geometry/polygonmesh for reference'''

    vert1 = np.roll(vert, shift=-1, axis=0)
    xi = vert[:,0]
    yi = vert[:,1]
    xip1 = vert1[:,0]
    yip1 = vert1[:,1]

    return 0.5 * np.sum( xi*yip1 - xip1*yi )


def polyCentroid(vert):
    '''Vertices must be an N_vert x 2 Numpy array; first vertex need not be
    the same as the last index (i.e., the algorithm "closes" the polygon for
    you).

    See http://paulbourke.net/geometry/polygonmesh for reference'''
    if isinstance(vert, mplPath.Path):
        vert = vert.vertices

    vert1 = np.roll(vert, shift=-1, axis=0)
    xi = vert[:,0]
    yi = vert[:,1]
    xip1 = vert1[:,0]
    yip1 = vert1[:,1]
    area = 0.5 * np.sum( xi*yip1 - xip1*yi )
    cx = 1/(6*area) * np.sum( (xi+xip1)*(xi*yip1-xip1*yi) )
    cy = 1/(6*area) * np.sum( (yi+yip1)*(xi*yip1-xip1*yi) )

    return np.array([cx,cy])


def expandPolyApprox(vert, perpDist):
    '''Approximate expansion of polygon by a perpendicular distance on all sides.
    A negative value for perpDist contracts the polygon.'''

    centroid = polyCentroid(vert)

    r0, theta = cart2pol(vert[:,0]-centroid[0], vert[:,1]-centroid[1])

    theta_ = theta.copy()
    theta_.sort()

    #-- find average angle (rays from centroid) separating vertices
    dtheta = np.mean( (theta_ - np.roll(theta_, shift=1, axis=0)) % (2*np.pi) )

    #-- half that angle is the angle to the center of each edge from the
    #   centroid, taken (erroneously, but good enough for now) to be the
    #   perpendicular. Obviously a worse approximation the less regular the
    #   polygon.
    rprime = r0 + perpDist/np.cos(dtheta/2.0)

    xprime, yprime = pol2cart(rprime, theta)

    return np.array([xprime+centroid[0], yprime+centroid[1]]).T


def contractPolyApprox(vert, perpDist):
    '''Convenience function. See expandPolyApprox for details; this just
    does the opposite, feeding -perpDist to that function.'''
    return expandPolyApprox(vert, -perpDist)


def cherenkovAngle(n, speed, light_vac_speed=const.c):
    '''Returns the Cherenkov angle, in radians, for a particle moving in a
    medium with index of refraction n at the given speed.'''
    ## TODO: add warning if the following holds! (but it's here because someone
    ## might say speed = 3e8 instead of specifying const.c
    if speed > light_vac_speed:
        speed = light_vac_speed
    beta = speed / light_vac_speed
    cos_theta = 1/(n*beta)
    return np.arccos(cos_theta)


def polyArea(x,y):
    '''From
    http://stackoverflow.com/questions/24467972/calculate-area-of-polygon-given-x-y-coordinates
    user Mahdi'''
    return 0.5*np.abs( np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)) )


def contourPaths(x, y, z, z_thresh):
    '''Returns list of matplotlib path(s) representing contour line(s) at
    z_thresh. Credit:Ian Thomas-8
    http://matplotlib.1069221.n5.nabble.com/pyplot-Extract-contourset-without-plotting-td15868.html
    '''
    contour_vertices = cntr.Cntr(x, y, z)
    nlist = contour_vertices.trace(z_thresh, z_thresh, 0)
    segments = nlist[:len(nlist)//2]
    paths = [mplPath.Path(s) for s in segments]
    return paths


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    vert = np.array(
      [[ -10.97,    6.72],
       [ -22.23,  -29.97],
       [ -14.31,  -67.24],
       [  15.13,  -99.93],
       [  36.65,  -95.36],
       [  79.69,  -86.21],
       [ 101.21,  -81.63],
       [ 114.81,  -39.79],
       [ 106.89,   -2.52],
       [  77.45,   30.17],
       [  55.93,   25.6 ],
       [  12.89,   16.45],
       [ -10.97,    6.72]])

    c = polyCentroid(vert)
    print c

    f = plt.figure(200)
    f.clf()
    ax = f.add_subplot(111)
    ax.add_patch(plt.Polygon(vert, closed=True, fill=True,
                        facecolor='w', edgecolor='k')
           )
    ax.plot(c[0],c[1], 'ko')
    plt.draw()
    plt.show()
    
    vert = np.array(
      [[ -1,    -1],
       [  1,    -1],
       [  1,     1],
       [ -1,     1]])

    c = polyCentroid(vert)
    print 'square centered at (0,0): computed centroid =', c

    f = plt.figure(201)
    f.clf()
    ax = f.add_subplot(111)
    ax.add_patch(plt.Polygon(vert, closed=True, fill=True,
                        facecolor='w', edgecolor='k')
           )
    ax.plot(c[0],c[1], 'ko')
    plt.draw()
    plt.show()
