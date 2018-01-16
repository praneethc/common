#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys,math,os,itertools,logging
import json
import termios, tty
import numpy as np
import plyfile
from datetime import datetime
from plyfile import PlyData, PlyElement


__author__   = ('Kaan Akşit')
__version__  = '0.1'

# Definition to prompt.
def prompt(txt,title='COMMON',logfn='output.log'):
    msg = '[%s] [%s] %s' % (str(datetime.now()),title,txt)
    print(msg)
    logging.basicConfig(level=logging.DEBUG, filename=logfn, filemode="a+", format="%(message)s")
    logging.info(msg)
    return msg

# Definition to save dictionary.
def savedict(dict,fn='./dictionary.txt'):
    json.dump(d, open("text.txt",'w'))
    return True

# Definition to load dictionary.
def loaddict(fn='./dictionary.txt'):
    return json.load(open(fn))

# Definition to generate a plane perpendicular to Z axis.
def generateplane(point,angles=[0.,0.,0.]):
    plane = np.array([
                      [10., 10., 0.],
                      [ 0., 10., 0.],
                      [ 0.,  0., 0.]
                     ])
    for i in range(0,plane.shape[0]):
        plane[i],_,_,_  = rotatepoint(plane[i],angles=angles)
        plane[i]       += point
    return plane

# Definition to generate a rotation matrix along X axis.
def rotmatx(angle):
    angle = np.radians(angle)
    return np.array([
                     [1.,            0.  ,           0.],
                     [0.,  np.cos(angle), np.sin(angle)],
                     [0., -np.sin(angle), np.cos(angle)]
                    ])

# Definition to generate a rotation matrix along Y axis.
def rotmaty(angle):
    angle = np.radians(angle)
    return np.array([
                     [np.cos(angle),  0., np.sin(angle)],
                     [0.,             1.,            0.],
                     [-np.sin(angle), 0., np.cos(angle)]
                    ])

# Definition to generate a rotation matrix along Z axis.
def rotmatz(angle):
    angle = np.radians(angle)
    return np.array([
                     [ np.cos(angle), np.sin(angle), 0.],
                     [-np.sin(angle), np.cos(angle), 0.],
                     [            0.,            0., 1.]
                    ])

# Definition to rotate a given point,
def rotatepoint(point,angles=[0,0,0]):
    rotx = rotmatx(angles[0])
    roty = rotmaty(angles[1])
    rotz = rotmatz(angles[2])
    return np.dot(rotz,np.dot(roty,np.dot(rotx,point))),rotx,roty,rotz

# Sample a circular aperture,
def samplecircular(no,aperture,loc=[0.,0.,0.],angles=[0.,0.,0.]):
    points = []
    for idx in range(0,no[0]):
        for idy in range(0,no[1]):
            point        = np.array([
                                     idy*aperture[0]/2/no[1]*np.cos(2*np.pi*idx/no[0]),
                                     idy*aperture[1]/2/no[1]*np.sin(2*np.pi*idx/no[0]),
                                     0.
                                    ])
            point,_,_,_  = rotatepoint(point,angles=angles)
            point       += loc
            points.append(point)
    return points


# Sample a planar aperture,
def sampleplanar(no,aperture,loc=[0.,0.,0.],angles=[0.,0.,0.]):
    points = []
    for idx in range(0,no[0]):
        for idy in range(0,no[1]):
            point        = np.array([
                                     -aperture[0]/2+idx*aperture[0]/no[0]+aperture[0]/no[0]/2.,
                                     -aperture[1]/2+idy*aperture[1]/no[1]+aperture[1]/no[1]/2.,
                                     0.
                                    ])
            point,_,_,_  = rotatepoint(point,angles=angles)
            point       += loc
            points.append(point)
    return points

# Define sphere for Odak,
def definesphere(var):
    return np.array([
                      var["location"][0],
                      var["location"][1],
                      var["location"][2],
                      var["curvature"],
                     ])

# Definition to generate vector from two points and visualize,
def twopointsvector(ray,port,ipo,drawgl=None,id=0,color=[1.0,1.0,1.0,1.0],debug=False):
    # Create a ray.
    vec0,s      = ray.createvectorfromtwopoints(
                                                (port[0],port[1],port[2]),
                                                (ipo[0],ipo[1],-ipo[2])
                                               )
    p           = np.asarray(port)
    p           = np.resize(p,(3,1))
    # Visual debug.
    if debug == True:
        p0 = np.array([
                       [ipo[0]],
                       [ipo[1]],
                       [ipo[2]],
                      ])
        p1 = np.array([
                       [port[0]],
                       [port[1]],
                       [port[2]],
                      ])
        drawgl.addray(p0,p1,color=color,id=id)
    return vec0

# Definition for an intersection chooser for Odak,
def intersect(ray,vec,surface):
    if ('data' in surface) == False:
        surface = generatesurface(surface)
    if surface["type"] == "sphere":
        return ray.findinterspher(
                                  vec,
                                  surface['data']
                                 )
    if  surface["type"] == "plane":
        return ray.findintersurface(
                                    vec,
                                    (
                                     surface['data'][0],
                                     surface['data'][1],
                                     surface['data'][2]
                                    )
                                   )
    self.prompt("Surface wasn't identified by intersect definition, terminating...")
    return False,False,False

# Definition to generate surface data.
def generatesurface(surface):
    if surface["type"] == "sphere":
        data = definesphere(surface)
        surface["data"] = data
        return surface
    if  surface["type"] == "plane":
        data = generateplane(surface["location"],angles=surface["angles"])
        surface["data"] = data
        return surface
    self.prompt("Surface wasn't identified by surface generation definition, terminating...")
    return False

# Definition for Odak for interaction between  ray and a surface,
def surfaceinteract(ray,vec,n0,n1,surface,id=0,color=[1.,1.,1.,1.],drawgl=None,debug=False):
    # Arange refractive indeces,
    n           = [n0,n1]
    # Find the intersection between ray and a surface,
    dist0,norm0 = intersect(ray,vec,surface)
    if type(dist0) == type(False):
       return False
    # Visual debug,
    if debug == True:
       drawgl.addray(vec[0],norm0[0],color=color,id=id)
    # Refract if you can,
    vec0 = ray.snell(vec,norm0,n[1],n[0])
    if type(vec0) != type(False):
       return vec0,norm0,surface,'refracted'
    elif type(vec0) == type(False):
       vec1 = ray.reflect(vec,norm0)
       return vec1,norm0,surface,'reflected'
    print('Something went wrong with surface intersect')
    return False,False,False

# Definition to generate gaussian kernel,
def gaussian_kernel(size, sizey=None):
    size = int(size)
    if not sizey:
        sizey = size
    else:
        sizey = int(sizey)
    x, y = np.mgrid[-size:size, -sizey:sizey]
    r    = size*8/4
    ry   = sizey*8/4
    g    = np.exp(-(x**2/float(r)+y**2/float(ry)))
    l    = g[int(size/2.):int(size*3./2),int(size/2.):int(size*3./2)]
    return l / l.sum()

# Definition to normalize a given data.
def normalize(data,k=1):
    result = abs(data)*1.
#    result = data.astype(float)
    if np.amax(result) == np.amin(result):
        if np.amax(result) != 0:
            return result/np.amax(result)*k
        elif np.amax(result) == 0:
            return result
    result -= np.amin(result)
    return result/np.amax(result)*k

# Definition to figure out if the point is on the rectangle,
def isitonrectangle(ray,vec,surface):
    if surface["aperture type"] != "rectangular":
        prompt("Surface type is not rectangular.")
        return False
    a      = surface["size"][0]/2.
    b      = surface["size"][1]/2.
    points = [
              [ a,  b, 0.],
              [-a, -b, 0.],
              [-a,  b, 0.],
              [ a, -b, 0.]
             ]
    for id in range(0,4):
        points[id],_,_,_  = rotatepoint(points[id],angles=surface["angles"])
        points[id][0]    += surface["location"][0]
        points[id][1]    += surface["location"][1]
        points[id][2]    += surface["location"][2]
    if ray.isitontriangle(vec[0],points[0],points[1],points[2]) == True or ray.isitontriangle(vec[0],points[0],points[1],points[3]) == True:
       return True
    return False

def convolve_images(Input1,Input2,ResultFilename='ConvolutionResult.png'):
    # Reading the input from the file.
    img1  = scipy.misc.imread(Input1)
    # Reading the input from the file.
    img2  = scipy.misc.imread(Input2)
    # Array to store the result of the convolution.
    result = np.zeros(img1.shape)
    # Convolving the image for each color channel.
    for m in xrange(0,3):
        result[:,:,m] = fftshift(ifft2(fft2(img1[:,:,m])*fft2(img2[:,:,m])).real)
    # Normalizing the output.
    result = result/np.amax(result)
    # Storing the result as a file.
    scipy.misc.imsave(ResultFilename,result)
    return result

# Weighted average and weighted standard deviation.
# This is for 3D points.
# Insipired from https://stackoverflow.com/questions/2413522/weighted-standard-deviation-in-numpy
def weighted_avg_and_std(values, weights):
    values   = np.asarray(values)
    average  = np.average(values, weights=weights,axis=0)
    dist     = np.sqrt(np.sum((values-average)**2,axis=1))
    dist     = np.average(dist,weights=weights,axis=0)
    variance = np.average((values-average)**2, weights=weights, axis=0)  # Fast and numerically precise
    standard = np.sqrt(variance)
    axis1    = np.sqrt(np.sum(standard**2))
    return average, standard, dist

# Taken from https://github.com/joshalbrecht/shinyshell/blob/5f0c5f22a3425003a10a43d79d53b413915ea252/graph_points.py
def polyfit2d(x, y, z, order=3):
    ncols = (order + 1)**2
    G     = np.zeros((x.size, ncols))
    ij    = itertools.product(range(order+1), range(order+1))
    for k, (i,j) in enumerate(ij):
        G[:,k] = x**i * y**j
    m, residuals, _, singular = np.linalg.lstsq(G, z)
    return m

# Taken from https://github.com/msahamed/ArcPy_GIS_tutorial/blob/eb5357b6437ec88b3d43a6e56d42a3f1e211257d/Python_scripts/surfacePolynomial.py
def polyval2d(x, y, m):
    order = int(np.sqrt(len(m))) - 1
    ij = itertools.product(range(order+1), range(order+1))
    z = np.zeros_like(x)
    for a, (i,j) in zip(m, ij):
        z += a * x**i * y**j
    return z

# Generate equation using the points and create vertices.
def surfacereconstruct(points,savefn=None,degree=11,samples=[500,500],extrude=False,t=1.,a=0.,colors=None):
    # Color generation
    if type(colors) == type(None):
        colors = np.ones(points.shape)*255.
    # Generate equation
    points = np.asarray(points)
    equ    = polyfit2d(points[:,0,0], points[:,1,0], points[:,2,0], order=degree)
    equ_c0 = polyfit2d(points[:,0,0], points[:,1,0], colors[:,0,0], order=degree)
    equ_c1 = polyfit2d(points[:,0,0], points[:,1,0], colors[:,1,0], order=degree)
    equ_c2 = polyfit2d(points[:,0,0], points[:,1,0], colors[:,2,0], order=degree)
    # Create surface sample points.
    xx     = np.linspace(np.amin(points[:,0])-a,np.amax(points[:,0])+a,samples[0])
    yy     = np.linspace(np.amin(points[:,1])-a,np.amax(points[:,1])+a,samples[1])
    roi    = np.meshgrid(xx,yy)
    zz     = polyval2d(roi[0], roi[1], equ)
    zz_c0  = polyval2d(roi[0], roi[1], equ_c0)
    zz_c1  = polyval2d(roi[0], roi[1], equ_c1)
    zz_c2  = polyval2d(roi[0], roi[1], equ_c2)
    # Generate vertices.
    pnts   = []
    tris   = []
    for idx in range(0,samples[0]):
        for idy in range(0,samples[1]):
            pnt  = (roi[0][idx][idy]    , roi[1][idx][idy]    , zz[idx][idy])
            pnts.append(pnt)
    if extrude == True:
        for idx in range(0,samples[0]):
            for idy in range(0,samples[1]):
                pnt  = (roi[0][idx][idy]    , roi[1][idx][idy]    , zz[idx][idy]-t)
                pnts.append(pnt)
    m = samples[0]*samples[1]
    for idx in range(0,samples[0]-1):
        for idy in range(0,samples[1]-1):
            color = [zz_c0[idx][idy], zz_c1[idx][idy] , zz_c2[idx][idy]]
            tris.append(([idy+(idx+1)*samples[0], idy+idx*samples[0]  , idy+1+idx*samples[0]], color[0], color[1], color[2]))
            tris.append(([idy+(idx+1)*samples[0], idy+1+idx*samples[0], idy+1+(idx+1)*samples[0]], color[0], color[1], color[2]))
            if extrude == True:
               tris.append(([idy+(idx+1)*samples[0]+m, idy+idx*samples[0]+m  , idy+1+idx*samples[0]+m], color[0], color[1], color[2]))
               tris.append(([idy+(idx+1)*samples[0]+m, idy+1+idx*samples[0]+m, idy+1+(idx+1)*samples[0]+m], color[0], color[1], color[2]))
            if idx == 0 and extrude == True:
               tris.append(([idy+1+(idx)*samples[0], idy+idx*samples[0]  , idy+idx*samples[0]+m], color[0], color[1], color[2]))
               tris.append(([idy+1+idx*samples[0]+m, idy+1+(idx)*samples[0], idy+idx*samples[0]+m], color[0], color[1], color[2]))
            if idy == 0 and extrude == True:
               tris.append(([idy+(idx+1)*samples[0], idy+idx*samples[0]  , idy+idx*samples[0]+m], color[0], color[1], color[2]))
               tris.append(([idy+(idx+1)*samples[0]+m, idy+(idx+1)*samples[0], idy+idx*samples[0]+m], color[0], color[1], color[2]))
            if idx == samples[0]-2 and extrude == True:
               tris.append(([idy+1+(idx+1)*samples[0], idy+(idx+1)*samples[0]  , idy+(idx+1)*samples[0]+m], color[0], color[1], color[2]))
               tris.append(([idy+1+(idx+1)*samples[0]+m, idy+1+(idx+1)*samples[0], idy+(idx+1)*samples[0]+m], color[0], color[1], color[2]))
            if idy == samples[1]-2 and extrude == True:
               tris.append(([idy+1+(idx+1)*samples[0], idy+1+idx*samples[0]  , idy+1+idx*samples[0]+m], color[0], color[1], color[2]))
               tris.append(([idy+1+(idx+1)*samples[0]+m, idy+1+(idx+1)*samples[0], idy+1+idx*samples[0]+m], color[0], color[1], color[2]))
    tris   = np.asarray(tris, dtype=[('vertex_indices', 'i4', (3,)),('red', 'u1'), ('green', 'u1'),('blue', 'u1')])
    pnts   = np.asarray(pnts, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    # Save mesh.
    el1       = PlyElement.describe(pnts, 'vertex', comments=['Vertex data'])
    el2       = PlyElement.describe(tris, 'face', comments=['Face data'])
    if savefn != None:
       savefn = '%s.ply' % savefn
       PlyData([el1,el2],text="True").write(savefn)
    return pnts,tris

def drawaray(point0,point1,oldpnts=None,oldtris=None,k=0.02,color=[255,255,255]):
    point0    = np.reshape(point0,(3))
    point1    = np.reshape(point1,(3))
    if type(oldpnts) == type(None):
        a = 0
    elif type(oldpnts) != type(None):
        a = np.asarray(oldpnts).shape[0]
    pnts      = np.array([
                          (point0[0]   ,point0[1]   ,point0[2]),
                          (point0[0]+k ,point0[1]   ,point0[2]),
                          (point0[0]+k ,point0[1]+k ,point0[2]),
                          (point0[0]   ,point0[1]+k ,point0[2]),
                          (point1[0]   ,point1[1]   ,point1[2]),
                          (point1[0]+k ,point1[1]   ,point1[2]),
                          (point1[0]+k ,point1[1]+k ,point1[2]),
                          (point1[0]   ,point1[1]+k ,point1[2])],
                          dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')]
                        )
    tris      = np.array([
                          ([0+a,3+a,1+a], color[0], color[1], color[2]),
                          ([1+a,3+a,2+a], color[0], color[1], color[2]),
                          ([0+a,4+a,7+a], color[0], color[1], color[2]),
                          ([0+a,7+a,3+a], color[0], color[1], color[2]),
                          ([4+a,5+a,6+a], color[0], color[1], color[2]),
                          ([4+a,6+a,7+a], color[0], color[1], color[2]),
                          ([5+a,1+a,2+a], color[0], color[1], color[2]),
                          ([5+a,2+a,6+a], color[0], color[1], color[2]),
                          ([2+a,3+a,6+a], color[0], color[1], color[2]),
                          ([3+a,7+a,6+a], color[0], color[1], color[2]),
                          ([0+a,1+a,5+a], color[0], color[1], color[2]),
                          ([0+a,5+a,4+a], color[0], color[1], color[2]) ],
                          dtype=[('vertex_indices', 'i4', (3,)),
                                 ('red', 'u1'), ('green', 'u1'),
                                 ('blue', 'u1')]
                        )
    if type(oldpnts) != type(None):
        tris      = np.concatenate([np.copy(oldtris),np.copy(tris)])
        pnts      = np.concatenate([np.copy(oldpnts),np.copy(pnts)])
    return pnts,tris

# Definition to add to an existing PLY.
def mergewithaPLY(fn,pnts,tris):
    if os.path.isfile(fn) == False:
       return pnts,tris
    plydata = plyfile.PlyData.read(fn)
    newtris = []
    a       = pnts.shape[0]
    for ele in plydata.elements[1].data:
        newtris.append(((ele[0][0]+a,ele[0][1]+a,ele[0][2]+a), ele[1], ele[2], ele[3]))
    newtris = np.asarray(newtris,dtype=[('vertex_indices', 'i4', (3,)),
                                        ('red', 'u1'), ('green', 'u1'),
                                        ('blue', 'u1')])
    tris    = np.concatenate([np.copy(newtris).data,np.copy(tris)])
    pnts    = np.concatenate([plydata.elements[0].data,np.copy(pnts)])
    return pnts,tris

# Definition to save PLY.
def savePLY(savefn,pnts,tris):
    el1       = PlyElement.describe(pnts, 'vertex', comments=['Vertex data'])
    el2       = PlyElement.describe(tris, 'face', comments=['Face data'])
    savefn    = '%s.ply' % savefn
    PlyData([el1,el2],text="True").write(savefn)
    return True

# Definition to build a point cloud.
def build(points,savefn='./waveguide',r=1.):
    # Generate vertices.
    pnts = None
    tris = None
    for point in points:
        pnts,tris = drawaray(point+r,point-r,oldpnts=pnts,oldtris=tris,k=0.02,color=[255,255,255])
    savePLY(savefn,pnts,tris)
    return True

# Elem terefiş, kem gözlere şiş!
if __name__ == '__main__':
    pass

