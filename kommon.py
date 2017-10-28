#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys
import json
import termios, tty
import numpy as np
from datetime import datetime

__author__  = ('Kaan Akşit')
__version__ = '0.1'

# Definition to prompt.
def prompt(txt,title='COMMON'):
    print('[%s] [%s] %s' % (str(datetime.now()),title,txt))

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
def surfaceinteract(ray,vec,n,surface,id=0,color=[1.,1.,1.,1.],drawgl=None,debug=False):
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
       return vec0,norm0,surface
    elif type(vec0) == type(False):
       vec1 = ray.reflect(vec,norm0)
       return vec1,norm0,surface
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

# Elem terefiş, kem gözlere şiş!
if __name__ == '__main__':
    pass

