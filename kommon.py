#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys
import json
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

# Sample a circular aperture.
def samplecircular(no,aperture,loc=[0.,0.,0.],angles=[0.,0.,0.]):
    points = []
    for idx in range(0,no[0]):
        for idy in range(0,no[1]):
            point        = np.array([
                                     idy*aperture[0]/2/no[1]*np.cos(2*np.pi*idx/no[0]),
                                     idy*aperture[1]/2/no[1]*np.sin(2*np.pi*idx/no[0]),
                                     0.
                                    ])
            point       += loc
            point,_,_,_  = rotatepoint(point,angles=angles)
            points.append(point)
    return points

# Elem terefiş, kem gözlere şiş!
if __name__ == '__main__':
    pass

