#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys
import numpy as np
from datetime import datetime

__author__  = ('Kaan Akşit')
__version__ = '0.1'

# Definition to prompt.
def prompt(txt,title='COMMON'):
    print('[%s] [%s] %s' % (str(datetime.now()),title,txt))

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
    roty = rotmatx(angles[1])
    rotz = rotmatx(angles[2])
    return np.dot(rotz,np.dot(roty,np.dot(rotx,point))),rotx,roty,rotz

# Elem terefiş, kem gözlere şiş!
if __name__ == '__main__':
    pass

