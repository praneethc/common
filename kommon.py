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
def generateplane(point):
    plane = np.array([
                      [point[0]+10 ,point[1]+10 ,point[2]],
                      [point[0]    ,point[1]+10 ,point[2]],
                      [point[0]    ,point[1]    ,point[2]]
                     ])
    return plane

# Elem terefiş, kem gözlere şiş!
if __name__ == '__main__':
    pass

