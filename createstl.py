#!/usr/bin/python
# -*- coding: utf-8 -*-

# Usual imports.
import sys,math
# Trick to use FreeCAD without any gui.
sys.path.insert(0,'/usr/lib/freecad/lib/')
# Always import FreeCAD before Part module!
import numpy as np
import FreeCAD
import Part, Drawing
import Mesh
from FreeCAD import Base
from datetime import datetime

__author__ = ('Kaan Akşit')

# Definition to show the part.
def show(part):
    Part.makeSolid(part)
    Part.show(part)
    return True

# Definition to save the part.
def save(part,filename='./'):
#    part.exportIges('%s.iges' % filename)
    part.exportStl('%s.stl' % filename)
    return True

# Definition to draw cylinder volume, all dimensions are in mm.
def DrawCylVol(XOffSet,YOffSet,ZOffSet,R,ThiZ,direction=[0,0,1],alpha=360):
    dir  = Base.Vector(direction[0],direction[1],direction[2])
    body = Part.makeCylinder(R,ThiZ,Base.Vector(XOffSet,YOffSet,ZOffSet),dir,alpha)
    return body

# Definition to fuse all the parts given as a list.
def FusePieces(pieces):
    OnePiece = pieces[0]
    for i in range(0,len(pieces)):
        OnePiece = OnePiece.fuse(pieces[i])
    return OnePiece

# Definition for dummy build.
def build(points,savefn='./waveguide'):
    pieceslst = []
    for i in range(0,points.shape[1]):
        item = points[:,i]
        p0   = Part.makeSphere(0.1,Base.Vector(item[0],item[1],item[2]),Base.Vector(0,0,1),-90,90,360)
        pieceslst.append(p0)
    # Fusing all the pieces.
    onepiece = FusePieces(pieceslst)
    save(onepiece,filename=savefn)
    return True

# Main definition to build multiple parts.
def main():
    pass

if __name__ == '__main__':
    main()
