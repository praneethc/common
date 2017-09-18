#!/usr/bin/python
# -*- coding: utf-8 -*-

from math import *
from kommon import *
import numpy as np
try:
    from OpenGL.GL import *
    from OpenGL.GLU import *
    from OpenGL.GLUT import *
except:
    prompt("OpenGL wrapper for python not found",title='OpenGL')
    sys.exit()

__author__  = ('Kaan Akşit')
__version__ = '0.1'


# Class for drawing using OpenGL.
class drawgl:
    # Constructor for the class
    def __init__(self,res=[600,600],loc=[0,0],title='OpenGL'):
        # Title of the window,
        self.title = title
        # Resolution and location.
        self.res   = np.asarray(res)
        self.loc   = np.asarray(loc)
        # Initialize the OpenGL pipeline,
        glutInit()
        # Set OpenGL display mode,
        glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH)
        # Set the Window size and position,
        glutInitWindowSize(self.res[0],self.res[1])
        glutInitWindowPosition(self.loc[0],self.loc[1])
        # Create the window with given title,
        glutCreateWindow(self.title)
        # Set background color,
        glClearColor(1.0, 1.0, 1.0, 0.0)
        # List of things to render,
        self.items     = []
        # Timer variable,
        self.last_time = 0
        # Direction of light,
        self.direction = [0.0, 2.0, -1.0, 1.0]
        # Intensity of light,
        self.intensity = [0.7, 0.7, 0.7, 1.0]
        # Intensity of ambient light,
        self.ambient_intensity = [0.3, 0.3, 0.3, 1.0]
        # The surface type(Flat or Smooth),
        self.surface = GL_FLAT
        # Viewport settings,
        self.camera_center   = np.array([0.,0.,0.])
        self.camera_rotation = np.array([0.,0.])
        self.camera_pos      = np.array([0.,0.1,20.])
        self.camera_shift    = False
        self.camera_rot      = False
        # Compute viewport location,
        self.compute_location()
        # Set OpenGL parameters,
        glEnable(GL_DEPTH_TEST)
        # Enable lighting,
        glEnable(GL_LIGHTING)
        # Set light model,
        glLightModelfv(GL_LIGHT_MODEL_AMBIENT, self.ambient_intensity)
        # Enable light number 0,
        glEnable(GL_LIGHT0)
        # Set position and intensity of light,
        glLightfv(GL_LIGHT0, GL_POSITION, self.direction)
        glLightfv(GL_LIGHT0, GL_DIFFUSE, self.intensity)
        # Setup the material,
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT, GL_AMBIENT_AND_DIFFUSE)
        # Set the callback function for display,
        glutDisplayFunc(self.display)
        # Set the callback function for the visibility,
        glutVisibilityFunc(self.visible)
        # Set the callback for keyboard,
        glutKeyboardFunc(self.keyboard)
        # Set the callback for mouse,
        glutMouseFunc(self.mouse)
        glutMotionFunc(self.mousemove)
        self.mouse_pos = np.array([0.,0.])
        # Load fragment shaders,
        self.path      =  os.path.dirname(os.path.realpath(__file__))
    # Definition for handling mouse events,
    def mousemove(self,x,y):
        rot0                     =  np.array([
                                              [1.,            0.,           0.],
                                              [0.,  np.cos(np.radians(self.camera_rotation[1])), np.sin(np.radians(self.camera_rotation[1]))],
                                              [0., -np.sin(np.radians(self.camera_rotation[1])), np.cos(np.radians(self.camera_rotation[1]))]
                                             ])
        rot1                     =  np.array([
                                              [np.cos(np.radians(self.camera_rotation[0])), 0., np.sin(np.radians(self.camera_rotation[0]))],
                                              [0.,            1.,           0.],
                                              [-np.sin(np.radians(self.camera_rotation[0])), 0., np.cos(np.radians(self.camera_rotation[0]))]
                                             ])
        diff                     = np.zeros((3))
        diff[0:2]                = (self.mouse_pos - np.array([x,y]))*10./self.res
        diff[1]                 *= -1
        d                        = np.sqrt(diff[0]**2+diff[1]**2+diff[2]**2)
        diff                    *= self.camera_shift*(not self.camera_rot)
        self.camera_center      -= diff
        self.camera_pos         -= diff
        if self.camera_rot*self.camera_shift == True:
            pdiff= np.array([
                             self.camera_pos[0]-self.camera_center[0],
                             self.camera_pos[1]-self.camera_center[1],
                             self.camera_pos[2]-self.camera_center[2]
                            ])
            self.camera_rotation    -= (np.array([x,y])-self.mouse_pos)/100.
            self.camera_pos          = self.camera_center+np.dot(rot0,np.dot(rot1,pdiff))
        self.mouse_pos           = np.array([x,y])
        self.compute_location()
    # Definition for handling mouse events,
    def mouse(self,button,state,x,y):
        if button == 0:
            self.camera_rot   = not self.camera_rot
            self.mouse_pos    = np.array([x,y])
        if button == 1:
            self.camera_shift = not self.camera_shift
            self.mouse_pos    = np.array([x,y])
        if button == 4:
            self.camera_mul = 0.99
            self.camera_pos = self.camera_center + self.camera_mul*(self.camera_pos-self.camera_center)
            self.compute_location()
        if button == 3:
            self.camera_mul = 1.01
            self.camera_pos = self.camera_center + self.camera_mul*(self.camera_pos-self.camera_center)
            self.compute_location()
    # Compute location,
    def compute_location(self):
        d = sqrt(
                 (self.camera_pos[0]-self.camera_center[0])**2+
                 (self.camera_pos[1]-self.camera_center[1])**2+
                 (self.camera_pos[2]-self.camera_center[2])**2
                )
        # Set matrix mode,
        glMatrixMode(GL_PROJECTION)
        # Reset matrix,
        glLoadIdentity()
        glFrustum(-d * 0.02, d * 0.02, -d * 0.02, d * 0.02, 1., 10*d)
        # Set camera,
        gluLookAt(
                  self.camera_pos[0],
                  self.camera_pos[1],
                  self.camera_pos[2],
                  self.camera_center[0],
                  self.camera_center[1],
                  self.camera_center[2],
                  0,
                  0,
                  1
                 )
    # Display definition,
    def display(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        # Set shade model,
        glShadeModel(self.surface)
        self.draw()
        glutSwapBuffers()
    # Definition to pass the uniforms,
    def uniforms(self):
        uniform0 = glGetUniformLocation(self.program0, "alpha")
        glUniform1f(uniform0,float(self.alpha))
    # Start displaying,
    def start(self):
        # Run the OpenGL main loop,
        glutMainLoop()
        return
    # Keyboard controller for the viewport,
    def keyboard(self, key, x, y):
        prompt('Pressed key: %s' % key,self.title)
        if key == b'q':
           sys.exit()
        # Toggle the surface,
        if key == GLUT_KEY_F1:
            if self.surface == GL_FLAT:
                self.surface = GL_SMOOTH
            else:
                self.surface = GL_FLAT
        self.compute_location()
        glutPostRedisplay()
    # The idle callback,
    def idle(self):
        time = glutGet(GLUT_ELAPSED_TIME)
        if self.last_time == 0 or time >= self.last_time + 40:
            self.last_time = time
            glutPostRedisplay()
    # The visibility callback,
    def visible(self, vis):
        if vis == GLUT_VISIBLE:
            glutIdleFunc(self.idle)
        else:
            glutIdleFunc(None)
    def draw(self):
        for item in self.items:
            if item[0] == 'sphere':
                self.sphere(
                            loc=[item[2],item[3],item[4]],
                            lats=item[5],
                            longs=item[6],
                            angs=[item[7],item[8]],
                            r=item[9],
                            color=item[10]
                           )
            elif item[0] == 'ray':
                self.ray(p0=item[2],p1=item[3],color=item[4])
    # Definition to add a ray to the rendering list,
    def addray(self,p0,p1,id=0,color=[1.,0.,0.]):
        self.items.append(['ray',id,p0,p1,color])
    # Draw a ray,
    def ray(self,p0=[0,0,0],p1=[10,10,10],color=[1.,0.,0.]):
        glBegin(GL_LINES)
        glColor3f(color[0],color[1],color[2])
        glVertex3f(p0[0], p0[1], p0[2])
        glVertex3f(p1[0], p1[1], p1[2])
        glEnd()
    # Definition to add a sphere to the rendering list,
    def addsphere(self,id=0,loc=[0,0,0],lats=10,longs=10,angs=[pi,pi],r=1.,color=[1.,0.,0.]):
        self.items.append(['sphere',id,loc[0],loc[1],loc[2],lats,longs,angs[0],angs[1],r,color])
    # Draw a sphere,
    def sphere(self,loc=[0,0,0],lats=10,longs=10,angs=[pi,pi],r=1.,color=[1.,0.,0.]):
        glPolygonMode( GL_FRONT_AND_BACK, GL_LINE )
        for i in range(0, lats + 1):
            lat0 = angs[0] * (-0.5 + float(float(i - 1) / float(lats)))
            z0 = sin(lat0)*r
            zr0 = cos(lat0)*r

            lat1 = angs[1] * (-0.5 + float(float(i) / float(lats)))
            z1 = sin(lat1)*r
            zr1 = cos(lat1)*r

            # Use Quad strips to draw the sphere,
            glBegin(GL_QUAD_STRIP)
            glColor4f(color[0],color[1],color[2],self.alpha)
            for j in range(0, longs + 1):
                lng = 2 * pi * float(float(j - 1) / float(longs))
                x = cos(lng)
                y = sin(lng)
                glNormal3f(x * zr0+loc[0], y * zr0+loc[1], z0+loc[2])
                glVertex3f(x * zr0+loc[0], y * zr0+loc[1], z0+loc[2])
                glNormal3f(x * zr1+loc[0], y * zr1+loc[1], z1+loc[2])
                glVertex3f(x * zr1+loc[0], y * zr1+loc[1], z1+loc[2])

            glEnd()

# Elem terefiş, kem gözlere şiş!
if __name__ == '__main__':
    pass

