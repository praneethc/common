#!/usr/bin/python
# -*- coding: utf-8 -*-

from math import *
from kommon import *
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
        # Title of the window.
        self.title = title
        # Initialize the OpenGL pipeline.
        glutInit()
        # Set OpenGL display mode
        glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH)
        # Set the Window size and position
        glutInitWindowSize(res[0],res[1])
        glutInitWindowPosition(loc[0],loc[1])
        # Create the window with given title
        glutCreateWindow(self.title)
        # Set background color to black
        glEnable(GL_BLEND)
        glBlendEquation(GL_FUNC_ADD)
        glBlendFunc(GL_ONE,GL_ONE)
        glClearColor(0.0, 0.0, 0.0, 0.0)
        # List of things to render.
        self.items     = []
        # Value of the alpha (1 means opaque, 0 means fully transparent).
        self.alpha     = 0.2
        # Timer variable.
        self.last_time = 0
        # Direction of light
        self.direction = [0.0, 2.0, -1.0, 1.0]
        # Intensity of light
        self.intensity = [0.7, 0.7, 0.7, 1.0]
        # Intensity of ambient light
        self.ambient_intensity = [0.3, 0.3, 0.3, 1.0]
        # The surface type(Flat or Smooth)
        self.surface = GL_FLAT
        # Viewport settings.
        self.user_theta = 0
        self.user_height = 0
        # Compute viewport location.
        self.compute_location()
        # Set OpenGL parameters
        glEnable(GL_DEPTH_TEST)
        # Enable lighting
        glEnable(GL_LIGHTING)
        # Set light model
        glLightModelfv(GL_LIGHT_MODEL_AMBIENT, self.ambient_intensity)
        # Enable light number 0
        glEnable(GL_LIGHT0)
        # Set position and intensity of light
        glLightfv(GL_LIGHT0, GL_POSITION, self.direction)
        glLightfv(GL_LIGHT0, GL_DIFFUSE, self.intensity)
        # Setup the material
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT, GL_AMBIENT_AND_DIFFUSE)
        # Set the callback function for display
        glutDisplayFunc(self.display)
        # Set the callback function for the visibility
        glutVisibilityFunc(self.visible)
        # Set the callback for special function
        glutKeyboardFunc(self.keyboard)
        # Load fragment shaders.
        self.path      =  os.path.dirname(os.path.realpath(__file__))
#        self.program0  = self.LoadShader(shaderloc='%s/shaders/simple.frag' % self.path, both=False)
    # Compute location
    def compute_location(self):
        x = 2 * cos(self.user_theta)
        y = 2 * sin(self.user_theta)
        z = self.user_height
        d = sqrt(x * x + y * y + z * z)
        # Set matrix mode
        glMatrixMode(GL_PROJECTION)
        # Reset matrix
        glLoadIdentity()
        glFrustum(-d * 0.5, d * 0.5, -d * 0.5, d * 0.5, d - 1.1, d + 1.1)
        # Set camera
        gluLookAt(x, y, z, 0, 0, 0, 0, 0, 1)
    # Display definition.
    def display(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        # Load the shader.
#        glUseProgram(self.program0)
        # Pass the uniforms.
#        self.uniforms()
        # Set shade model
        glShadeModel(self.surface)
        self.draw()
        glutSwapBuffers()
    # Definition to pass the uniforms.
    def uniforms(self):
        uniform0 = glGetUniformLocation(self.program0, "alpha")
        glUniform1f(uniform0,float(self.alpha))
    # Start displaying.
    def start(self):
        # Run the OpenGL main loop
        glutMainLoop()
        return
    # Keyboard controller for the viewport.
    def keyboard(self, key, x, y):
        prompt('Pressed key: %s' % key,self.title)
        if key == b'q':
           sys.exit()
        # Scale the sphere up or down
        if key == b'w':
            self.user_height += 0.1
        if key == b's':
            self.user_height -= 0.1
        # Rotate the cube
        if key == b'a':
            self.user_theta += 0.1
        if key == b'd':
            self.user_theta -= 0.1
        # Toggle the surface
        if key == GLUT_KEY_F1:
            if self.surface == GL_FLAT:
                self.surface = GL_SMOOTH
            else:
                self.surface = GL_FLAT
        self.compute_location()
        glutPostRedisplay()
    # The idle callback
    def idle(self):
        time = glutGet(GLUT_ELAPSED_TIME)
        if self.last_time == 0 or time >= self.last_time + 40:
            self.last_time = time
            glutPostRedisplay()
    # The visibility callback
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
                            r=item[9]
                           )
    # Definition to add a sphere to rendering list.
    def addsphere(self,id=0,loc=[0,0,0],lats=100,longs=100,angs=[pi,pi],r=1.,color=[1.,0.,0.]):
        self.items.append(['sphere',id,loc[0],loc[1],loc[2],lats,longs,angs[0],angs[1],r,color])
    # Draw a sphere
    def sphere(self,loc=[0,0,0],lats=100,longs=100,angs=[pi,pi],r=1.,color=[1.,0.,0.]):
        for i in range(0, lats + 1):
            lat0 = angs[0] * (-0.5 + float(float(i - 1) / float(lats)))
            z0 = sin(lat0)*r
            zr0 = cos(lat0)*r

            lat1 = angs[1] * (-0.5 + float(float(i) / float(lats)))
            z1 = sin(lat1)*r
            zr1 = cos(lat1)*r

            # Use Quad strips to draw the sphere
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
    def LoadShader(self,shaderloc='simple.frag',ShadersType=GL_FRAGMENT_SHADER,both=False,color=None):
        prompt('Shader status: Loading...',title=self.title)
        # Load shader as a string.
        source  = open(shaderloc,'r').read()
        shader  = glCreateShader(ShadersType)
        glShaderSource(shader,source)
        glCompileShader(shader)
        if both == True:
             sourcev  = open(shaderloc.replace('frag','vertex'),'r').read()
             shaderv  = glCreateShader(GL_VERTEX_SHADER)
             glShaderSource(shaderv,sourcev)
             glCompileShader(shaderv)
        # Definition to load shader, GL_VERTEX_SHADER, GL_FRAGMENT_SHADER.

        program = glCreateProgram()
        glAttachShader(program, shader)
        if both == True:
             glAttachShader(program, shaderv)

        glLinkProgram(program)

        prompt('Shader status: %s' % str(glGetProgramInfoLog(program)),title=self.title)

        return program

# Elem terefiş, kem gözlere şiş!
if __name__ == '__main__':
    pass

