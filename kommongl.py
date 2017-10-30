#!/usr/bin/python
# -*- coding: utf-8 -*-

from math import *
from kommon import *
import numpy as np
import pyrr
from pyrr import Matrix44, Vector4, Vector3, Quaternion
from scipy import linalg
import ast
try:
    from OpenGL.GL import *
    from OpenGL.GLU import *
    from OpenGL.GLUT import *
    import openvr
    import imgui
    import imgui.integrations.opengl
    from imgui.integrations.opengl import ProgrammablePipelineRenderer,FixedPipelineRenderer
except:
    prompt("OpenGL wrapper, Imgui or OpenVR for python not found",title='OpenGL')
    sys.exit()

__author__  = ('Kaan Akşit')
__version__ = '0.1'


def matrixForOpenVrMatrix(mat):
    if len(mat.m) == 4: # HmdMatrix44_t?
        result = np.matrix(
                           ((mat.m[0][0], mat.m[1][0], mat.m[2][0], mat.m[3][0]),
                            (mat.m[0][1], mat.m[1][1], mat.m[2][1], mat.m[3][1]),
                            (mat.m[0][2], mat.m[1][2], mat.m[2][2], mat.m[3][2]),
                            (mat.m[0][3], mat.m[1][3], mat.m[2][3], mat.m[3][3]),), np.float32)
        return result
    elif len(mat.m) == 3: # HmdMatrix34_t?
        result = np.matrix(
                           ((mat.m[0][0], mat.m[1][0], mat.m[2][0], 0.0),
                            (mat.m[0][1], mat.m[1][1], mat.m[2][1], 0.0),
                            (mat.m[0][2], mat.m[1][2], mat.m[2][2], 0.0),
                            (mat.m[0][3], mat.m[1][3], mat.m[2][3], 1.0),), np.float32)
        return result

# Class for drawing using OpenGL.
class drawgl:
    # Constructor for the class
    def __init__(self,res=[1000,1000],loc=[0,0],title='OpenGL',lightflag=True,transparencyflag=True,vrflag=False):
        # Title of the window,
        self.title = title
        # Resolution and location,
        self.res   = np.asarray(res)
        self.loc   = np.asarray(loc)
        # Initialize the OpenGL pipeline,
        glutInit()
        # Selected id for drawing.
        self.selectedid  = False
        self.maxid       = 0
        # Set the light flag,
        self.lightflag   = lightflag
        # Set the polygon mode: GL_FILL or GL_LINE,
        self.polygonmode = GL_FILL
        # Set OpenGL display mode,
        glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH)
        # Set the Window size and position,
        self.vrflag      = vrflag
        if self.vrflag == True:
            self.vr_system = openvr.init(openvr.VRApplication_Scene)
            self.vr_width, self.vr_height = self.vr_system.getRecommendedRenderTargetSize()
            self.compositor = openvr.VRCompositor()
            if self.compositor is None:
                self.vrflag = False
                raise Exception("Unable to create compositor")
            self.res[0]                   = self.vr_width
            self.res[1]                   = self.vr_height
        glutInitWindowSize(self.res[0],self.res[1])
        glutInitWindowPosition(self.loc[0],self.loc[1])
        # Create the window with given title,
        glutCreateWindow(self.title)
        # Set background color,
        glClearColor(0.0, 0.0, 0.0, 0.0)
        # Transparency settings,
        self.transparencyflag = transparencyflag
        if self.transparencyflag == True:
            glEnable(GL_BLEND)
            glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA)
        # List of things to render,
        self.items            = []
        self.items_rays       = []
        self.items_rays_c     = []
        self.items_rays_i     = []
        self.vertex_rays      = []
        # Timer variable,
        self.last_time        = 0
        # The surface type(Flat or Smooth),
        self.surface          = GL_SMOOTH
        # Viewport settings,
        self.camera_center    = np.array([0.,0.,0.])
        self.camera_rotation  = np.array([180.,90.])
        self.camera_pos       = np.array([0.,0.1,10.])
        self.camera_shift     = False
        self.camera_rot       = False
        self.pmax             = np.zeros((3))
        self.pmin             = np.zeros((3))
        # Compute viewport location,
        self.compute_location()
        # Set OpenGL parameters,
        glEnable(GL_DEPTH_TEST)
        # Light settings,
        if self.lightflag == True:
            # Intensity of light,
            self.intensity         = [0.7, 0.7, 0.7, 1.0]
            # Intensity of ambient light,
            self.ambient_intensity = [0.3, 0.3, 0.3, 1.0]
            # Enable lighting,
            glEnable(GL_LIGHTING)
            # Set light model,
            glLightModelfv(GL_LIGHT_MODEL_AMBIENT, self.ambient_intensity)
            # Enable light number 0,
            glEnable(GL_LIGHT0)
            # Set position and intensity of light,
            glLightfv(GL_LIGHT0, GL_POSITION, [0.,0.,100.])
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
        self.mouse_pos  = np.array([0.,0.])
        # Load fragment shaders,
        self.path           =  os.path.dirname(os.path.realpath(__file__))
#        fn                  = '%s/shaders/simple.frag' % os.path.dirname(os.path.realpath(__file__))
        if self.vrflag == True:
            fn = '%s/shaders/openvr.frag' % os.path.dirname(os.path.realpath(__file__))
        else:
            fn = '%s/shaders/blank.frag' % os.path.dirname(os.path.realpath(__file__))
        self.program        = self.LoadShader(shaderloc=fn,both=True)
        glUseProgram(self.program)
        self.mvp            = glGetUniformLocation(self.program, 'MVP')
        self.projmat        = np.array([
                                        [1.,0.,0.,0.],
                                        [0.,1.,0.,0.],
                                        [0.,0.,1.,0.],
                                        [0.,0.,0.,1.]
                                       ],np.float32)
        glUniformMatrix4fv(self.mvp, 1, GL_FALSE, self.projmat)
        # Imgui settings,
        renderer            = FixedPipelineRenderer()
        io                  = imgui.get_io()
        io.display_size     = self.res[0],self.res[1]
        io.display_fb_scale = 1.,1.
        io.delta_time       = 1.0/60
        # Generating projection matrix,
        self.zNear          = 0.01
        self.zFar           = 5000.0
        self.projcyclope    = self.projectionmatrix(60.,self.zNear, self.zFar)
        # OpenVR for near eye display support,
        if self.vrflag == True:
            poses_t = openvr.TrackedDevicePose_t * openvr.k_unMaxTrackedDeviceCount
            self.poses = poses_t()
            # Set up framebuffer and render textures
            self.fbl = glGenFramebuffers(1)
            glBindFramebuffer(GL_FRAMEBUFFER, self.fbl)
            self.depth_bufferl = glGenRenderbuffers(1)
            glBindRenderbuffer(GL_RENDERBUFFER, self.depth_bufferl)
            glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, self.vr_width, self.vr_height)
            glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, self.depth_bufferl)
            self.texture_id_l = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, self.texture_id_l)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, self.vr_width, self.vr_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, None)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 0)
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, self.texture_id_l, 0)
            status = glCheckFramebufferStatus(GL_FRAMEBUFFER)
            if status != GL_FRAMEBUFFER_COMPLETE:
                glBindFramebuffer(GL_FRAMEBUFFER, 0)
                raise Exception("Incomplete framebuffer")
            glBindFramebuffer(GL_FRAMEBUFFER, 0)
            self.fbr = glGenFramebuffers(1)
            glBindFramebuffer(GL_FRAMEBUFFER, self.fbr)
            self.depth_bufferr = glGenRenderbuffers(1)
            glBindRenderbuffer(GL_RENDERBUFFER, self.depth_bufferr)
            glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, self.vr_width, self.vr_height)
            glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, self.depth_bufferr)
            self.texture_id_r = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, self.texture_id_r)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, self.vr_width, self.vr_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, None)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 0)
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, self.texture_id_r, 0)
            status = glCheckFramebufferStatus(GL_FRAMEBUFFER)
            glBindFramebuffer(GL_FRAMEBUFFER, 0)
            if status != GL_FRAMEBUFFER_COMPLETE:
                glBindFramebuffer(GL_FRAMEBUFFER, 0)
                raise Exception("Incomplete framebuffer")
            glBindFramebuffer(GL_FRAMEBUFFER, 0)
            # OpenVR texture data,
            self.texturel             = openvr.Texture_t()
            self.texturel.handle      = self.texture_id_l
            self.texturel.eType       = openvr.TextureType_OpenGL
            self.texturel.eColorSpace = openvr.ColorSpace_Gamma
            self.texturer             = openvr.Texture_t()
            self.texturer.handle      = self.texture_id_r
            self.texturer.eType       = openvr.TextureType_OpenGL
            self.texturer.eColorSpace = openvr.ColorSpace_Gamma
            self.texture              = [self.texturel,self.texturer]
            self.eyes                 = [openvr.Eye_Left,openvr.Eye_Right]
            # Compute projection matrix
            self.view        = []
            self.projection  = []
            self.projection.append(np.asarray(matrixForOpenVrMatrix(self.vr_system.getProjectionMatrix(openvr.Eye_Left, self.zNear, self.zFar))))
            self.projection.append(np.asarray(matrixForOpenVrMatrix(self.vr_system.getProjectionMatrix(openvr.Eye_Right, self.zNear, self.zFar))))
            self.view.append(matrixForOpenVrMatrix(self.vr_system.getEyeToHeadTransform(openvr.Eye_Left)).I)
            self.view.append(matrixForOpenVrMatrix(self.vr_system.getEyeToHeadTransform(openvr.Eye_Right)).I)
    # Definition to exit class,
    def __exit__(self, type_arg, value, traceback):
        if self.vr_system is not None:
            openvr.shutdown
            self.vr_system = None
        if self.vrflag == True:
            glDeleteTextures([self.texture_id])
            glDeleteRenderbuffers([self.depth_buffer])
            glDeleteFramebuffers([self.fb])
            self.fb = 0
    # Definition to draw a gradient background,
    def gradient(self):
        glDisable(GL_DEPTH_TEST)
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glDisable(GL_LIGHTING)

        glBegin(GL_QUADS);
        pos          = np.array([1.,1.])
        glColor3f(0.,0.,1.)
        glVertex2f(-pos[0],-pos[1])
        glVertex2f( pos[0],-pos[1])

        glColor3f(1.,1.,1.)
        glVertex2f( pos[0], pos[1])
        glVertex2f(-pos[0], pos[1])
        glEnd()

        glEnable(GL_DEPTH_TEST)
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)

    # Definition for handling menu of imgui,
    def menu(self):
        imgui.new_frame()
        imgui.begin(self.title, True)
        imgui.text("Ray tracer")
        imgui.end()
        imgui.render()
    # Definition for handling mouse events,
    def mousemove(self,x,y):
        diff                     = np.zeros((3))
        diff[0:2]                = (self.mouse_pos - np.array([x,y]))*-10./self.res
        diff,_,_,_               = rotatepoint(diff,[
                                                     self.camera_rotation[1],
                                                     self.camera_rotation[0],
                                                     0
                                                    ])
        diff                    *= self.camera_shift*(not self.camera_rot)
        self.camera_center      -= diff
        self.camera_pos         -= diff
        if self.camera_rot*self.camera_shift == True:
            pdiff= np.array([
                             self.camera_pos[0]-self.camera_center[0],
                             self.camera_pos[1]-self.camera_center[1],
                             self.camera_pos[2]-self.camera_center[2]
                            ])
            rot_diff                 = (self.mouse_pos-np.array([x,y]))/10.
            self.camera_rotation    += np.array([-rot_diff[0],rot_diff[1]])
            pdiff                    = np.array([0.,0.,np.sqrt(np.sum(pdiff**2))])
            self.camera_pos,_,_,_    = rotatepoint(pdiff,[self.camera_rotation[0],
                                                          self.camera_rotation[1],
                                                          0
                                                         ])
            self.camera_pos         += self.camera_center
        self.mouse_pos = np.array([x,y])
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
        self.rotateviewport()
        self.d = sqrt(
                      (self.camera_pos[0]-self.camera_center[0])**2+
                      (self.camera_pos[1]-self.camera_center[1])**2+
                      (self.camera_pos[2]-self.camera_center[2])**2
                     )
        # Set matrix mode,
        glMatrixMode(GL_PROJECTION)
        # Reset matrix,
        glLoadIdentity()
        glFrustum(-self.d * 0.02, self.d * 0.02, -self.d * 0.02, self.d * 0.01, 1., 1000*self.d)
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
        # Set light.
        if self.lightflag == True:
            glLightfv(GL_LIGHT0, GL_POSITION, self.camera_pos)
    # Definition to generate projection matrix,
    def projectionmatrix(self,fovy,near,far):
        mvp      = pyrr.matrix44.create_perspective_projection(fovy, float(self.res[1])/float(self.res[0]), near, far)
        return pyrr.matrix44.inverse(mvp)
    # Display definition,
    def display(self):
        #  For VR support,
        if self.vrflag == True:
            glBindFramebuffer(GL_FRAMEBUFFER, 0)
            self.compositor.waitGetPoses(self.poses, openvr.k_unMaxTrackedDeviceCount, None, 0)
            hmd_pose0 = self.poses[openvr.k_unTrackedDeviceIndex_Hmd]
            if hmd_pose0.bPoseIsValid:
                mat             = hmd_pose0.mDeviceToAbsoluteTracking
                self.pose       = np.asarray(matrixForOpenVrMatrix(mat).I)
            for id,framebuffer in enumerate([self.fbl,self.fbr]):
                glUseProgram(self.program)
                self.MVP        = np.dot(self.pose,np.dot(self.view[id],self.projection[id]))
                glUniformMatrix4fv(self.mvp, 1, GL_FALSE, self.MVP)
                glBindFramebuffer(GL_FRAMEBUFFER, framebuffer)
                glClearColor(0.0, 0.0, 0.5, 0)
                glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
                self.gradient()
                self.gradient()
                self.draw()
                self.compositor.submit(self.eyes[id], self.texture[id])
            glUseProgram(0)
        else:
            glBindFramebuffer(GL_FRAMEBUFFER, 0)
            glUseProgram(0)
            glViewport(0,0, self.res[0], self.res[1])
            # Clearing the depth and color,
            glClearColor(0., 0., 0., 0.)
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            # Set shade model,
            glShadeModel(self.surface)
            self.gradient()
            # Imgui,
            self.menu()
            # Wireframe or fill?
            glPolygonMode(GL_FRONT_AND_BACK, self.polygonmode)
            self.draw()
            # Swap buffers,
            glutSwapBuffers()
    # Start displaying,
    def start(self):
        # Run the OpenGL main loop,
        self.camera_center  = (self.pmax+self.pmin)/2.
        if np.count_nonzero(self.pmax) == 0 :
            self.pmax = np.array([10.,10.,10.])
        self.camera_pos    += self.camera_center+np.zeros([3])
        self.camera_pos[2]  = (np.amax(self.pmax[0:2])+np.abs(np.amin(self.pmin[0:2])))*2.
        self.compute_location()
        self.generatealldata(self.program)
        glutMainLoop()
        return
    # Definition to rotate the viewport.
    def rotateviewport(self):
        pdiff= np.array([
                         self.camera_pos[0]-self.camera_center[0],
                         self.camera_pos[1]-self.camera_center[1],
                         self.camera_pos[2]-self.camera_center[2]
                        ])
        pdiff                 = np.array([0.,0.,np.sqrt(np.sum(pdiff**2))])
        self.camera_pos,_,_,_ = rotatepoint(pdiff,[self.camera_rotation[0],
                                                   self.camera_rotation[1],
                                                   0.
                                                  ])
        self.camera_pos      += self.camera_center

    # Keyboard controller for the viewport,
    def keyboard(self, key, x, y):
        prompt('Pressed key: %s' % key,self.title)
        if key == b'r':
           self.selectedid += 1
           self.selectedid %= self.maxid+1
        if key == b'e':
           self.selectedid = False
        if key == b'q':
           sys.exit()
        if key == b'a':
           self.camera_rotation[0] += 1
        if key == b'z':
           self.camera_rotation[0] -= 1
        if key == b's':
           self.camera_rotation[1] += 1
        if key == b'x':
           self.camera_rotation[1] -= 1
        if key == b't':
            self.camera_mul = 0.99
            self.camera_pos = self.camera_center + self.camera_mul*(self.camera_pos-self.camera_center)
        if key == b'y':
            self.camera_mul = 1.01
            self.camera_pos = self.camera_center + self.camera_mul*(self.camera_pos-self.camera_center)
        self.compute_location()
        glutPostRedisplay()
    # The idle callback,
    def idle(self):
        time = glutGet(GLUT_ELAPSED_TIME)
        if (self.last_time == 0) or( time >= self.last_time + 40):
            self.last_time = time
            glutPostRedisplay()
    # The visibility callback,
    def visible(self, vis):
        if vis == GLUT_VISIBLE:
            glutIdleFunc(self.idle)
        else:
            glutIdleFunc(None)
    # Definition to draw a scene.
    def draw(self):
        for item in self.items:
            if type(self.selectedid) == type(False):
                self.drawprimitaves(item)
            if item[1] == self.selectedid and type(self.selectedid) != type(False):
                self.drawprimitaves(item)
        for id in range(0,self.maxid+1):
            if len(self.vertex_rays) != 0:
                if type(self.selectedid) == type(False):
                    self.display_rays(self.vertex_rays[id])
                if self.vertex_rays[id][4] == self.selectedid and type(self.selectedid) != type(False):
                    self.display_rays(self.vertex_rays[id])
    # Definition for drawing primitive interpretation.
    def drawprimitaves(self,item):
            if item[0] == 'sphere':
                self.sphere(
                            loc=item[2],
                            lats=item[3],
                            longs=item[4],
                            angs=item[5],
                            r=item[6],
                            color=item[7]
                           )
            elif item[0] == 'box':
                self.rectangularbox(loc=item[2],angles=item[3],size=item[4],color=item[5])
            elif item[0] == 'plane':
                self.plane(loc=item[2],size=item[3],angles=item[4],color=item[5])
    # Definition to shift everything from CPU to GPU,
    def generatealldata(self,shader):
        if len(self.items_rays) == 0:
           return True
        for id in range(0,self.maxid+1):
            # Rays are being pushed to GPU,
            vertices    = self.items_rays[id]
            colors      = self.items_rays_c[id]
            indices     = self.items_rays_i[id]
            vertex_data = self.create_object(id,shader,vertices,indices,colors,objtype=0)
            self.vertex_rays.append(vertex_data)
        return True
    # Definition to generate vertices, objtype=0 is a ray,...
    def create_object(self,id,shader,vertices,indices,colors,objtype=0):
        buffers = glGenBuffers(3)
        glBindBuffer(GL_ARRAY_BUFFER, buffers[0])
        glBufferData(GL_ARRAY_BUFFER,
                     len(vertices)*4,
                     (ctypes.c_float*len(vertices))(*vertices),
                     GL_STATIC_DRAW)
        glBindBuffer(GL_ARRAY_BUFFER, buffers[1])
        glBufferData(GL_ARRAY_BUFFER,
                     len(colors)*4,
                     (ctypes.c_float*len(colors))(*colors),
                     GL_STATIC_DRAW)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, buffers[2])
        glBufferData(GL_ELEMENT_ARRAY_BUFFER,
                     len(indices)*4,
                     (ctypes.c_uint*len(indices))(*indices),
                     GL_STATIC_DRAW)
        buffers = np.append(buffers,len(vertices)/2).astype(int)
        buffers = np.append(buffers,id).astype(int)
        buffers = np.append(buffers,objtype).astype(int)
        return buffers
    # Definition to add a ray to the rendering list,
    def addray(self,p0,p1,id=0,color=[1.,0.,0.,0.5],adddots=True):
        if id > self.maxid:
            self.maxid = id
        if id+1 > len(self.items_rays):
            self.items_rays.append([])
            self.items_rays_c.append([])
            self.items_rays_i.append([])
        self.items_rays[id].append(p0.T.tolist()[0][0])
        self.items_rays[id].append(p0.T.tolist()[0][1])
        self.items_rays[id].append(p0.T.tolist()[0][2])
        self.items_rays[id].append(p1.T.tolist()[0][0])
        self.items_rays[id].append(p1.T.tolist()[0][1])
        self.items_rays[id].append(p1.T.tolist()[0][2])
        for i in range(0,2):
            self.items_rays_c[id].append(color[0])
            self.items_rays_c[id].append(color[1])
            self.items_rays_c[id].append(color[2])
        self.items_rays_i[id].append(len(self.items_rays_i))
        self.items_rays_i[id].append(len(self.items_rays_i)+1)
        if adddots == True:
            self.addsphere(id=id,loc=p1,lats=1,longs=1,r=0.1,color=color)
        self.maxmin(p0)
        self.maxmin(p1)
    # Definition to display rays.
    def display_rays(self,buffers):
        glUseProgram(self.program)
        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_COLOR_ARRAY)
        glBindBuffer(GL_ARRAY_BUFFER, buffers[0])
        glVertexPointer(3, GL_FLOAT, 0, None);
        glBindBuffer(GL_ARRAY_BUFFER, buffers[1]);
        glColorPointer(3, GL_FLOAT, 0, None);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, buffers[2]);
        glDrawArrays(GL_LINES, 0, 2*buffers[3])
        glDisableClientState(GL_COLOR_ARRAY)
        glDisableClientState(GL_VERTEX_ARRAY);
        glUseProgram(0)
    # Definition to update the maximum and minimum.
    def maxmin(self,p):
        for id in range(0,3):
            if p[id] > self.pmax[id]:
               self.pmax[id] = p[id]
            if p[id] < self.pmin[id]:
               self.pmin[id] = p[id]
    # Draw a ray,
    def ray(self,p0=[0,0,0],p1=[10,10,10],color=[1.,0.,0.,0.5]):
        glBegin(GL_LINES)
        glColor4f(color[0],color[1],color[2],color[3])
        glVertex3f(p0[0], p0[1], p0[2])
        glVertex3f(p1[0], p1[1], p1[2])
        glEnd()
    # Definition to add a sphere to the rendering list,
    def addsphere(self,id=0,loc=[0,0,0],lats=10,longs=10,angs=[pi,pi],r=1.,color=[1.,0.,0.,0.5]):
        self.items.append(['sphere',id,loc,lats,longs,angs,r,color])
        self.maxmin(loc)
        if id > self.maxid:
            self.maxid = id
    # Draw a sphere,
    def sphere(self,loc=[0,0,0],lats=10,longs=10,angs=[pi,pi],r=1.,color=[1.,0.,0.,0.5]):
        for i in range(0, lats + 1):
            lat0 = angs[0] * (-0.5 + float(float(i - 1) / float(lats)))
            z0 = sin(lat0)*r
            zr0 = cos(lat0)*r

            lat1 = angs[1] * (-0.5 + float(float(i) / float(lats)))
            z1 = sin(lat1)*r
            zr1 = cos(lat1)*r

            # Use Quad strips to draw the sphere,
            glBegin(GL_QUAD_STRIP)
            glColor4f(color[0],color[1],color[2],color[3])
            for j in range(0, longs + 1):
                lng = 2 * pi * float(float(j - 1) / float(longs))
                x = cos(lng)
                y = sin(lng)
                glNormal3f(x * zr0+loc[0], y * zr0+loc[1], z0+loc[2])
                glVertex3f(x * zr0+loc[0], y * zr0+loc[1], z0+loc[2])
                glNormal3f(x * zr1+loc[0], y * zr1+loc[1], z1+loc[2])
                glVertex3f(x * zr1+loc[0], y * zr1+loc[1], z1+loc[2])
            glEnd()
    # Definition to add a plane to the rendering list,
    def addplane(self,loc,size,angles,id=0,color=[0.,0.1,0.,0.5]):
        self.items.append(['plane',id,loc,size,angles,color])
        loc = np.asarray(loc)
        self.maxmin(loc+size[0]/2.)
        self.maxmin(loc+size[1]/2.)
        self.maxmin(loc-size[0]/2.)
        self.maxmin(loc-size[1]/2.)
        if id > self.maxid:
            self.maxid = id
    # Draw a plane.
    def plane(self,loc=[0,0,0],size=[10.,10.],angles=[0.,0.,0.],color=[0.,0.1,0.,0.5]):
        glBegin(GL_QUADS)
        glColor4f(color[0],color[1],color[2],color[3])
        pos  = np.array([
                         [ size[0]/2.,  size[1]/2., 0.],
                         [-size[0]/2.,  size[1]/2., 0.],
                         [-size[0]/2., -size[1]/2., 0.],
                         [ size[0]/2., -size[1]/2., 0.]
                        ])
        for id in range(0,4):
            item        = pos[id]
            item,_,_,_  = rotatepoint(item,angles=angles)
            item       += loc
            glVertex3f(item[0],item[1],item[2])
        glEnd()
    # Definition to add a rectangular box to the rendering list,
    def addbox(self,loc,angles,size,id=0,color=[0.,1.,0.,0.5]):
        self.items.append(['box',id,loc,angles,size,color])
        loc = np.asarray(loc)
        self.maxmin(loc+size[0]/2.)
        self.maxmin(loc+size[1]/2.)
        self.maxmin(loc+size[2]/2.)
        self.maxmin(loc-size[0]/2.)
        self.maxmin(loc-size[1]/2.)
        self.maxmin(loc-size[2]/2.)
        if id > self.maxid:
            self.maxid = id
    # Draw a box.
    def rectangularbox(self,loc=[0,0,0],angles=[0,0,0],size=[10.,20.,30.],color=[0.,1.,0.,0.5]):
        width  = size[0]
        height = size[1]
        length = size[2]

        glBegin(GL_QUADS)
        glColor4f(color[0],color[1],color[2],color[3])

        pos = np.array([
                        [ width/2.,  height/2., -length/2.],
                        [-width/2.,  height/2., -length/2.],
                        [-width/2.,  height/2.,  length/2.],
                        [ width/2.,  height/2.,  length/2.],

                        [ width/2., -height/2.,  length/2.],
                        [-width/2., -height/2.,  length/2.],
                        [-width/2., -height/2., -length/2.],
                        [ width/2., -height/2., -length/2.],

                        [ width/2.,  height/2.,  length/2.],
                        [-width/2.,  height/2.,  length/2.],
                        [-width/2., -height/2.,  length/2.],
                        [ width/2., -height/2.,  length/2.],

                        [ width/2., -height/2., -length/2.],
                        [-width/2., -height/2., -length/2.],
                        [-width/2.,  height/2., -length/2.],
                        [ width/2.,  height/2., -length/2.],

                        [-width/2.,  height/2.,  length/2.],
                        [-width/2.,  height/2., -length/2.],
                        [-width/2., -height/2., -length/2.],
                        [-width/2., -height/2.,  length/2.],

                        [ width/2.,  height/2., -length/2.],
                        [ width/2.,  height/2.,  length/2.],
                        [ width/2., -height/2.,  length/2.],
                        [ width/2., -height/2., -length/2.]
                       ])

        for id in range(0,24):
            item        = pos[id]
            item,_,_,_  = rotatepoint(item,angles=angles)
            item       += loc
            glVertex3f(item[0],item[1],item[2])

        glEnd()
    def LoadShader(self,shaderloc='./shaders/simple.frag',ShadersType=GL_FRAGMENT_SHADER,both=False,color=None):
        prompt('Shader status: Loading...')
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
        prompt('Shader status: %s' % str(glGetProgramInfoLog(program)))
        return program

# Elem terefiş, kem gözlere şiş!
if __name__ == '__main__':
    pass

