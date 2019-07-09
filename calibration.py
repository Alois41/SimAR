from time import clock, sleep
from OpenGL.GL import *
from OpenGL.GLUT import *
from source.Frame import Frame
from source.Frame import Camera
from OpenGL.GLU import *
from source.Global_tools import Config as Conf, Globals as Glob
from source.drawing import *
from source.Global_tools import Config as Conf, Globals as Glob
import sys
import cv2
import matplotlib.pyplot as plt
sys.path += ['.']
OpenGL.ERROR_ON_COPY = True

texture_array = None

camera = Camera(Conf.width, Conf.height)


def init():
    glClearColor(0.0, 0.0, 0.0, 1.0)
    glutDisplayFunc(display)
    # glutReshapeFunc(reshape)
    glutKeyboardFunc(keyboard)
    glutIdleFunc(idle)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)


def idle():
    camera.take_frame()

    image = cv2.flip(cv2.cvtColor(camera.image_raw, cv2.COLOR_BGR2RGBA), 0)
    image = cv2.addWeighted(image, 2, image, 0, -300)

    glBindTexture(GL_TEXTURE_2D, texture_array)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, image.shape[1], image.shape[0],
                 0, GL_RGBA, GL_UNSIGNED_BYTE, image)
    glutPostRedisplay()


def display():
    global texture_array
    # Set Projection Matrix
    if texture_array is None:
        texture_array = glGenTextures(1)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluOrtho2D(0, Conf.width, 0, Conf.height)
    # Switch to Model View Matrix
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    draw_rectangle(.5*Conf.width, 0, 100, 100, 1, 1, 1)

    draw_texture(texture_array, 100, 0, 200, 200)

    glutSwapBuffers()


def reshape(w, h):
    w = max(1E-4, w)
    h = max(1E-4, h)
    if h == 0:
        h = 1
    glViewport(0, 0, w, h)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    # allows for reshaping the window without distoring shape
    if w <= h:
        glOrtho(-Glob.nRange, Glob.nRange, -Glob.nRange * h / w, Glob.nRange * h / w, -Glob.nRange, Glob.nRange)
    else:
        glOrtho(-Glob.nRange * w / h, Glob.nRange * w / h, -Glob.nRange, Glob.nRange, -Glob.nRange, Glob.nRange)

    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()


def keyboard(key, x, y):
    if key == b'\x1b':
        sys.exit()


def draw_texture(tex_loc: int, x: int, y: int, l: int, h: int) -> void:
    glEnable(GL_TEXTURE_2D)
    glBindTexture(GL_TEXTURE_2D, tex_loc)
    draw_textured_rectangle(x, y, l, h)
    glDisable(GL_TEXTURE_2D)


glutInit(sys.argv)
glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH)
glutInitWindowSize(Conf.width, Conf.height)
glutInitWindowPosition(1366 + 10, 0)  # main window dim + 1
glutCreateWindow("OpenGL + OpenCV")
glutFullScreen()
init()
glutMainLoop()

