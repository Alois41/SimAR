from time import clock, sleep
from OpenGL.GL import *
from OpenGL.GLUT import *
from source.Frame import Frame
from OpenGL.GLU import *
from source.Global_tools import Config as Conf, Globals as Glob
from source.brick import BrickArray
import sys
sys.path += ['.']
OpenGL.ERROR_ON_COPY = True


def init():
    glClearColor(0.0, 0.0, 0.0, 1.0)
    glutDisplayFunc(display)
    glutReshapeFunc(reshape)
    glutKeyboardFunc(keyboard)
    glutIdleFunc(idle)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    Glob.f = Frame(Conf.width, Conf.height)
    # Glob.f.run()


def idle():
    try:
        # capture next frame
        Glob.delta_t = clock() - Glob.t_ref
        Glob.t_ref = clock()
        sleep(max(0, 1/30 - Glob.delta_t))
        print(1/30 - Glob.delta_t)

        # update frame
        Glob.f.cam.take_frame()
        Glob.f.update_bricks()

        Conf.hand_text = Glob.f.detect_hand()
        glutPostRedisplay()

    except Exception as e:
        if "empty" in str(e):
            raise IOError("Camera not working or camera index wrong")
        else:
            raise e


def display():

    # Set Projection Matrix
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluOrtho2D(0, Conf.width, 0, Conf.height)

    # Switch to Model View Matrix
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    Glob.f.render()
    glutSwapBuffers()


def reshape(w, h):
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


glutInit(sys.argv)
glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH)
glutInitWindowSize(Conf.width, Conf.height)
glutInitWindowPosition(1366 + 10, 0)  # main window dim + 1
glutCreateWindow("OpenGL + OpenCV")
glutFullScreen()
init()
glutMainLoop()

