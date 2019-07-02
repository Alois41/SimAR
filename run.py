import sys
import os.path
sys.path.append(os.path.abspath("./source"))
from source import *
from time import clock
from OpenGL.GL import *
from OpenGL.GLUT import *
from Frame import Frame
from OpenGL.GLU import *
from OpenGL.GLUT.fonts import *
from Global_tools import Config as Conf, Globals as Glob
from brick import BrickArray


def init():
    glClearColor(0.0, 0.0, 0.0, 1.0)
    glutDisplayFunc(display)
    glutReshapeFunc(reshape)
    glutKeyboardFunc(keyboard)
    glutIdleFunc(idle)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    Glob.f = Frame(Conf.width, Conf.height)


def idle():
    if 1:
        # capture next frame
        Conf.delta_t = clock() - Glob.t_ref

        # update frame
        Glob.f.cam.take_frame()
        # TODO: separate f for modes
        Glob.frame, new_bricks = Glob.f.update_bricks()

        if Glob.mode == 0:  # building mode

            # new_bricks = p.brick_array if p.f.triggered else new_bricks

            if Glob.brick_array is None or len(Glob.brick_array.array) == 0:
                Glob.brick_array = BrickArray(new_bricks)

            else:
                Glob.brick_array.invalidate()
                for nb in new_bricks:
                    prev_b = Glob.brick_array.get(nb.indexes[0][0], nb.indexes[0][1])
                    if prev_b is not None:
                        prev_b.replace(nb)
                    else:
                        for index in nb.indexes:
                            print("Brick added: " + str(nb.indexes[0]) + nb.material.color)
                            Glob.brick_array.set(index[0], index[1], nb)

                Glob.brick_array.clear_invalid()

        elif Glob.mode == 1:  # testing mode
            for j in range(Conf.dim_grille[1]):
                b = Glob.brick_array.get(0, j)
                if b is not None:
                    if Conf.cooling:
                        Conf.t_chamber = Conf.temperature if Glob.f.triggered_start and Glob.f.triggered_number > 0 \
                            else max(Conf.t_chamber - Conf.cooling_factor * Conf.delta_t, 293)
                        Glob.brick_array.update(Conf.t_chamber)
                    else:
                        Glob.brick_array.update(Conf.temperature, Glob.f.triggered_start and Glob.f.triggered_number > 0)

        Conf.hand_text = Glob.f.detect_hand()
        glutPostRedisplay()
    # except Exception as e:
    #     if "empty" in str(e):
    #        raise IOError("Camera not working or camera index wrong")
    #     else:
    #         raise e


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

