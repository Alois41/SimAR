from time import clock
from OpenGL.GL import *
from OpenGL.GLUT import *
from Frame import Frame
from OpenGL.GLU import *
from OpenGL.GLUT.fonts import *
from Global_tools import Param as p
from brick import Brick, BrickArray

import main__init__


def init():
    glClearColor(0.0, 0.0, 0.0, 1.0)
    glutDisplayFunc(display)
    glutReshapeFunc(reshape)
    glutKeyboardFunc(keyboard)
    glutIdleFunc(idle)
    p.f = Frame(p.width, p.height)


def idle():
    # capture next frame
    p.delta_t = clock() - p.t_ref

    # update frame
    p.f.cam.take_frame()
    # TODO: separate f for modes
    p.frame, new_bricks = p.f.update_bricks()

    if p.mode == 0:  # building mode

        # new_bricks = p.brick_array if p.f.triggered else new_bricks

        if p.brick_array is None or len(p.brick_array.array) == 0:
            p.brick_array = BrickArray(new_bricks)

        else:
            p.brick_array.invalidate()
            for nb in new_bricks:
                prev_b = p.brick_array.get(nb.indexes[0][0], nb.indexes[0][1])
                if prev_b is not None:
                    prev_b.replace(nb, p.f.grid)
                else:
                    for index in nb.indexes:
                        print("Brick added: " + str(nb.indexes[0]) + nb.material.color)
                        p.brick_array.set(index[0], index[1], nb)

            p.brick_array.clear_invalid()

    elif p.mode == 1:  # testing mode
        for j in range(p.dim_grille[1]):
            b = p.brick_array.get(0, j)
            if b is not None:
                if p.cooling:
                    p.t_chamber = p.temperature if p.f.triggered_start \
                        else max(p.t_chamber - p.cooling_factor * p.delta_t, 293)
                    p.brick_array.update(p.t_chamber)
                else:
                    p.brick_array.update(p.temperature, p.f.triggered_start)

    p.hand_text = p.f.detect_hand()
    glutPostRedisplay()


def display():

    # Set Projection Matrix
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluOrtho2D(0, p.width, 0, p.height)

    # Switch to Model View Matrix
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    p.f.render()
    glutSwapBuffers()


def reshape(w, h):
    if h == 0:
        h = 1
    glViewport(0, 0, w, h)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    # allows for reshaping the window without distoring shape
    if w <= h:
        glOrtho(-p.nRange, p.nRange, -p.nRange * h / w, p.nRange * h / w, -p.nRange, p.nRange)
    else:
        glOrtho(-p.nRange * w / h, p.nRange * w / h, -p.nRange, p.nRange, -p.nRange, p.nRange)

    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()


def keyboard(key, x, y):
    if key == b'\x1b':
        sys.exit()


glutInit(sys.argv)
glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH)
glutInitWindowSize(p.width, p.height)
glutInitWindowPosition(1366 + 1, 0)  # main window dim + 1
glutCreateWindow("OpenGL + OpenCV")
glutFullScreen()
init()
glutMainLoop()

