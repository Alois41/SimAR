from time import clock
from OpenGL.GL import *
from OpenGL.GLUT import *
from Frame import Frame
from OpenGL.GLU import *
from OpenGL.GLUT.fonts import *
import numpy as np
from Global_tools import glut_print
from Global_tools import Param as P
from brick import Brick


def init():
    glClearColor(0.0, 0.0, 0.0, 1.0)
    glutDisplayFunc(display)
    glutReshapeFunc(reshape)
    glutKeyboardFunc(keyboard)
    glutIdleFunc(idle)


def idle():
    # capture next frame
    P.delta_t = clock() - P.t_ref

    # update contours
    P.f.cam.take_frame()
    P.frame, new_bricks = P.f.update_bricks()

    new_bricks = P.brick_array if P.f.triggered else new_bricks

    P.f.detect_hand()

    # update brick array if changes
    temp_b = []
    for nb in new_bricks:
        temp = False
        for b in P.brick_array:
            if nb.is_almost(b):
                temp = True
                b.replace(nb)
                temp_b.append(b)
                break

        if not temp:
            temp_b.append(nb)
    P.brick_array = temp_b
    for j in range(P.dim_grille[1]):
        b = Brick.get_brick(P.brick_array, 0, j)
        if b is not None:
            P.t_chamber = 1600 if P.f.triggered else max(P.t_chamber - 10, 293)
            b.update(P.brick_array, t_out=P.t_chamber)
    glutPostRedisplay()


def display():

    # Set Projection Matrix
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluOrtho2D(0, P.width, 0, P.height)

    # Switch to Model View Matrix
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    if P.frame is not None:
        x_start = 2.5 * P.width / 10

        P.f.draw_frame(P.frame)
        P.f.draw_ui()

        # draw board informations

        swap_time = 3
        swap = True
        if 0 <= clock() % (3 * swap_time) <= swap_time or not swap:
            P.f.draw_temperatures(x_start, 0, P.brick_array)
            glut_print(x_start, 100, GLUT_BITMAP_HELVETICA_18, "Temperatures", 0.0, 0.0, 0.0, 1.0, 1)

        elif swap_time <= clock() % (3 * swap_time) <= 2 * swap_time:
            P.f.draw_resistance_th(x_start, 0, P.brick_array)
            glut_print(x_start, 100, GLUT_BITMAP_HELVETICA_18, "Resistances thermiques", 0.0, 0.0, 0.0, 1.0, 1)

        else:
            P.f.draw_resistance_corr(x_start, 0, P.brick_array)
            glut_print(x_start, 100, GLUT_BITMAP_HELVETICA_18, "Resistances à la corrosion", 0.0, 0.0, 0.0, 1.0, 1)
            pass

        glut_print(x_start, P.height - 30, GLUT_BITMAP_HELVETICA_18,
                   "Nombre de coulées : %i" % P.f.triggered_number, 0.0, 0.0, 0.0, 1.0, 20)

        P.f.draw_lava(P.width / 10, 0, 1.4 * P.width / 10, 2.2 * P.height / 3, 1, 0.250, 0.058)
    # [box.draw() for box in brick_array]

    glutSwapBuffers()


def reshape(w, h):
    if h == 0:
        h = 1

    glViewport(0, 0, w, h)
    glMatrixMode(GL_PROJECTION)

    glLoadIdentity()
    # allows for reshaping the window without distoring shape

    if w <= h:
        glOrtho(-P.nRange, P.nRange, -P.nRange * h / w, P.nRange * h / w, -P.nRange, P.nRange)
    else:
        glOrtho(-P.nRange * w / h, P.nRange * w / h, -P.nRange, P.nRange, -P.nRange, P.nRange)

    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()


def keyboard(key, x, y):
    if key == b'\x1b':
        sys.exit()

def main():
    glutInit(sys.argv)
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH)
    glutInitWindowSize(P.width, P.height)
    glutInitWindowPosition(1366 + 1, 0)  # main window dim + 1
    glutCreateWindow("OpenGL + OpenCV")
    glutFullScreen()

    P.f = Frame(P.width, P.height)
    init()
    glutMainLoop()


main()
