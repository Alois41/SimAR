from time import clock
from OpenGL.GL import *
from OpenGL.GLUT import *
from Frame import Frame
from OpenGL.GLU import *
from OpenGL.GLUT.fonts import *
import freetype
import numpy as np

# window dimensions

width = 1024
height = 768
nRange = 1.0

dim_grille = (7, 4)

f, brick_array = None, []
frame, frame_hand = None, None
t_ref = clock()


def init():
    glClearColor(0.0, 0.0, 0.0, 1.0)
    glutDisplayFunc(display)
    glutReshapeFunc(reshape)
    glutKeyboardFunc(keyboard)
    glutIdleFunc(idle)


def idle():
    # capture next frame
    global f, brick_array, t_ref, frame
    delta_t = clock() - t_ref

    # update contours
    f.take_frame()
    frame, new_bricks = f.update_bricks()
    result = f.detect_hand()
    if result:
        print("yes")

    # update brick array if changes
    temp_b = []
    for nb in new_bricks:
        temp = False
        for b in brick_array:
            if nb.is_almost(b):
                temp = True
                b.replace(nb)
                temp_b.append(b)
                break

        if not temp:
            temp_b.append(nb)
    brick_array = temp_b
    glutPostRedisplay()


def display():
    global brick_array, frame

    # Set Projection Matrix
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluOrtho2D(0, width, 0, height)

    # Switch to Model View Matrix
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    if frame is not None:
        f.draw_frame(frame)
        f.draw_ui()

        x_start = 2.5 * width / 10
        swap_time = 3
        if 0 <= clock() % (3 * swap_time) <= swap_time:
            f.draw_temperatures(x_start, 0, brick_array)
            glut_print(x_start, 100, GLUT_BITMAP_HELVETICA_18, "Temperatures", 0.0, 0.0, 0.0, 1.0)

        elif swap_time <= clock() % (3 * swap_time) <= 2 * swap_time:
            f.draw_resistance_th(x_start, 0, brick_array)
            glut_print(x_start, 100, GLUT_BITMAP_HELVETICA_18, "Resistances thermiques", 0.0, 0.0, 0.0, 1.0)

        else:
            f.draw_resistance_corr(x_start, 0, brick_array)
            glut_print(x_start, 100, GLUT_BITMAP_HELVETICA_18, "Resistances à la corrosion", 0.0, 0.0, 0.0, 1.0)
            pass

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
        glOrtho(-nRange, nRange, -nRange * h / w, nRange * h / w, -nRange, nRange)
    else:
        glOrtho(-nRange * w / h, nRange * w / h, -nRange, nRange, -nRange, nRange)

    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()


def keyboard(key, x, y):
    if key == b'\x1b':
        sys.exit()


def glut_print(x, y, font, text, r, g, b, a):
    blending = False
    if glIsEnabled(GL_BLEND):
        blending = True

    # glEnable(GL_BLEND)
    glColor3f(r, g, b)
    glRasterPos2f(x, y)
    for ch in text:
        glutBitmapCharacter(font, ctypes.c_int(ord(ch)))

    if not blending:
        glDisable(GL_BLEND)


def main():
    global f

    glutInit(sys.argv)
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH)
    glutInitWindowSize(width, height)
    glutInitWindowPosition(1366 + 1, 0)  # main window dim + 1
    glutCreateWindow("OpenGL + OpenCV")
    glutFullScreen()

    f = Frame(width, height)
    init()
    glutMainLoop()


main()
