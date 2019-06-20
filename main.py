from time import clock
from OpenGL.GL import *
from OpenGL.GLUT import *
from Frame import Frame
from OpenGL.GLU import *
import numpy as np

# window dimensions

width = 1024
height = 768
nRange = 1.0

dim_grille = (7, 4)

f, boxArray = None, []
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
    global f, boxArray, t_ref, frame
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
        for b in boxArray:
            if nb.is_almost(b):
                temp = True
                b.replace(nb)
                temp_b.append(b)
                break

        if not temp:
            temp_b.append(nb)
    boxArray = temp_b

    # [print(int(5*brick.xStart / 768)) for brick in boxArray]


def display():
    global boxArray, frame

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

    [box.draw() for box in boxArray]
    # print("Brick number : %i " % len(boxArray))
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
