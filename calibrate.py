from time import clock
from OpenGL.GL import *
from OpenGL.GLUT import *
from Frame import Frame
from OpenGL.GLU import *
from Global_tools import Param as P


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
    P.frame, new_bricks = P.f.update_bricks(True)
    result = P.f.detect_hand()
    if result:
        print("yes")

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

        P.f.draw_frame(P.frame, True)

        # draw board informations

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


def glut_print(x, y, font, text, r, g, b, a, scale):
    blending = False
    if glIsEnabled(GL_BLEND):
        blending = True

    # glEnable(GL_BLEND)
    glPushMatrix()
    glColor3f(r, g, b)
    glRasterPos2f(x, y)

    for ch in text:
        glutBitmapCharacter(font, ctypes.c_int(ord(ch)))

    if not blending:
        glDisable(GL_BLEND)
    glPopMatrix()


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
