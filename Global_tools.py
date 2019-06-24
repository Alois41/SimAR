from time import clock
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
from OpenGL.GLUT.fonts import *
import numpy as np

def get_file_content(file):
    content = open(file, 'r').read()
    return content


def debug_draw_image(image):
    import matplotlib.pyplot as plt
    plt.ion()
    plt.imshow(image)
    plt.pause(0.01)


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


class Param:
    width = 1024
    height = 768
    nRange = 1.0

    cam_area = ((192, 256), (576, 768))
    hand_area_1 = ((200, 768), (376, 1024))
    cam_area_width = cam_area[0][1] - cam_area[0][0]
    cam_area_height = cam_area[1][1] - cam_area[1][0]

    dim_grille = (7, 4, 2)

    f, brick_array = None, []
    frame, frame_hand = None, None
    t_chamber = 293
    t_ref = clock()

    min_brick_size = 2_000.0
    max_brick_size = 20_000.0

    hand_threshold = 100

    color_dict = \
        {
            60: "Yellow",
            340: "Magenta",
            190: "Cyan",
            35: "Orange",
            0: "Red",
            80: "Green"
        }
