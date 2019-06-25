from time import clock
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
from OpenGL.GLUT.fonts import *
import numpy as np
import yaml


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
    with open("config.yml", 'r') as file:
        cfg = yaml.load(file)

    # Var from config

    section = cfg["screen"]
    width = section["width"]
    height = section["height"]

    section = cfg["grid"]
    dim_grille = section["dim_grille"]

    section = cfg["camera"]
    cam_area = section["cam_area"]

    section = cfg["button"]
    hand_area_1 = section["hand_area_1"]
    hand_threshold = section["hand_threshold"]

    section = cfg["brick"]
    min_brick_size = section["min_brick_size"]
    max_brick_size = section["max_brick_size"]

    section = cfg["program"]
    swap = section["swap"]
    swap_time = section["swap_time"]

    color_dict = cfg["color"]
    color_to_mat = cfg["color_mat"]

    section = cfg["steel"]
    temperature = section["temperature"]
    cooling = section["cooling"]

    # Global var

    cam_area_width = cam_area[0][1] - cam_area[0][0]
    cam_area_height = cam_area[1][1] - cam_area[1][0]
    nRange = 1.0
    f, brick_array = None, None
    frame, frame_hand = None, None
    t_chamber = 293
    t_ref = clock()
    hand_text = None

    updating = False
    update_timer = 0
