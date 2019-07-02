from time import clock
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import yaml
from docutils.nodes import section


def get_file_content(file):
    content = open(file, 'r').read()
    return content


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


class Config:
    with open(os.path.abspath("./config.yml"), 'r') as file:
        cfg = yaml.load(file)

    # Var from config

    section = cfg["screen"]
    width = section["width"]
    height = section["height"]

    section = cfg["grid"]
    dim_grille = section["dim_grille"]

    section = cfg["camera"]
    cam_area = section["cam_area"]
    cam_number = section["cam_number"]

    section = cfg["button"]
    hand_area_1 = section["hand_area_1"]
    hand_area_2 = section["hand_area_2"]
    hand_threshold_1 = section["hand_threshold_1"]
    hand_threshold_2 = section["hand_threshold_2"]

    section = cfg["brick"]
    min_brick_size = section["min_brick_size"]
    max_brick_size = section["max_brick_size"]

    section = cfg["program"]
    swap = section["swap"]
    swap_time = section["swap_time"]
    test_model = section["test_model"]

    color_dict = cfg["color"]
    color_to_mat = cfg["color_mat"]

    section = cfg["steel"]
    temperature = section["temperature"]
    cooling = section["cooling"]
    cooling_factor = section["cooling_factor"]


class Globals:

    cam_area_width = Config.cam_area[0][1] - Config.cam_area[0][0]
    cam_area_height = Config.cam_area[1][1] - Config.cam_area[1][0]
    nRange = 1.0
    f, brick_array = None, None
    frame, frame_hand = None, None
    t_chamber = 25
    t_ref = clock()
    hand_text = None

    updating = False
    update_timer = 0

    mode = 0
