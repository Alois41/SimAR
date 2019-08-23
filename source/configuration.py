from time import clock
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import yaml
import numpy as np


class Config:
    with open(os.path.abspath("./config.yml"), 'r') as file:
        cfg = yaml.load(file)

    # Var from config

    section = cfg["screen"]
    width = section["width"]
    height = section["height"]

    section = cfg["grid"]
    dim_grille = section["dim_grille"]
    force_full = section["force_full"]

    section = cfg["camera"]
    cam_area = np.array([[.20*height, .20*width], [.80*height, .87*width]], dtype=np.uint16)
        #section["cam_area"]
    cam_number = section["cam_number"]
    refresh_rate = section["refresh_rate"]

    section = cfg["button"]
    hand_area_1 = section["hand_area_1"]
    hand_area_2 = section["hand_area_2"]
    hand_threshold_1 = section["hand_threshold_1"]
    hand_threshold_2 = section["hand_threshold_2"]
    cooldown = section["cooldown"]

    section = cfg["brick"]
    min_brick_size = section["min_brick_size"]
    max_brick_size = section["max_brick_size"]

    section = cfg["program"]
    swap = section["swap"]
    swap_time = section["swap_time"]
    test_model = section["test_model"]
    text_color = section["text_color"]

    color_dict = cfg["color"]
    color_to_mat = cfg["color_mat"]

    section = cfg["steel"]
    temperature = section["temperature"]
    cooling = section["cooling"]
    cooling_factor = section["cooling_factor"]


class Globals:

    debug = False

    cam_area_width = Config.cam_area[0][1] - Config.cam_area[0][0]
    cam_area_height = Config.cam_area[1][1] - Config.cam_area[1][0]
    nRange = 1.0
    brick_array = None
    frame, frame_hand = None, None
    t_chamber = 25
    t_ref, delta_t = clock(), 0
    hand_text = None

    updating = False
    update_timer = 0

    mode = 0
