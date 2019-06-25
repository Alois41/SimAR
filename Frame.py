import cv2
import imutils

from brick import Brick
import numpy as np
from time import clock
from Global_tools import get_file_content, debug_draw_image
from Global_tools import Param as p
from Global_tools import glut_print
from image_tools import *
from drawing import *


class Camera:

    def __init__(self, width: int, height: int) -> void:
        # TODO detect the good camera instead of changing the number
        self.capture = cv2.VideoCapture(p.cam_number)
        self.capture.set(3, width)
        self.capture.set(4, height)
        _, self.image_raw = self.capture.read()

    def take_frame(self) -> void:
        """ Update current raw frame in BGR format"""
        _, self.image_raw = self.capture.read()


class Frame:
    """ Take and process webcam frames"""

    def __init__(self, width: int, height: int) -> void:
        self.grid = np.empty(p.dim_grille)
        self.grid[:] = np.nan
        self.width = width
        self.height = height
        self.cam = Camera(width, height)

        self.renderer = FrameRenderer()
        self.shader_handler = ShaderHandler("./shader/LavaShader")

        self.old_hand_zone = None
        self.triggered = False
        self.wait_time = 1
        self.stay_triggered = False
        self.trigger_cooldown = False
        self.triggered_number = 0
        self.shaderProgram = 0
        self.texture_location = 0
        self.time_location = 0
        self.offset = 0
        self.shader_clock = clock()

        self.lava_start = False
        self.hand_texture = None

        self.tex_handler = TextureHandler()

    def update_bricks(self, calibration: bool = False) -> (np.ndarray, list):
        """ Process the current raw frame and return a texture to draw and the  detected bricks"""

        image_raw = cv2.cvtColor(self.cam.image_raw, cv2.COLOR_BGR2RGBA)  # convert to RGBA
        image = adjust_gamma(image_raw.copy(), 1 if calibration else 2)  # enhance gamma of the image

        contours, image = find_center_contours(image)  # find contours in the center of the image

        if calibration:
            return image, []

        bricks = self.isolate_bricks(contours, image)  # detect bricks

        self.tex_handler.bind_update_texture(0, cv2.flip(image, 0), self.width, self.height)

        return image, bricks

    def render(self) -> void:
        if p.frame is not None:
            self.draw_frame(p.frame)
            self.draw_ui()

            self.draw_lava(p.width / 10, 0, 1.4 * p.width / 10, 2.2 * p.height / 3, 1, 0.250, 0.058)

    def draw_frame(self, image: np.ndarray, calibrate: bool = False) -> void:
        """ Draw the frame with OpenGL as a texture in the entire screen"""
        glClearColor(1, 1, 1, 1)
        if type(image) != int:

            # draw background
            draw_rectangle(0, 0, p.width, p.height, 1, 1, 1)

            # draw bricks
            ratio = (1 / 4, 1 / 4)
            self.draw_texture(0, 0, (1 - ratio[1]) * p.height, ratio[0] * p.width, ratio[1] * p.height)
            self.draw_texture(1, (1 - ratio[1]) * p.width, 0, ratio[0] * p.width, ratio[1] * p.height)

            if calibrate:
                draw_rectangle_empty(0, (1 - ratio[1]) * p.height, ratio[0] * p.width, ratio[1] * p.height, 0, 0, 0, 5)

            # draw grid
            y0, x0 = p.cam_area[0]
            yf, xf = p.cam_area[1]
            step_i = int((xf - x0) / p.dim_grille[0])
            step_j = int((yf - y0) / p.dim_grille[1])
            for i in range(p.dim_grille[0]):
                for j in range(p.dim_grille[1]):
                    draw_rectangle_empty(x0 + i * step_i, y0 + j * step_j, step_i, step_j,
                                         0, 0, 0, 5 if calibrate else 0.2)

    def draw_ui(self) -> void:
        """ Draw user interface and decoration"""
        y0, x0 = p.hand_area_1[0]
        yf, xf = p.hand_area_1[1]
        if not self.triggered:
            draw_rectangle(x0 + 40, yf, xf - x0, yf - y0, 0.4, 0.4, 0.4)
        else:
            draw_rectangle(x0 + 40, yf, xf - x0, yf - y0, 0, 1, 0)

        glut_print(x0 + 40 + 0.2 * (xf - x0), yf + 0.5 * (yf - y0), GLUT_BITMAP_HELVETICA_18, "", 0, 0, 0, 1.0, 1)

        # draw board informations

        x_start = 2.5 * p.width / 10

        if 0 <= clock() % (3 * p.swap_time) <= p.swap_time or not p.swap:
            self.draw_temperatures(x_start, 0, p.brick_array)
            glut_print(x_start, 100, GLUT_BITMAP_HELVETICA_18, "Temperatures", 0.0, 0.0, 0.0, 1.0, 1)

        elif p.swap_time <= clock() % (3 * p.swap_time) <= 2 * p.swap_time:
            self.draw_resistance_th(x_start, 0, p.brick_array)
            glut_print(x_start, 100, GLUT_BITMAP_HELVETICA_18, "Resistances thermiques", 0.0, 0.0, 0.0, 1.0, 1)

        else:
            self.draw_resistance_corr(x_start, 0, p.brick_array)
            glut_print(x_start, 100, GLUT_BITMAP_HELVETICA_18, "Resistances à la corrosion", 0.0, 0.0, 0.0, 1.0, 1)
            pass

        glut_print(x_start, p.height - 30, GLUT_BITMAP_HELVETICA_18,
                   "Nombre de coulées : %i" % p.f.triggered_number, 0.0, 0.0, 0.0, 1.0, 20)

    def isolate_bricks(self, contours: list, image: np.ndarray) -> list:
        """ create bricks from contours"""
        self.grid = np.empty(p.dim_grille)
        self.grid[:] = np.nan
        bricks = []

        # loop over the contours
        for index, c in enumerate(contours):
            # filter with Area
            area = cv2.contourArea(c)
            if area > p.max_brick_size or area < p.min_brick_size:
                continue

            # compute the center of the contour
            m = cv2.moments(c)
            if m["m00"] == 0:
                break
            c_x = int((m["m10"] / m["m00"]))
            c_y = int((m["m01"] / m["m00"]))

            # get a rotated rectangle from the contour
            rect = cv2.minAreaRect(c)
            # get vertices
            box = cv2.boxPoints(rect)
            box = np.int0(box)

            # Create a mask with the rectangle
            mask = np.zeros(image.shape[:2], np.uint8)
            cv2.drawContours(mask, [box], -1, (255, 255, 255), -1)

            # Shrink the mask to make sure that we are only inside the rectangle
            kernel = np.ones((10, 10), np.uint8)
            mask = cv2.erode(mask, kernel, iterations=4)

            # convert image in Hue Lightness Saturation space, better for taking mean color.
            image_hls = cv2.cvtColor(image.copy(), cv2.COLOR_RGB2HLS)
            # take the mean hue
            mean = 2 * cv2.mean(image_hls, mask=mask)[0]

            # Compare it to the colors dict and find the closest one
            colors = list(p.color_dict.keys())
            closest_colors = sorted(colors, key=lambda color: np.abs(color - mean))
            closest_color = closest_colors[0]
            name_color = p.color_dict[closest_color]

            # draw a bright color with the same hue
            c_rgb = tuple(cv2.cvtColor(np.uint8([[[0.5 * closest_color, 128, 255]]]), cv2.COLOR_HLS2RGB)[0][0])
            cv2.drawContours(image, [box], 0, tuple([int(x) for x in c_rgb]), thickness=cv2.FILLED)

            # create a brick and add it to the array
            b = Brick(rect, name_color, self.grid)
            cv2.putText(image, "%0.1f" % mean, (c_x - 20, c_y - 20), cv2.FONT_HERSHEY_SIMPLEX,
                        1.5, (0, 0, 0), thickness=10, lineType=10)
            cv2.putText(image, "%i%i" % (tuple(b.indexes[0])), (c_x - 20, c_y + 40), cv2.FONT_HERSHEY_SIMPLEX,
                        2, (0, 0, 0), thickness=10, lineType=10)

            bricks.append(b)

        return bricks

    def detect_hand(self) -> void:
        """ Detect hand in the button area """
        # hand detector cooldown
        if self.stay_triggered:
            if clock() - self.wait_time >= 2.0:
                self.stay_triggered = False
                self.trigger_cooldown = True
                self.wait_time = clock()
            return

        elif self.trigger_cooldown:
            if clock() - self.wait_time >= 2.0:
                self.trigger_cooldown = False
                y0, x0 = p.hand_area_1[0]
                yf, xf = p.hand_area_1[1]
                crop = cv2.cvtColor(crop_zone(self.cam.image_raw, x0, y0, xf - x0, yf - y0), cv2.COLOR_BGR2RGBA)
                self.tex_handler.bind_update_texture(1, cv2.flip(crop, 0), crop.shape[1], crop.shape[0])
            return

        # crop to the button zone
        y0, x0 = p.hand_area_1[0]
        yf, xf = p.hand_area_1[1]
        crop = cv2.cvtColor(crop_zone(self.cam.image_raw, x0, y0, xf - x0, yf - y0), cv2.COLOR_BGR2RGBA)

        # Luminance analysis is enough
        crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        crop_gray = cv2.GaussianBlur(crop_gray, (5, 5), 13)
        crop_gray = cv2.Canny(crop_gray, 40, 40)

        # enlarge shapes
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 9))
        crop_gray = cv2.dilate(crop_gray, kernel, 10)

        # help to close contours
        crop_gray[-1:, :] = 255
        crop_gray[:, :1] = 255
        crop_gray[:, -1:] = 255

        # find contour with a large enough area
        for c in cv2.findContours(crop_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]:
            self.triggered = cv2.contourArea(c) > p.hand_threshold
            if self.triggered:
                cv2.drawContours(crop, c, -1, (255, 0, 0), 3)
                self.tex_handler.bind_update_texture(1, cv2.flip(crop, 0), crop.shape[1], crop.shape[0])
                self.stay_triggered = True
                self.wait_time = clock()
                self.triggered_number += 1
                self.lava_start = True
                break

        self.tex_handler.bind_update_texture(1, cv2.flip(crop, 0), crop.shape[1], crop.shape[0])

    def draw_lava(self, x_s: int, y_s: int, w: int, h: int, r: float, g: float, b: float) -> void:
        """ draw "lava" from a texture"""

        draw_rectangle(x_s, y_s - 20, w, h, 0.3, 0.3, 0.3)
        draw_rectangle(x_s - 10, y_s - 20, 10, h, 0.1, 0.1, 0.1)

        draw_rectangle(x_s, 120, w, 20, 0.1, 0.1, 0.1)

        # load shader

        if self.triggered:
            delta_t = clock() - self.shader_clock
            self.offset += delta_t * 0.01
            self.shader_clock = clock()

            self.shader_handler.bind(data=self.offset)
            # update offset from clock to scroll the texture
            glEnable(GL_TEXTURE_2D)
            self.tex_handler.use_texture(2)
            draw_textured_rectangle(x_s, y_s, w, h)
            glDisable(GL_TEXTURE_2D)

            self.shader_handler.unbind()

        draw_rectangle(x_s, 120, .5 * w - 20, 100, 0.1, 0.1, 0.1)
        draw_rectangle(x_s + .5 * w + 20, 120, .5 * w - 20, 100, 0.1, 0.1, 0.1)

        draw_rectangle(x_s, 160, w, 0.5 * w, 0.1, 0.1, 0.1)

        draw_rectangle(x_s, 0, .5 * w - 20, 120, 0.3, 0.3, 0.3)
        draw_rectangle(x_s + .5 * w + 20, 0, .5 * w - 20, 120, 0.3, 0.3, 0.3)

    def draw_resistance_th(self, x_s: int, y_s: int, bricks: BrickArray) -> void:
        r_th = 255 * self.grid[:, :, 0]
        y0, x0 = p.cam_area[0]
        yf, xf = p.cam_area[1]
        step = int((xf - x0) / p.dim_grille[0])
        for index_c, t_l in enumerate(r_th):
            for index_l, t in enumerate(t_l):
                b_xy = bricks.get(index_c, index_l)
                t = b_xy.material.r_th if b_xy is not None else np.nan
                if not np.isnan(t):
                    draw_rectangle(x_s + index_c * step, y_s + (len(t_l) - index_l) * 20, step, 20, t, 0, 1 - t)
                else:
                    draw_rectangle(x_s + index_c * step, y_s + (len(t_l) - index_l) * 20, step, 20, 0, 0, 0)

                if b_xy is not None:
                    glut_print(x_s + index_c * step, y_s + (len(t_l) - index_l) * 20 + 5,
                               GLUT_BITMAP_HELVETICA_12, "%0.0f%%" % (100.0 * b_xy.material.r_th), 1, 1, 1, 1.0, 1)

    def draw_resistance_corr(self, x_s: int, y_s: int, bricks: BrickArray) -> void:
        r_corr = 255 * self.grid[:, :, 1]
        y0, x0 = p.cam_area[0]
        yf, xf = p.cam_area[1]
        step = int((xf - x0) / p.dim_grille[0])
        for index_c, t_l in enumerate(r_corr):
            for index_l, t in enumerate(t_l):
                b_xy = bricks.get(index_c, index_l)
                t = b_xy.material.r_cor if b_xy is not None else np.nan
                if not np.isnan(t):
                    draw_rectangle(x_s + index_c * step, y_s + (len(t_l) - index_l) * 20, step, 20, t, 0, 1 - t)
                else:
                    draw_rectangle(x_s + index_c * step, y_s + (len(t_l) - index_l) * 20, step, 20, 0, 0, 0)

                if b_xy is not None:
                    glut_print(x_s + index_c * step, y_s + (len(t_l) - index_l) * 20 + 5,
                               GLUT_BITMAP_HELVETICA_12, "%0.0f%%" % (100.0 * b_xy.material.r_cor), 1, 1, 1, 1.0, 1)

    def draw_temperatures(self, x_s: int, y_s: int, bricks: BrickArray) -> void:
        r_th = 255 * self.grid[:, :, 0]
        y0, x0 = p.cam_area[0]
        yf, xf = p.cam_area[1]
        step = int((xf - x0) / p.dim_grille[0])
        for index_c, t_l in enumerate(r_th):
            for index_l, t in enumerate(t_l):
                b_xy = bricks.get(index_c, index_l)
                t = b_xy.material.T_in / 1_600 if b_xy is not None else np.nan
                if not np.isnan(t):
                    draw_rectangle(x_s + index_c * step, y_s + (len(t_l) - index_l) * 20, step, 20, t, 0, 1 - t)
                else:
                    draw_rectangle(x_s + index_c * step, y_s + (len(t_l) - index_l) * 20, step, 20, 0, 0, 0)

                if b_xy is not None:
                    glut_print(x_s + index_c * step, y_s + (len(t_l) - index_l) * 20 + 5,
                               GLUT_BITMAP_HELVETICA_12, "%0.0f" % (b_xy.material.T_in - 273), 1, 1, 1, 1.0, 1)

    def draw_texture(self, tex_loc: int, x: int, y: int, l: int, h: int) -> void:
        glEnable(GL_TEXTURE_2D)
        self.tex_handler.use_texture(tex_loc)
        draw_textured_rectangle(x, y, l, h)
        glDisable(GL_TEXTURE_2D)


class FrameRenderer:

    def __init__(self):
        pass

    def render(self):
        pass


class TextureHandler:
    def __init__(self):
        # load texture
        self.texture_array = glGenTextures(3)
        lava_texture = cv2.imread("./texture/lava_diff.png")
        im = cv2.cvtColor(lava_texture, cv2.COLOR_BGR2RGBA)

        self.bind_texture(0, None, p.width, p.height)
        y0, x0 = p.hand_area_1[0]
        yf, xf = p.hand_area_1[1]
        self.bind_texture(1, None, xf - x0, yf - y0)
        self.bind_texture(2, im, im.shape[0], im.shape[1])

    def bind_texture(self, index: int, texture: np.ndarray or None, width: int, height: int) -> void:
        """ bind and create a texture for the first time in this loc"""
        glBindTexture(GL_TEXTURE_2D, self.texture_array[index])
        glShadeModel(GL_SMOOTH)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height,
                     0, GL_RGBA, GL_UNSIGNED_BYTE, texture)

    def bind_update_texture(self, index: int, texture: np.ndarray or None, width: int, height: int) -> void:
        """ bind and update a texture in this loc, faster"""
        glBindTexture(GL_TEXTURE_2D, self.texture_array[index])
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height,
                        GL_RGBA, GL_UNSIGNED_BYTE, texture)

    def use_texture(self, index):
        glBindTexture(GL_TEXTURE_2D, self.texture_array[index])


class ShaderHandler:

    def __init__(self, path: str) -> void:
        f_shader = compileShader(get_file_content(path + ".fs"), GL_FRAGMENT_SHADER)
        v_shader = compileShader(get_file_content(path + ".vs"), GL_VERTEX_SHADER)
        self.shaderProgram = glCreateProgram()
        glAttachShader(self.shaderProgram, v_shader)
        glAttachShader(self.shaderProgram, f_shader)
        glLinkProgram(self.shaderProgram)
        self.texture_location = glGetUniformLocation(self.shaderProgram, "myTexture")
        self.time_location = glGetUniformLocation(self.shaderProgram, "Time")

    def bind(self, data: any) -> void:
        glUseProgram(self.shaderProgram)
        glUniform1f(self.time_location, data)

    @staticmethod
    def unbind() -> void:
        glUseProgram(0)
