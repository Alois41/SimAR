import cv2
import imutils

from brick import Brick
from OpenGL.GL import *
from OpenGL.GL.shaders import *
from OpenGL.GLUT import *
import numpy as np
from time import clock
from Global_tools import get_file_content, debug_draw_image
from PIL import Image
from Global_tools import Param as p
from Global_tools import glut_print


class Texture(object):
    """Texture either loaded from a file or initialised with random colors."""

    def __init__(self):
        self.xSize, self.ySize = 0, 0
        self.rawRefence = None


class FileTexture(Texture):
    """Texture loaded from a file."""

    def __init__(self, file_name):
        super().__init__()
        im = Image.open(file_name)
        self.xSize = im.size[0]
        self.ySize = im.size[1]
        self.rawReference = im.tobytes("raw", "RGB", 0, -1)


class Webcam:

    def __init__(self, width, height):
        # TODO detect the good camera instead of changing the number
        self.capture = cv2.VideoCapture(1)
        self.capture.set(3, width)
        self.capture.set(4, height)
        _, self.image_raw = self.capture.read()

    def take_frame(self):
        """ Update current raw frame in BGR format"""
        _, self.image_raw = self.capture.read()


def zoom_center(image):
    """ zoom in the center of the image"""
    crop_img = image[p.cam_area[0][0]:p.cam_area[1][0], p.cam_area[0][1]:p.cam_area[1][1]]
    return imutils.resize(crop_img, width=image.shape[1])


def find_center_contours(image):
    """ find all the contours in the center of a given image"""

    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    thresh_rgb = blurred.copy()

    # for each color channel
    for index, channel in enumerate(cv2.split(thresh_rgb)):
        # make channel binary with an adaptive threshold
        thresh = cv2.adaptiveThreshold(channel, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 103, 8)
        # Structure the channel with Rectangle shapes
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        # merge channels
        thresh_rgb[:, :, index] = thresh

    # convert into Gray scale(luminance)
    thresh_gray = cv2.cvtColor(thresh_rgb, cv2.COLOR_RGB2GRAY)
    # zoom in the center
    thresh_gray = zoom_center(thresh_gray)
    image = zoom_center(image)
    # invert black/white
    thresh_gray = cv2.bitwise_not(thresh_gray)

    kernel = np.ones((5, 5), np.uint8)
    thresh_gray = cv2.erode(thresh_gray, kernel, iterations=10)

    # find contours
    contours = cv2.findContours(thresh_gray.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)

    return contours, image


class Frame:
    """ Take and process webcam frames"""

    def __init__(self, width, height):
        self.grid = np.empty(p.dim_grille)
        self.grid[:] = np.nan
        self.width = width
        self.height = height
        self.cam = Webcam(width, height)

        self.old_hand_zone = None
        self.triggered = False
        self.wait_time = 1
        self.wait = False
        self.triggered_number = 0
        self.shaderProgram = 0
        self.texture_location = 0
        self.time_location = 0
        self.shader_init()
        self.offset = 0
        self.lava_height, self.lava_bottom = 1, 0
        self.shaderclock = clock()
        self.grid_R_th, self.grid_R_corr = 0, 0
        # load texture
        self.lava_texture = FileTexture("./texture/lava_diff.png")
        self.lava_start = False

    def update_bricks(self, calibration=False):
        """ Process the current raw frame and return a texture to draw and the  detected bricks"""

        image_raw = cv2.cvtColor(self.cam.image_raw, cv2.COLOR_BGR2RGBA)  # convert to RGBA
        image = adjust_gamma(image_raw.copy(), 1 if calibration else 3)  # enhance gamma of the image

        contours, image = find_center_contours(image)  # find contours in the center of the image

        if calibration:
            map = add_small_map(cv2.rectangle(image, (0, 0), (image.shape[1], image.shape[0]), (0, 0, 0),
                                              thickness=20))

            return map, []

        bricks = self.isolate_bricks(contours, image)  # detect bricks

        frame = add_small_map(image)  # put the image in a small part of the frame

        return frame, bricks

    def draw_frame(self, image, calibrate=False):
        """ Draw the frame with OpenGL as a texture in the entire screen"""
        if type(image) != int:
            y0, x0 = p.cam_area[0]
            yf, xf = p.cam_area[1]
            step_i = int((xf - x0) / p.dim_grille[0])
            step_j = int((yf - y0) / p.dim_grille[1])
            for i in range(p.dim_grille[0] + 1):
                image = cv2.line(image, (x0 + i * step_i, y0), (x0 + i * step_i, yf),
                                 (80, 80, 80) if not calibrate else (0, 0, 0),
                                 thickness=2 if not calibrate else 10)
            for j in range(p.dim_grille[1] + 1):
                image = cv2.line(image, (x0, y0 + j * step_j), (xf, y0 + j * step_j),
                                 (80, 80, 80) if not calibrate else (0, 0, 0),
                                 thickness=2 if not calibrate else 10)

            # Create Texture
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, self.width, self.height, 0, GL_RGBA,
                         GL_UNSIGNED_BYTE, cv2.flip(image, 0))

            glEnable(GL_TEXTURE_2D)

            glColor3f(1.0, 1.0, 1.0)
            glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)

            # Draw textured Quads
            glBegin(GL_QUADS)
            glTexCoord2f(0.0, 0.0)
            glVertex2f(0.0, 0.0)
            glTexCoord2f(1.0, 0.0)
            glVertex2f(self.width, 0.0)
            glTexCoord2f(1.0, 1.0)
            glVertex2f(self.width, self.height)
            glTexCoord2f(0.0, 1.0)
            glVertex2f(0.0, self.height)
            glEnd()

    def draw_ui(self):
        """ Draw user interface and decoration"""
        y0, x0 = p.hand_area_1[0]
        yf, xf = p.hand_area_1[1]
        if not self.triggered:
            draw_rectangle(x0 + 40, yf, xf - x0, yf - y0, 0.4, 0.4, 0.4)
        else:
            draw_rectangle(x0 + 40, yf, xf - x0, yf - y0, 0, 1, 0)

        glut_print(x0 + 40 + 0.2 * (xf - x0), yf + 0.5 * (yf - y0), GLUT_BITMAP_HELVETICA_18, "START", 0, 0, 0, 1.0, 1)

    def isolate_bricks(self, contours, image):
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

    def detect_hand(self):
        """ Detect hand in the button area """
        # hand detector cooldown
        if self.wait:
            self.lava_bottom = max(0.0, self.lava_bottom - 0.05)
            if clock() - self.wait_time >= 0.5:
                self.wait = False
            return False

        # CIE color space for visual differences
        image_raw = cv2.cvtColor(self.cam.image_raw, cv2.COLOR_BGR2RGB)
        image_raw = adjust_gamma(image_raw, 2)

        # crop to the button zone
        y0, x0 = p.hand_area_1[0]
        yf, xf = p.hand_area_1[1]
        crop = crop_zone(image_raw, x0, y0, xf - x0, yf - y0)

        # if first time, set reference to this
        if self.old_hand_zone is None:
            self.old_hand_zone = crop

        # compute difference between reference and now
        diff = cv2.subtract(self.old_hand_zone, crop)
        # take maximal value of difference, every pxls of diff should be black if nothing changed
        value = np.max(diff[:, :, :])

        # if above the threshold, trigger the button
        self.triggered = value > p.hand_threshold
        print(value)

        # if below the threshold, change the reference to this to counter light changes
        # if 5 < value < 90:
        #    self.old_hand_zone = crop

        # activate cooldown and setup "lava" drawing
        if self.triggered:
            self.wait = True
            self.wait_time = clock()
            self.triggered_number += 1
            self.lava_start = True

        return self.triggered

    def draw_lava(self, x_s, y_s, w, h, r, g, b):
        """ draw "lava" from a texture"""

        draw_rectangle(x_s, y_s - 20, w, h, 0.3, 0.3, 0.3)
        draw_rectangle(x_s - 10, y_s - 20, 10, h, 0.1, 0.1, 0.1)

        draw_rectangle(x_s, 120, w, 20, 0.1, 0.1, 0.1)

        # load shader

        if self.triggered:
            glUseProgram(self.shaderProgram)
            # update offset from clock to scroll the texture
            delta_t = clock() - self.shaderclock
            self.offset += delta_t * 0.01
            self.shaderclock = clock()

            glUniform1f(self.time_location, self.offset)
            glShadeModel(GL_SMOOTH)
            glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
            glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
            glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexImage2D(GL_TEXTURE_2D, 0, 3, self.lava_texture.xSize, self.lava_texture.ySize, 0,
                         GL_RGB, GL_UNSIGNED_BYTE, self.lava_texture.rawReference)

            glEnable(GL_TEXTURE_2D)

            glPushMatrix()

            glTranslatef(x_s, y_s, 0)
            glScalef(w, h, 1)

            glColor3f(1, 1, 1)

            glBegin(GL_QUADS)
            glTexCoord2f(0.0, self.lava_height)
            glVertex2f(0.0, self.lava_height)
            glTexCoord2f(0.0, self.lava_bottom)
            glVertex2f(0.0, self.lava_bottom)
            glTexCoord2f(1.0, self.lava_bottom)
            glVertex2f(1.0, self.lava_bottom)
            glTexCoord2f(1.0, self.lava_height)
            glVertex2f(1, self.lava_height)
            glEnd()

            glPopMatrix()
            glDisable(GL_TEXTURE_2D)
            glUseProgram(0)

        draw_rectangle(x_s, 120, .5 * w - 20, 100, 0.1, 0.1, 0.1)
        draw_rectangle(x_s + .5 * w + 20, 120, .5 * w - 20, 100, 0.1, 0.1, 0.1)

        draw_rectangle(x_s, 160, w, 0.5 * w, 0.1, 0.1, 0.1)

        draw_rectangle(x_s, 0, .5 * w - 20, 120, 0.3, 0.3, 0.3)
        draw_rectangle(x_s + .5 * w + 20, 0, .5 * w - 20, 120, 0.3, 0.3, 0.3)

    def shader_init(self):
        f_shader = compileShader(get_file_content("./shader/LavaShader.fs"), GL_FRAGMENT_SHADER)
        v_shader = compileShader(get_file_content("./shader/LavaShader.vs"), GL_VERTEX_SHADER)
        self.shaderProgram = glCreateProgram()
        glAttachShader(self.shaderProgram, v_shader)
        glAttachShader(self.shaderProgram, f_shader)
        glLinkProgram(self.shaderProgram)
        self.texture_location = glGetUniformLocation(self.shaderProgram, "myTexture")
        self.time_location = glGetUniformLocation(self.shaderProgram, "Time")

    def draw_resistance_th(self, x_s, y_s, bricks):
        r_th = 255 * self.grid[:, :, 0]
        y0, x0 = p.cam_area[0]
        yf, xf = p.cam_area[1]
        step = int((xf - x0) / p.dim_grille[0])
        for index_c, t_l in enumerate(r_th):
            for index_l, t in enumerate(t_l):
                b_xy = Brick.get_brick(bricks, index_c, index_l)
                t = b_xy.r_th if b_xy is not None else np.nan
                if not np.isnan(t):
                    draw_rectangle(x_s + index_c * step, y_s + (len(t_l) - index_l) * 20, step, 20, t, 0, 1 - t)
                else:
                    draw_rectangle(x_s + index_c * step, y_s + (len(t_l) - index_l) * 20, step, 20, 0, 0, 0)

                if b_xy is not None:
                    glut_print(x_s + index_c * step, y_s + (len(t_l) - index_l) * 20 + 5,
                               GLUT_BITMAP_HELVETICA_12, "%0.0f%%" % (100.0 * b_xy.r_th), 1, 1, 1, 1.0, 1)

    def draw_resistance_corr(self, x_s, y_s, bricks):
        r_corr = 255 * self.grid[:, :, 1]
        y0, x0 = p.cam_area[0]
        yf, xf = p.cam_area[1]
        step = int((xf - x0) / p.dim_grille[0])
        for index_c, t_l in enumerate(r_corr):
            for index_l, t in enumerate(t_l):
                b_xy = Brick.get_brick(bricks, index_c, index_l)
                t = b_xy.r_cor if b_xy is not None else np.nan
                if not np.isnan(t):
                    draw_rectangle(x_s + index_c * step, y_s + (len(t_l) - index_l) * 20, step, 20, t, 0, 1 - t)
                else:
                    draw_rectangle(x_s + index_c * step, y_s + (len(t_l) - index_l) * 20, step, 20, 0, 0, 0)

                if b_xy is not None:
                    glut_print(x_s + index_c * step, y_s + (len(t_l) - index_l) * 20 + 5,
                               GLUT_BITMAP_HELVETICA_12, "%0.0f%%" % (100.0 * b_xy.r_cor), 1, 1, 1, 1.0, 1)

    def draw_temperatures(self, x_s, y_s, bricks):
        r_th = 255 * self.grid[:, :, 0]
        y0, x0 = p.cam_area[0]
        yf, xf = p.cam_area[1]
        step = int((xf - x0) / p.dim_grille[0])
        for index_c, t_l in enumerate(r_th):
            for index_l, t in enumerate(t_l):
                b_xy = Brick.get_brick(bricks, index_c, index_l)
                t = b_xy.T_in / 1_600 if b_xy is not None else np.nan
                if not np.isnan(t):
                    draw_rectangle(x_s + index_c * step, y_s + (len(t_l) - index_l) * 20, step, 20, t, 0, 1 - t)
                else:
                    draw_rectangle(x_s + index_c * step, y_s + (len(t_l) - index_l) * 20, step, 20, 0, 0, 0)

                if b_xy is not None:
                    glut_print(x_s + index_c * step, y_s + (len(t_l) - index_l) * 20 + 5,
                               GLUT_BITMAP_HELVETICA_12, "%0.0f" % (b_xy.T_in - 273), 1, 1, 1, 1.0, 1)


def add_small_map(image, x_offset=0, y_offset=0, x_scale=0.35, y_scale=0.15):
    blank = 255 * np.ones(image.shape)
    new_image = cv2.resize(image, (int(x_scale * image.shape[0]), int(y_scale * image.shape[1])))
    blank[int(y_offset):int(y_offset + new_image.shape[0]),
    int(x_offset):int(x_offset + new_image.shape[1])] = new_image
    return blank


def draw_rectangle(x_s, y_s, w, h, r, g, b):
    glUseProgram(0)
    glClearColor(0, 0, 0, 1)
    glPushMatrix()
    glColor3f(r, g, b)
    glTranslatef(x_s, y_s, 0)
    glScalef(w, h, 1)
    glBegin(GL_QUADS)  # start drawing a rectangle
    glVertex2f(0, 0)  # bottom left point
    glVertex2f(1, 0)  # bottom right point
    glVertex2f(1, 1)  # top right point
    glVertex2f(0, 1)  # top left point
    glEnd()
    glPopMatrix()


def adjust_gamma(image, gamma=2.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")

    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)


def crop_zone(image, x, y, l, h):
    return image[int(y):int(y + h), int(x):int(x + l)]
