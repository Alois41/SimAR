import cv2
import imutils

from brick import Brick
from OpenGL.GL import *
from OpenGL.GL.shaders import *
from OpenGL.GLUT import *
import numpy as np
from time import clock
from Global_tools import get_file_content
from PIL import Image

dim_grille = (7, 4, 2)
grid = np.empty(dim_grille)
grid[:] = np.nan

color_dict = \
    {
        (0, 0, 0): "Black",
        (0.6, 0.9, 1): "Blue",
        (0.90, 1, 0.7): "Yellow",
        (1, 0.8, 0.9): "Magenta",
        (0, 1, 1): "Cyan",
        (1, 0.9, 0.7): "Orange",
        (1, 0, 0): "Red",
        (0.85, 1, 0.7): "Green"
    }

color_mat = \
    {
        "Black": (0, 0, 0),
        "Blue": (0, 0, 1),
        "Yellow": (1, 1, 0),
        "Magenta": (1, 0.078, 0),
        "Cyan": (0, 1, 1),
        "Orange": (1, 0.2, 0),
        "Red": (1, 0, 0),
        "Green": (0, 1, 0)
    }


class Texture(object):
    """Texture either loaded from a file or initialised with random colors."""

    def __init__(self):
        self.xSize, self.ySize = 0, 0
        self.rawRefence = None


class FileTexture(Texture):
    """Texture loaded from a file."""

    def __init__(self, fileName):
        im = Image.open(fileName)

        self.xSize = im.size[0]

        self.ySize = im.size[1]

        self.rawReference = im.tobytes("raw", "RGB", 0, -1)


class Frame:

    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.capture = cv2.VideoCapture(2)
        self.capture.set(3, self.width)
        self.capture.set(4, self.height)
        _, self.image_raw = self.capture.read()
        self.old_hand_zone = None
        self.triggered = False
        self.wait_time = 1
        self.wait = False
        self.triggered_number = 0
        self.shaderProgram = 0
        self.texture_location = 0
        self.time_location = 0
        self.shaderInit()
        self.offset = 0
        self.lava_height, self.lava_bottom = 0, 0
        self.shaderclock = clock()
        self.grid_R_th, self.grid_R_corr = 0, 0

    def take_frame(self):
        _, self.image_raw = self.capture.read()

    def update_bricks(self):
        image_raw = cv2.cvtColor(self.image_raw, cv2.COLOR_BGR2RGBA)
        image = adjust_gamma(image_raw.copy(), 2.0)

        contours = self.find_center_contours(image)
        image = self.zoom_center(0.5, image)

        bricks = self.isolate_bricks(contours, image)

        map = add_small_map(image)

        map = cv2.putText(map, "Nombre de coulees : %i" % self.triggered_number,
                          (250, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0),
                          thickness=2, lineType=10)

        self.get_grid(cv2.cvtColor(image.copy(), cv2.COLOR_RGBA2RGB))

        return map, bricks

    def zoom_center(self, ratio, image):
        ratio = ratio * .5
        crop_img = image[int((.5 - ratio) * self.height):int((.5 + ratio) * self.height),
                   int((.5 - ratio) * self.width):int((.5 + ratio) * self.width)]
        return imutils.resize(crop_img, image.shape[1], image.shape[0])

    def draw_frame(self, image):
        if type(image) != int:
            self.draw_background(image)

    def draw_background(self, image):
        # Create Texture
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, self.width, self.height, 0, GL_RGBA,
                     GL_UNSIGNED_BYTE, cv2.flip(image, 0))
        glutPostRedisplay()

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
        self.draw_lava(self.width / 10, self.height / 10, 1.4 * self.width / 10, 2 * self.height / 3, 1, 0.250, 0.058)
        if not self.triggered:
            draw_rectangle(8.8 * self.width / 10, self.height / 10, 1.4 * self.width / 10, 2 * self.height / 3, 0, 0, 1)
        else:
            draw_rectangle(8.8 * self.width / 10, self.height / 10, 1.4 * self.width / 10, 2 * self.height / 3, 0, 1, 0)

        self.draw_temperature(2.8 * self.width / 10, 0, 7 * self.width / 10, self.height / 10, 0, 1, 0)

    def find_center_contours(self, image):
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        thresh_rgb = blurred.copy()

        for index, channel in enumerate(cv2.split(thresh_rgb)):
            thresh = cv2.adaptiveThreshold(channel, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 103, 8)

            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

            thresh_rgb[:, :, index] = thresh

        thresh_gray = cv2.cvtColor(thresh_rgb, cv2.COLOR_RGB2GRAY)
        thresh_gray = self.zoom_center(0.5, thresh_gray)

        thresh_gray = cv2.bitwise_not(thresh_gray)
        cnts = cv2.findContours(thresh_gray.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        return cnts

    def isolate_bricks(self, contours, image):
        global grid
        grid = np.empty(dim_grille)
        grid[:] = np.nan

        # loop over the contours
        bricks = []
        for index, c in enumerate(contours):
            area = cv2.contourArea(c)
            if area > 30000 or area < 1000:
                continue
            # compute the center of the contour, then detect the name of the
            # shape using only the contour
            m = cv2.moments(c)
            if m["m00"] == 0:
                break
            c_x = int((m["m10"] / m["m00"]))
            c_y = int((m["m01"] / m["m00"]))

            rect = cv2.minAreaRect(c)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(image, [box], 0, (0, 191, 255), 2)

            # MEAN
            mask = np.zeros(image.shape[:2], np.uint8)
            cv2.drawContours(mask, [box], -1, (255, 255, 255), -1)
            mean = cv2.mean(image.copy(), mask=mask)
            mean = mean / np.max(mean[:3])

            colors = list(color_dict.keys())
            closest_colors = sorted(colors, key=lambda color: distance(color, mean[:3]))
            closest_color = closest_colors[0]

            name_color = color_dict[closest_color]
            cv2.drawContours(image, [box], 0, (255 * color_mat[name_color][0], 255 * color_mat[name_color][1],
                                               255 * color_mat[name_color][2], 1), thickness=cv2.FILLED)

            b = Brick(rect, name_color, grid)
            cv2.putText(image, (name_color[0] + "%i%i" % (b.xIndex, b.yIndex)),
                        (c_x - 20, c_y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), thickness=8, lineType=10)

            bricks.append(b)
        return bricks

    def detect_hand(self):
        if self.wait:
            self.lava_bottom = max(0.0, self.lava_bottom - 0.05)
            if clock() - self.wait_time >= 0.5:
                self.wait = False
            return False

        image_raw = cv2.cvtColor(self.image_raw, cv2.COLOR_BGR2LAB)  # CIE color space for visual differences
        # image = adjust_gamma(image_raw.copy(), 2)

        crop = crop_zone(image_raw, 3 * self.width / 4, self.height / 4, self.width / 10, self.height / 2)

        if self.old_hand_zone is None:
            self.old_hand_zone = crop

        diff = cv2.subtract(self.old_hand_zone, crop)

        value = np.max(diff)

        self.triggered = value > 100
        print(value)
        if 5 < value < 90:
            self.old_hand_zone = crop

        if self.triggered:
            self.wait = True
            self.wait_time = clock()
            self.triggered_number += 1
            self.lava_height = 1
            self.lava_bottom = 1
        else:
            if self.lava_bottom <= 0.0:
                self.lava_height = max(0, self.lava_height - 0.02)
            else:
                self.lava_bottom = max(0.0, self.lava_bottom - 0.05)
        return self.triggered

    def draw_lava(self, x_s, y_s, w, h, r, g, b):
        texture = FileTexture("./texture/lava_diff.png")
        # glClearColor(0, 0, 0, 0)
        glUseProgram(self.shaderProgram)

        self.offset += (clock() - self.shaderclock) * 0.01
        self.shaderclock = clock()
        glUniform1f(self.time_location, self.offset)

        glShadeModel(GL_SMOOTH)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexImage2D(GL_TEXTURE_2D, 0, 3, texture.xSize, texture.ySize, 0,
                     GL_RGB, GL_UNSIGNED_BYTE, texture.rawReference)

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
        glutSwapBuffers()
        glUseProgram(0)

    def shaderInit(self):
        f_shader = compileShader(get_file_content("LavaShader.fs"), GL_FRAGMENT_SHADER)
        v_shader = compileShader(get_file_content("LavaShader.vs"), GL_VERTEX_SHADER)
        self.shaderProgram = glCreateProgram()
        glAttachShader(self.shaderProgram, v_shader)
        glAttachShader(self.shaderProgram, f_shader)
        glLinkProgram(self.shaderProgram)
        self.texture_location = glGetUniformLocation(self.shaderProgram, "myTexture")
        self.time_location = glGetUniformLocation(self.shaderProgram, "Time")

    def get_grid(self, image):

        self.grid_R_th = np.array(imutils.resize(image, width=dim_grille[0] * 2)[:, :, 0]) / 255.0
        self.grid_R_corr = np.array(imutils.resize(image, width=dim_grille[0] * 2)[:, :, 1]) / 255.0
        pass

    def draw_temperature(self, x_s, y_s, w, h, r, g, b):
        global grid

        temperatures = 255 * grid[:, :, 0]
        for index_c, t_l in enumerate(temperatures):
            for index_l, t in enumerate(t_l):
                if not np.isnan(t):
                    draw_rectangle(x_s + index_c * 90, y_s + (len(t_l) - index_l) * 20, 85, 15, t, 0, 255 - t)
                else:
                    draw_rectangle(x_s + index_c * 90, y_s + (len(t_l) - index_l) * 20, 85, 15, 0, 0, 0)
        pass


def distance(c1, c2):
    (r1, g1, b1) = c1
    (r2, g2, b2) = c2
    return np.sqrt((r1 - r2) ** 2 + (g1 - g2) ** 2 + (b1 - b2) ** 2)


def add_small_map(image, x_offset=0, y_offset=0, x_scale=0.35, y_scale=0.2):
    blank = 255 * np.ones(image.shape)
    new_image = cv2.resize(image, (int(x_scale * image.shape[0]), int(y_scale * image.shape[1])))
    blank[int(y_offset):int(y_offset + new_image.shape[0]),
    int(x_offset):int(x_offset + new_image.shape[1])] = new_image
    return blank


def draw_rectangle(x_s, y_s, w, h, r, g, b):
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
