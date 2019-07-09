import time

from source.brick import Brick, BrickArray
from time import clock
from source.Global_tools import get_file_content
from source.Global_tools import Config as Conf, Globals as Glob
from source.Global_tools import glut_print
from source.image_tools import *
from source.drawing import *
from OpenGL.GL import *
from datetime import timedelta
from source.resources import Resources
import random

strings = Resources.strings['fr']


class Camera:

    def __init__(self, width: int, height: int) -> None:
        self.capture = cv2.VideoCapture(Conf.cam_number)
        if not self.capture.isOpened():
            raise IOError("Webcam not plugged or wrong webcam index")
        time.sleep(3)
        self.capture.set(3, width)
        self.capture.set(4, height)
        self.capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, .25)
        self.capture.set(cv2.CAP_PROP_EXPOSURE, -5)
        self.capture.set(cv2.CAP_PROP_BRIGHTNESS, 5)
        # self.capture.set(cv2.CAP_PROP_CONTRAST, 128)
        for i in range(10):
            _, self.image_raw = self.capture.read()

    def take_frame(self) -> void:
        """ Update current raw frame in BGR format"""
        _, self.image_raw = self.capture.read()


class Frame:
    """ Take and process webcam frames"""

    def __init__(self, width: int, height: int) -> void:
        self.width = width
        self.height = height
        self.cam = Camera(width, height)

        self.shader_handler_molten_steel = ShaderHandler("./shader/LavaShader", ["myTexture", "offset_x", "offset_y", "delta_t"])
        self.shader_handler_brick = ShaderHandler("./shader/TemperatureShader",
                                                  ["Corrosion", "T", "brick_dim", "grid_pos", "step", "border"])

        self.old_hand_zone = None
        self.triggered_start, self.triggered_reset = False, False
        self.wait_time = 1
        self.stay_triggered = False
        self.trigger_cooldown = False
        self.triggered_number = 0
        self.shaderProgram = 0
        self.texture_location = 0
        self.time_location = 0
        self.offset_x, self.offset_y = 0, 0
        self.shader_clock = clock()
        self.steel_flow_dir_x, self.steel_flow_dir_y = 1, 1

        self.hand_texture = None

        self.tex_handler = TextureHandler()

        self.buttonStart = HandButton(3, None, Conf.hand_area_1, Conf.hand_threshold_1, 4800)
        self.buttonReset = HandButton(3, self.tex_handler, Conf.hand_area_2, Conf.hand_threshold_2, 2300)

        self.frame_count = 0

    def update_bricks(self, calibration: bool = False) -> (np.ndarray, list):
        """ Process the current raw frame and return a texture to draw and the  detected bricks"""

        image = None
        self.frame_count += 1
        # update each n frames
        if Glob.mode == 0 and self.frame_count % 5 == 1:  # building mode
            image_raw = cv2.cvtColor(self.cam.image_raw, cv2.COLOR_BGR2RGBA)  # convert to RGBA
            image_raw[np.std(image_raw[:, :, :3], axis=2) < 12.0] = [255, 255, 255, 255]
            # image_raw = adjust_gamma(image_raw.copy(), 1 if calibration else 2)  # enhance gamma of the image
            contours, image = find_center_contours(image_raw)  # find contours in the center of the image

            if calibration:
                return image, []

            bricks = self.isolate_bricks(contours, image)  # detect bricks

            image[:, :, 3] = 255 * np.ones((image.shape[0], image.shape[1]))

            if Glob.brick_array is None or len(Glob.brick_array.array) == 0:
                Glob.brick_array = BrickArray(bricks)

            else:
                # Update array with new bricks if needed
                Glob.brick_array.invalidate()
                for nb in bricks:
                    for index in nb.indexes:
                        prev_b = Glob.brick_array.get(index[0], index[1])
                        if prev_b is not None and prev_b.is_almost(nb):
                            prev_b.replace(nb)
                        else:
                            # print("Brick added: " + str(nb.indexes[0]) + " (%s)" % nb.material.color)
                            Glob.brick_array.set(index[0], index[1], nb)

                Glob.brick_array.clear_invalid()

        if image is not None:
            texture = cv2.resize(image, (Conf.width, Conf.height))
            self.tex_handler.bind_texture(0, cv2.flip(texture, 0), Conf.width, Conf.height)
            Glob.frame = image

    def render(self) -> void:
        if Glob.frame is not None:
            self.draw_frame(Glob.frame)
            y0, x0 = Conf.cam_area[0]
            yf, xf = Conf.cam_area[1]

            self.draw_molten_steel(x0-100, y0, xf - x0 + 100, yf - y0, 1, 1, 1)
            self.draw_ui()

    def draw_frame(self, image: np.ndarray) -> void:
        """ Draw the frame with OpenGL as a texture in the entire screen"""
        # glClearColor(1, 1, 1, 1)
        if type(image) != int:
            # draw background
            draw_rectangle(0, 0, Conf.width, Conf.height, 1, 1, 1)

            y0, x0 = Conf.cam_area[0]
            yf, xf = Conf.cam_area[1]

            if Glob.mode == 1:
                draw_rectangle(x0, y0, xf - x0, yf - y0, 0, 0, 0)

            # draw bricks
            ratio = (.2, .2)
            # hand image

            y0, x0 = Conf.hand_area_1[0]
            yf, xf = Conf.hand_area_1[1]
            self.draw_texture(1, Conf.width - (xf - x0), 0, xf - x0, yf - y0)
            draw_rectangle_empty(0, (1 - ratio[1]) * Conf.height, ratio[0] * Conf.width, ratio[1] * Conf.height,
                                 0, 0, 0, 5)

    def draw_ui(self) -> void:
        """ Draw user interface and decoration"""

        # draw button interface (start)

        y0, x0 = Conf.hand_area_1[0]
        yf, xf = Conf.hand_area_1[1]
        if not self.buttonStart.stay_triggered:
            draw_rectangle(x0, yf, xf - x0, yf - y0, 0.4, 0.4, 0.4)
        else:
            draw_rectangle(x0, yf, xf - x0, yf - y0, 0, 1, 0)

        if Glob.mode == 0:
            message = "PRET"
        elif not self.buttonStart.trigger_cooldown:
            message = "PLUS"
        else:
            message = ""

        glut_print(x0 + 20, yf + 0.5 * (yf - y0), GLUT_BITMAP_HELVETICA_18, message, 0, 0, 0)

        # draw button interface (reset)

        y0, x0 = Conf.hand_area_2[0]
        yf, xf = Conf.hand_area_2[1]
        if not self.triggered_reset:
            draw_rectangle(x0, yf, xf - x0, yf - y0, 0.4, 0.4, 0.4)
        else:
            draw_rectangle(x0, yf, xf - x0, yf - y0, 0, 1, 0)

        glut_print(x0, yf + 0.5 * (yf - y0), GLUT_BITMAP_HELVETICA_18, " RST", 0, 0, 0)

        # draw board informations

        x_start = 2.5 * Conf.width / 10

        if 0 <= clock() % (3 * Conf.swap_time) <= Conf.swap_time or not Conf.swap or Glob.mode == 1:
            self.draw_temperatures(Conf.cam_area[0][1], 0, Glob.brick_array)

        elif Conf.swap_time <= clock() % (3 * Conf.swap_time) <= 2 * Conf.swap_time:
            self.draw_thermal_diffusivity(Conf.cam_area[0][1], 0)
        else:
            self.draw_resistance_corr(Conf.cam_area[0][1], 0)
            pass

        if Glob.mode == 0:
            title = strings['title_build']
            subtitle = strings['sub_title_build']
        else:
            title = strings['title_test']
            subtitle = strings['sub_title_test1i'] % self.triggered_number

        glut_print(x_start, Conf.height - 30, GLUT_BITMAP_HELVETICA_18,
                   title + "    %0.0f fps" % (1 / Glob.delta_t), 0.0, 0.0, 0.0)
        glut_print(x_start, Conf.height - 60, GLUT_BITMAP_HELVETICA_18, subtitle, 0.0, 0.0, 0.0)

        ratio = (.2, .2)

        y0, x0 = Conf.cam_area[0]
        yf, xf = Conf.cam_area[1]

        # brick map
        self.draw_texture(0, 0, (1 - ratio[1]) * Conf.height, ratio[0] * Conf.width, ratio[1] * Conf.height)
        # draw grid
        step_i = int((xf - x0) / Conf.dim_grille[0])
        step_j = int((yf - y0) / Conf.dim_grille[1])
        for i in range(Conf.dim_grille[0]):
            for j in range(Conf.dim_grille[1]):
                draw_rectangle_empty(x0 + i * step_i, y0 + j * step_j, step_i, step_j, 0.2, 0.2, 0.2, 4)

    @staticmethod
    def isolate_bricks(contours: list, image: np.ndarray) -> list:
        """ create bricks from contours"""
        bricks = []

        # loop over the contours
        for index, c in enumerate(contours):
            # filter with Area
            area = cv2.contourArea(c)
            if area > Conf.max_brick_size or area < Conf.min_brick_size:
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
            cv2.drawContours(mask, [box], -1, (255, 255, 255, 255), -1)

            # Shrink the mask to make sure that we are only inside the rectangle
            # kernel = np.ones((10, 10), np.uint8)
            # mask = cv2.erode(mask, kernel, iterations=4)

            # convert image in Hue Lightness Saturation space, better for taking mean color.
            image_hls = cv2.cvtColor(image.copy(), cv2.COLOR_RGB2HLS)
            # take the mean hue
            mean = 2 * cv2.mean(image_hls, mask=mask)[0]

            # Compare it to the colors dict and find the closest one
            colors = list(Conf.color_dict.keys())
            closest_colors = sorted(colors, key=lambda color: np.abs(color - mean))
            closest_color = closest_colors[0]
            name_color = Conf.color_dict[closest_color]

            # draw a bright color with the same hue
            c_rgb = cv2.cvtColor(np.uint8([[[0.5 * closest_color, 128, 255]]]), cv2.COLOR_HLS2RGB)[0][0]
            cv2.drawContours(image, [box], 0, tuple([int(x) for x in c_rgb]), thickness=cv2.FILLED)

            # create a brick and add it to the array
            b = Brick.new(rect, closest_color)
            cv2.putText(image, "%0.0f" % mean, (c_x - 20, c_y - 20), cv2.FONT_HERSHEY_SIMPLEX,
                        1.5, (0, 0, 0), thickness=10, lineType=10)
            cv2.putText(image, "%i%i" % (tuple(b.indexes[0])), (c_x - 20, c_y + 40), cv2.FONT_HERSHEY_SIMPLEX,
                        2, (0, 0, 0), thickness=10, lineType=10)

            bricks.append(b)

        return bricks

    def detect_hand(self) -> void:
        self.triggered_start, new = self.buttonStart.detect_hand(self.cam.image_raw)
        if Glob.mode == 0 and new:
            if Glob.brick_array.is_valid():
                Glob.brick_array.init_heat_eq()
                Glob.mode = 1
            else:
                strings['sub_title_build'] = "La grille n'est pas remplie"
        elif Glob.mode == 1:
            if new:
                self.triggered_number += 1

        self.triggered_reset, new = self.buttonReset.detect_hand(self.cam.image_raw)
        if self.triggered_reset:
            self.triggered_number = 0
            Glob.brick_array.reset()  # temperature reset
            Glob.mode = 0

    def draw_molten_steel(self, x_s: int, y_s: int, w: int, h: int, r: float, g: float, b: float) -> void:
        """ draw "molten_steel" from a texture"""

        #draw_rectangle(x_s - 20, y_s - 10, w + 30, h + 20, 0.2, 0.2, 0.2)
        draw_rectangle(x_s - 20, y_s - 10, w + 30, h + 20, 0.2, 0.2, 0.2)
        draw_rectangle(x_s, y_s, w, h, .3, .3, .3)

        # load shader

        if self.triggered_start and self.triggered_number > 0:

            delta_t = clock() - self.shader_clock

            # if random.random() > .999:
            #     self.steel_flow_dir_x *= -1
            # if random.random() > .999:
            #     self.steel_flow_dir_y *= -1

            self.offset_x += self.steel_flow_dir_x * random.random() * delta_t * 0.005
            self.offset_y += self.steel_flow_dir_y * random.random() * delta_t * 0.005
            self.shader_clock = clock()

            self.shader_handler_molten_steel.bind(data=[None, self.offset_x, self.offset_y, delta_t])
            # update offset from clock to scroll the texture
            glEnable(GL_TEXTURE_2D)
            self.tex_handler.use_texture(2)
            glColor(r, g, b, 1)
            draw_textured_rectangle(x_s, y_s, w, h)
            glDisable(GL_TEXTURE_2D)

            self.shader_handler_molten_steel.unbind()

    @staticmethod
    def draw_thermal_diffusivity(x_s: int, y_s: int) -> void:
        glut_print(x_s, 100, GLUT_BITMAP_HELVETICA_18, "Diffusivité thermique", 0.0, 0.0, 0.0)
        y0, x0 = Conf.cam_area[0]
        yf, xf = Conf.cam_area[1]
        step = int((xf - x0) / Conf.dim_grille[0])
        for c in range(Conf.dim_grille[0]):
            for l in range(Conf.dim_grille[1]):
                b_xy = Glob.brick_array.get(c, l)
                t = b_xy.material.diffusivity if b_xy is not None else np.nan
                if not np.isnan(t):
                    draw_rectangle(x_s + c * step, y_s + (Conf.dim_grille[1] - l - 1) * 20, step, 20, t * 1E5, 0,
                                   1 - t * 1E5)
                else:
                    draw_rectangle(x_s + c * step, y_s + (Conf.dim_grille[1] - l - 1) * 20, step, 20, 0, 0, 0)

                if b_xy is not None:
                    glut_print(x_s + c * step, y_s + (Conf.dim_grille[1] - l - 1) * 20 + 5,
                               GLUT_BITMAP_HELVETICA_12, "%0.2E" % t, 1, 1, 1)

    @staticmethod
    def draw_resistance_corr(x_s: int, y_s: int) -> void:
        glut_print(x_s, 100, GLUT_BITMAP_HELVETICA_18, "Resistances à la corrosion", 0.0, 0.0, 0.0)
        y0, x0 = Conf.cam_area[0]
        yf, xf = Conf.cam_area[1]
        step = int((xf - x0) / Conf.dim_grille[0])
        for c in range(Conf.dim_grille[0]):
            for l in range(Conf.dim_grille[1]):
                b_xy = Glob.brick_array.get(c, l)
                t = b_xy.material.r_cor if b_xy is not None else np.nan
                if not np.isnan(t):
                    draw_rectangle(x_s + c * step, y_s + (Conf.dim_grille[1] - l - 1) * 20, step, 20, t, 0, 1 - t)
                else:
                    draw_rectangle(x_s + c * step, y_s + (Conf.dim_grille[1] - l - 1) * 20, step, 20, 0, 0, 0)

                if b_xy is not None:
                    glut_print(x_s + c * step, y_s + (Conf.dim_grille[1] - l - 1) * 20 + 5,
                               GLUT_BITMAP_HELVETICA_12, "%0.0f%%" % (100.0 * b_xy.material.r_cor), 1, 1, 1)

    def draw_temperatures(self, x_s: int, y_s: int, bricks) -> void:
        glut_print(x_s, 100, GLUT_BITMAP_HELVETICA_18,
                   "Temperatures %s" % str(timedelta(seconds=Glob.brick_array.sim_time)), 0.0, 0.0, 0.0)
        for i in range(2):
            y0, x0 = Conf.cam_area[0]
            yf, xf = Conf.cam_area[1]
            h = 20

            if i == 1 and Glob.mode == 1:
                w, h = xf - x0, (yf - y0) / Conf.dim_grille[1]
                x_s, y_s = x0, y0

            step = int((xf - x0) / Conf.dim_grille[0])
            for index_c in range(Conf.dim_grille[0]):
                for index_l in range(Conf.dim_grille[1]):
                    b_xy: Brick = bricks.get(index_c, index_l)
                    if not b_xy.is_void and not b_xy.material.is_broken:
                        index = b_xy.indexes[0]
                        temp_array = Glob.brick_array.T
                        pos = index[1] * Glob.brick_array.step_x * (temp_array.shape[1]) + index[
                            0] * Glob.brick_array.step_y

                        border = 0
                        if index[0] < Conf.dim_grille[0] - 1:
                            border += 1
                        if index[1] < Conf.dim_grille[1] - 1:
                            border += 2

                        self.shader_handler_brick.bind([b_xy.material.health, temp_array.flatten().copy(),
                                                        [Glob.brick_array.step_x, Glob.brick_array.step_y],
                                                        pos, Glob.brick_array.nx, border])

                        draw_rectangle(x_s + index_c * step, y_s + (Conf.dim_grille[1] - index_l - 1) * h, step, h)
                        self.shader_handler_brick.unbind()
                        glut_print(x_s + index_c * step, y_s + (Conf.dim_grille[1] - index_l - 1) * h + 5,
                                   GLUT_BITMAP_HELVETICA_12,
                                   "%0.0f" % np.mean(Glob.brick_array.get_temp(index[0], index[1])), 1, 1, 1)

                    else:
                        draw_rectangle(x_s + index_c * step, y_s + (Conf.dim_grille[1] - index_l - 1) * h,
                                       step, h, 0, 0, 0, .5 if b_xy.drowned else 1)

    def draw_texture(self, tex_loc: int, x: int, y: int, l: int, h: int) -> void:
        glEnable(GL_TEXTURE_2D)
        self.tex_handler.use_texture(tex_loc)
        draw_textured_rectangle(x, y, l, h)
        glDisable(GL_TEXTURE_2D)


class HandButton:

    def __init__(self, wait_time: float, texture_handler, hand_area: list, threshold: int, max: int) -> void:
        """ Init of one button """
        self.stay_triggered = False
        self.trigger_cooldown = False
        self.wait_time = wait_time
        self.tex_handler = texture_handler
        self.hand_area = hand_area
        self.threshold = threshold
        self.max = max

        y0, x0 = self.hand_area[0]
        yf, xf = self.hand_area[1]
        if self.tex_handler is not None:
            self.tex_handler.bind_texture(1, None, xf - x0, yf - y0)

    def detect_hand(self, image: np.ndarray) -> tuple:
        """ Detect hand in the button area """
        # hand detector cooldown
        y0, x0 = self.hand_area[0]
        yf, xf = self.hand_area[1]

        if self.stay_triggered:
            if clock() - self.wait_time >= .5:
                self.stay_triggered = False
                self.wait_time = clock()
            return True, False
        elif self.trigger_cooldown:
            if clock() - self.wait_time >= Conf.cooldown:
                self.trigger_cooldown = False

                crop = cv2.cvtColor(crop_zone(image, x0, y0, xf - x0, yf - y0), cv2.COLOR_BGR2RGBA)
                if self.tex_handler is not None:
                    self.tex_handler.bind_texture(1, cv2.flip(crop, 0), xf - x0, yf - y0)
            return True, False

        # crop to the button zone
        crop = cv2.cvtColor(crop_zone(image, x0, y0, xf - x0, yf - y0), cv2.COLOR_BGR2RGBA)
        crop[:, :, 3] = 255.0 * np.ones((crop.shape[0], crop.shape[1]))

        # crop = cv2.addWeighted(crop, 1.5, crop, 0, -200)

        # Luminance analysis is enough
        crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        crop_gray = cv2.GaussianBlur(crop_gray, (5, 5), 13)
        crop_gray = cv2.Canny(crop_gray, 40, 40)
        crop_gray = cv2.bitwise_not(crop_gray)

        # enlarge shapes
        # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 9))
        # crop_gray = cv2.erode(crop_gray, kernel, 2)

        # help to close contours
        crop_gray[int(0.5 * (yf - y0)), :] = 255
        crop_gray[:, int(0.5 * (xf - x0))] = 255

        # find contour with a large enough area
        for c in cv2.findContours(crop_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[1]:
            triggered = self.threshold < cv2.contourArea(c) < self.max
            print(cv2.contourArea(c))
            if triggered:
                cv2.drawContours(crop, c, -1, (255, 0, 0, 255), 1)
                if self.tex_handler is not None:
                    self.tex_handler.bind_update_texture(1, cv2.flip(crop, 0), xf - x0, yf - y0)
                self.stay_triggered = True
                self.trigger_cooldown = True
                self.wait_time = clock()
                return True, True

        if self.tex_handler is not None:
            # crop = cv2.cvtColor(crop_gray, cv2.COLOR_GRAY2RGBA)
            self.tex_handler.bind_update_texture(1, cv2.flip(crop, 0), xf - x0, yf - y0)
        return False, False


class TextureHandler:
    def __init__(self) -> void:
        """ load fixed textures and prepare all textures location in OPENGL"""
        self.texture_array = glGenTextures(4)
        molten_steel_texture = cv2.imread("./texture/molten_steel_diff.png")
        im = cv2.cvtColor(molten_steel_texture, cv2.COLOR_BGR2RGBA)

        self.bind_texture(0, None, Conf.height, Conf.width)
        self.bind_texture(1, None, Conf.width, Conf.height)
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
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, height, width,
                        GL_RGBA, GL_UNSIGNED_BYTE, texture)

    def use_texture(self, index):
        """ just bind texture for next draw"""
        glBindTexture(GL_TEXTURE_2D, self.texture_array[index])

    @staticmethod
    def next_power_of_2(x):
        return 1 if x == 0 else 2 ** (x - 1).bit_length()


class ShaderHandler:

    def __init__(self, path: str, data_names: list) -> void:
        """ Load a given vertex and fragment shader from files """
        f_shader = compileShader(get_file_content(path + ".fs"), GL_FRAGMENT_SHADER)
        v_shader = compileShader(get_file_content(path + ".vs"), GL_VERTEX_SHADER)
        self.shaderProgram = glCreateProgram()
        glAttachShader(self.shaderProgram, v_shader)
        glAttachShader(self.shaderProgram, f_shader)
        glLinkProgram(self.shaderProgram)
        self.data_location = []
        for name in data_names:
            self.data_location.append(glGetUniformLocation(self.shaderProgram, name))

    def bind(self, data: list) -> void:
        """ bind the shader and send data to it """
        glUseProgram(self.shaderProgram)
        self.update(data)

    def update(self, data: list) -> void:
        if len(data) != len(self.data_location):
            raise IndexError("Not enough data")
        for i in range(len(data)):
            if data[i] is not None:
                glUniform1fv(self.data_location[i], np.size(data[i], None), np.array(data[i]).flatten())

    @staticmethod
    def unbind() -> void:
        """ unbind shader, no shader will be used after that"""
        glUseProgram(0)
