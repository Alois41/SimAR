import time
from source.brick import Brick, BrickArray
from time import clock
from source.configuration import Config as Conf, Globals as Glob
from source.image_tools import *
from source.drawing import *
from source.hand_button import HandButton
from OpenGL.GL import *
from source.resources import Resources
from multiprocessing import SimpleQueue
import random

# strings ressources for futur languages
strings = Resources.strings['fr']


class AugmentedReality:
    """ Take and process webcam frames"""

    def __init__(self, width: int, height: int, q_activate: SimpleQueue, liquid_im, liquid_grid) -> void:

        # Attributes from parameters
        self.width, self.liquid_height = width, height
        self.cam = Camera(width, height)
        self.q_activate = q_activate
        self.liquid_im = liquid_im
        self.liquid_grid = liquid_grid

        # Other attributes
        self.triggered_start, self.triggered_reset = False, False
        self.triggered_number = 0
        self.wait_time = 1
        self.liquid_height = 0.0
        self.clock_liquid = 0.0

        # Create a texture handler with 5 different textures
        self.tex_handler = TextureHandler(5)

        # Create button Handler and start them for image detection
        self.buttonStart = HandButton(10, self.tex_handler, 3, Conf.hand_area_1, Conf.hand_threshold_1)
        self.buttonReset = HandButton(1, self.tex_handler, 1, Conf.hand_area_2, Conf.hand_threshold_2)
        self.buttonStart.daemon = True
        self.buttonReset.daemon = True
        self.buttonStart.start()
        self.buttonReset.start()
        self.buttonReset.title = "RETOUR"

        # Create a handler for every drawing functions
        self.draw_handler = DrawingHandler(self.tex_handler, q_activate, liquid_im)

        # Create a class that will extract "bricks" from image
        self.brick_recognition = BrickRecognition(liquid_im)

    def reset(self):
        """ Reset program state"""

        self.triggered_start, self.triggered_reset = False, False
        self.triggered_number = 0
        self.wait_time = 1

        self.draw_handler = DrawingHandler(self.tex_handler, self.q_activate, self.liquid_im)
        self.brick_recognition = BrickRecognition(self.liquid_im)
        self.liquid_height = 0.0
        self.clock_liquid = 0.0

        self.buttonReset.is_triggered, self.buttonReset.is_waiting = False, False
        Glob.brick_array.reset()

    def render(self) -> void:
        """ render the scene with OpenGL"""

        if Glob.frame is not None:

            # Set button text
            if Glob.mode == 0:
                self.buttonStart.title = "VALIDER"
            elif self.buttonStart.is_ready():
                self.buttonStart.title = "CONTINUER"
            else:
                self.buttonStart.title = "%i" % self.buttonStart.remaining_time

            # Set liquid state from buttons
            poor_liquid = False
            if self.buttonStart.is_triggered and self.triggered_number > 0:
                poor_liquid = True

            # step 1 : draw background
            self.draw_handler.draw_frame(Glob.frame)

            y0, x0 = Conf.cam_area[0]
            yf, xf = Conf.cam_area[1]
            # step 2 : draw molten steel
            self.draw_handler.draw_molten_steel(x0, y0, xf - x0, yf - y0, 1, 1, 1, poor_liquid)

            # step 3 : draw user interface
            self.draw_handler.draw_ui(self.buttonStart, self.buttonStart.number)

            # step 4 : draw buttons interfaces, reset button depends on the mode
            self.buttonStart.draw()
            if Glob.mode == 1 and self.buttonStart.is_ready():
                self.buttonReset.unpause()
                self.buttonReset.draw()
            else:
                self.buttonReset.pause()

    def check_buttons(self) -> void:
        """ Update button image and read button state """

        # Set image to the newest one
        self.buttonStart.image = self.cam.image_raw
        self.buttonReset.image = self.cam.image_raw

        # Change mode with button state
        if Glob.mode == 0 and Glob.brick_array is not None:
            self.triggered_number = self.buttonStart.number

            if self.buttonStart.is_triggered:
                Glob.mode = 1
                Glob.brick_array.init_heat_eq()
            pass

            if self.buttonReset.is_triggered:
                Glob.mode = 0
                return
            pass

    def detect_brick(self):
        """ Execute brick detection tools """
        image = self.brick_recognition.update_bricks(self.cam.image_raw.copy())

        # if we are calibrating print brick map on the screen (upper left corner)
        if image is not None:
            if Glob.debug:
                texture = cv2.resize(image, (Conf.width, Conf.height))
                self.tex_handler.bind_texture(0, cv2.flip(texture, 0), Conf.width, Conf.height)
            Glob.frame = image

    def lost_screen(self):
        """ Draw a message on the screen """
        self.draw_handler.draw_text_screen()


class BrickRecognition:
    """ Detect bricks from webcam frame """

    def __init__(self, liquid_im):
        # keep dynamic array in class
        self.liquid_im = liquid_im
        self.frame_count = 0

    def update_bricks(self, image_raw) -> (np.ndarray, list):
        """ Process the current raw frame and return a texture to draw and the  detected bricks"""
        image = None
        self.frame_count += 1
        # update each n frames
        if Glob.mode == 0 and self.frame_count % Conf.refresh_rate == 0:  # building mode
            image_raw = cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGBA)  # convert to RGBA
            # image_raw = adjust_gamma(image_raw.copy(), 1 if calibration else 2)  # enhance gamma of the image

            if False:  # futur work : no grid mode
                bricks, image = self.discretise_area(image_raw)
            else:
                # Set almost white to white
                image_raw[np.std(image_raw[:, :, :3], axis=2) < 12.0] = [255, 255, 255, 255]
                contours, image = find_center_contours(image_raw)  # find contours in the center of the image
                bricks = self.isolate_bricks(contours, image)  # detect bricks

            # Make image opaque
            image[:, :, 3] = 255 * np.ones((image.shape[0], image.shape[1]))

            # Initialise brick array if needed
            if Glob.brick_array is None or len(Glob.brick_array.array) == 0:
                Glob.brick_array = BrickArray(bricks, self.liquid_im)

            else:
                # Update array with new bricks if needed
                Glob.brick_array.invalidate()
                for nb in bricks:
                    for index in nb.indexes:
                        if index == [0, 0]:
                            Glob.brick_array.set(0, 0, Brick.void(nb.indexes))
                            continue

                        prev_b = Glob.brick_array.get(index[0], index[1])
                        if prev_b is not None and prev_b.is_almost(nb):
                            prev_b.replace()

                        else:
                            # print("Brick added: " + str(nb.indexes[0]) + " (%s)" % nb.material.color)
                            Glob.brick_array.set(index[0], index[1], nb)

                Glob.brick_array.clear_invalid()

        return image

    @staticmethod
    def discretise_area(image):
        """ not implemented yet, no grid mode """

        raise NotImplementedError

        # blurred = cv2.GaussianBlur(image, (5, 5), 0)
        # thresh_rgb = blurred.copy()
        #
        # # for each color channel
        # for index, channel in enumerate(cv2.split(thresh_rgb)):
        #     # make channel binary with an adaptive threshold
        #     thresh = cv2.adaptiveThreshold(channel, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 103, 8)
        #     # Structure the channel with Rectangle shapes
        #     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
        #     thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        #
        #     # merge channels
        #     thresh_rgb[:, :, index] = thresh
        #
        # # convert into Gray scale(luminance)
        # thresh_gray = cv2.cvtColor(thresh_rgb, cv2.COLOR_RGB2GRAY)
        # # zoom in the center
        # thresh_gray = zoom_center(thresh_gray)
        # image = zoom_center(image)
        # # invert black/white
        # thresh_gray = cv2.bitwise_not(thresh_gray)
        #
        # kernel = np.ones((5, 5), np.uint8)
        # thresh_gray = cv2.erode(thresh_gray, kernel, iterations=10)
        #
        # bricks = []
        #
        # step_x = image.shape[1] / Conf.dim_grille[0]
        # step_y = image.shape[0] / Conf.dim_grille[1]
        #
        # # import matplotlib.pyplot as plt
        # # plt.subplots(Conf.dim_grille[1], Conf.dim_grille[0])
        #
        # image_hls = cv2.cvtColor(image.copy(), cv2.COLOR_RGB2HLS)
        #
        # for x in range(Conf.dim_grille[0]):
        #     for y in range(Conf.dim_grille[1]):
        #         crop = image_hls[int(y * step_y):int((y + 1) * step_y),
        #                int(x * step_x):int((x + 1) * step_x)]
        #         mean = 2 * np.mean(crop, axis=(0, 1))[0]
        #
        #         if mean > 20:
        #             colors = list(Conf.color_dict.keys())
        #             closest_colors = sorted(colors, key=lambda color: np.abs(color - mean))
        #             closest_color = closest_colors[0]
        #
        #             b = Brick.new([[x * step_x, y * step_y], [step_x, step_y], 0], closest_color)
        #             bricks.append(b)
        #
        #             c_rgb = cv2.cvtColor(np.uint8([[[0.5 * closest_color, 128, 255]]]), cv2.COLOR_HLS2RGB)[0][0]
        #             image = cv2.rectangle(image,
        #                                   (int(x * step_x), int(y * step_y)),
        #                                   (int((x + 1) * step_x), int((y + 1) * step_y)),
        #                                   tuple([int(n) for n in c_rgb]), thickness=10)
        #             # cv2.putText(image, "%.2f" % mean, (int(x * step_x), int((y + .5) * step_y)),
        #             #             cv2.FONT_HERSHEY_SIMPLEX,
        #             #             1.5, (0, 0, 0), thickness=10, lineType=10)
        #
        # # plt.show()
        #
        # return bricks, image

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
            # name_color = Conf.color_dict[closest_color]

            # draw a bright color with the same hue
            c_rgb = cv2.cvtColor(np.uint8([[[0.5 * closest_color, 128, 255]]]), cv2.COLOR_HLS2RGB)[0][0]
            cv2.drawContours(image, [box], 0, tuple([int(n) for n in c_rgb]), thickness=cv2.FILLED)

            # create a brick and add it to the array
            b = Brick.new(rect, closest_color)
            cv2.putText(image, "%0.0f" % mean, (c_x - 20, c_y - 20), cv2.FONT_HERSHEY_SIMPLEX,
                        1.5, (0, 0, 0), thickness=10, lineType=10)
            cv2.putText(image, "%i%i" % (tuple(b.indexes[0])), (c_x - 20, c_y + 40), cv2.FONT_HERSHEY_SIMPLEX,
                        2, (0, 0, 0), thickness=10, lineType=10)

            bricks.append(b)

        return bricks


class DrawingHandler:
    """ Handle all drawing with OpenGL
        control shaders and textures
    """

    def __init__(self, texture_handler, q_activate: SimpleQueue, liquid_image):
        self.q_activate = q_activate
        with liquid_image.get_lock():
            self.liquid_image = np.frombuffer(liquid_image.get_obj())
        self.tex_handler = texture_handler
        self.offset_x, self.offset_y = 0, 0
        self.shader_handler_molten_steel = ShaderHandler("./shader/LavaShader",
                                                         {"myTexture": None, "offset_x": float, "offset_y": float,
                                                          "delta_t": float, "height": float})
        self.shader_handler_brick = ShaderHandler("./shader/TemperatureShader",
                                                  {"Corrosion": float, "temp_buffer": GL_SHADER_STORAGE_BUFFER,
                                                   "brick_dim": np.ndarray, "grid_pos": int, "step": float,
                                                   "border": float})

        self.steel_flow_dir_x, self.steel_flow_dir_y = 1, 1
        self.shader_clock = clock()
        self.q = 0
        self.previous_number = 0

        k_factor = 3
        self.kernel = np.ones((k_factor, k_factor), np.float32) / (k_factor * k_factor)

    def draw_molten_steel(self, x_s: int, y_s: int, w: int, h: int,
                          r: float, g: float, b: float, active) -> void:
        """ draw "molten_steel" from a texture"""

        draw_rectangle(x_s - 20, y_s - 10, w + 20, h + 10, 0.1, 0.1, 0.1)
        if Glob.mode == 0:
            draw_rectangle(x_s, y_s, w, h, 1, 1, 1)

        # load shader
        delta_t = clock() - self.shader_clock
        # if random.random() > .999:
        #     self.steel_flow_dir_x *= -1
        # if random.random() > .999:
        #     self.steel_flow_dir_y *= -1

        self.offset_x += self.steel_flow_dir_x * random.random() * delta_t * 0.005
        self.offset_y += self.steel_flow_dir_y * random.random() * delta_t * 0.005
        self.shader_clock = clock()

        self.shader_handler_molten_steel.bind(data={"myTexture": None, "offset_x": self.offset_x,
                                                    "offset_y": self.offset_y, "delta_t": delta_t, "height": 0})
        # update offset from clock to scroll the texture
        glEnable(GL_TEXTURE_2D)
        self.tex_handler.use_texture(2)
        glColor(r, g, b, 1)

        if self.q_activate.empty():
            self.q_activate.put(active)

        if Glob.mode == 1:
            image = np.reshape(self.liquid_image.copy(), (10 * Conf.dim_grille[1], 10 * (Conf.dim_grille[0] + 1), 4))

            image[:, :, :] = cv2.filter2D(image[:, :, :], -1, self.kernel)
            image = cv2.flip(image, 0)

            # not needed, opengl can do it by itself
            # image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
            # glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, image.shape[1], image.shape[0],
            #                 GL_RGBA, GL_FLOAT, cv2.flip(image.astype(np.float32), 0))

            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, image.shape[1], image.shape[0],
                         0, GL_RGBA, GL_FLOAT, cv2.flip(image.astype(np.float32), 0))
            # draw_textured_rectangle(0, 0, image.shape[1], image.shape[0])
            draw_textured_rectangle(x_s, y_s, (Conf.dim_grille[0] + 1) * w / Conf.dim_grille[0], h)

        glDisable(GL_TEXTURE_2D)

        self.shader_handler_molten_steel.unbind()

    @staticmethod
    def draw_thermal_diffusivity(x_s: int, y_s: int) -> void:
        glut_print(0, 100, GLUT_BITMAP_HELVETICA_18, "Resistance thermique", *Conf.text_color)
        y0, x0 = Conf.cam_area[0]
        yf, xf = Conf.cam_area[1]
        step_x = (xf - x0) / Conf.dim_grille[0]
        step_y = (y0 - 10) / Conf.dim_grille[1]
        for c in range(Conf.dim_grille[0]):
            for l in range(Conf.dim_grille[1]):
                b_xy = Glob.brick_array.get(c, l)
                t = b_xy.material.diffusivity * 1E6
                if not b_xy.is_void:
                    color = b_xy.material.color
                    draw_rectangle(x_s + c * step_x, (Conf.dim_grille[1] - l - 1) * step_y, step_x, step_y, *color)
                else:
                    draw_rectangle(x_s + c * step_x, (Conf.dim_grille[1] - l - 1) * step_y, step_x, step_y, 0, 0, 0)

                if b_xy is not None:
                    if not b_xy.is_void:
                        dict_values = {0: "+ + +", 1.2: "+ +", 2.4: "+", 3.6: "-", 4.8: "- -", 6: "- - -"}
                        tmp = list(dict_values.keys())
                        closest_value = sorted(tmp, key=lambda v: np.abs(v - t))
                        txt = dict_values[closest_value[0]]
                        glut_print(x_s + c * step_x + .5 * step_x - 2.5 * len(txt),
                                   (Conf.dim_grille[1] - l - 1) * step_y + .3 * step_y, GLUT_BITMAP_HELVETICA_18, txt, 1, 1, 1)

    @staticmethod
    def draw_resistance_corr(x_s: int, y_s: int) -> void:
        glut_print(0, 100, GLUT_BITMAP_HELVETICA_18, "Resistances à la", *Conf.text_color)
        glut_print(0, 80, GLUT_BITMAP_HELVETICA_18, "corrosion", *Conf.text_color)
        y0, x0 = Conf.cam_area[0]
        yf, xf = Conf.cam_area[1]
        step_x = (xf - x0) / Conf.dim_grille[0]
        step_y = (y0 - 10) / Conf.dim_grille[1]
        for c in range(Conf.dim_grille[0]):
            for l in range(Conf.dim_grille[1]):
                b_xy = Glob.brick_array.get(c, l)
                t = b_xy.material.r_cor if b_xy is not None else np.nan
                if not np.isnan(t) and not b_xy.is_void:
                    color = b_xy.material.color
                    draw_rectangle(x_s + c * step_x, (Conf.dim_grille[1] - l - 1) * step_y, step_x, step_y, *color)
                else:
                    draw_rectangle(x_s + c * step_x, (Conf.dim_grille[1] - l - 1) * step_y, step_x, step_y, 0, 0, 0)

                if b_xy is not None and not b_xy.drowned and not b_xy.is_void:
                    if b_xy.material.r_cor != -1:
                        dict_values = {0: "- - -", 0.2: "- -", 0.4: "-", 0.6: "+", 0.8: "+ +", 1: "+ + +"}
                        tmp = list(dict_values.keys())
                        closest_value = sorted(tmp, key=lambda v: np.abs(v - b_xy.material.r_cor))
                        txt = dict_values[closest_value[0]]

                        # txt = "%0.0f%%" % (100.0 * b_xy.material.r_cor)

                        glut_print(x_s + c * step_x + .5 * step_x - 2.5 * len(txt),
                                   (Conf.dim_grille[1] - l - 1) * step_y + .3 * step_y, GLUT_BITMAP_HELVETICA_18, txt, 1, 1, 1)

    def draw_temperatures(self, x_s: int, y_s: int, bricks) -> void:
        glut_print(0, 100, GLUT_BITMAP_HELVETICA_18, "Temperatures", *Conf.text_color)

        _range = 2 if Glob.mode == 1 else 1
        for i in range(_range):
            y0, x0 = Conf.cam_area[0]
            yf, xf = Conf.cam_area[1]
            h = 20
            a = 1

            if i == 1:
                w, h = xf - x0, (yf - y0) / Conf.dim_grille[1]
                x_s, y_s = x0, y0
                a = 0

            # request temperature buffer update
            self.shader_handler_brick.invalidate_buffer()

            step = int((xf - x0) / Conf.dim_grille[0])
            for index_c in range(Conf.dim_grille[0]):
                for index_l in range(Conf.dim_grille[1]):
                    b_xy: Brick = bricks.get(index_c, index_l)
                    if not b_xy.is_void and not b_xy.material.is_broken or i == 0:
                        index = b_xy.indexes[0]
                        temp_array = np.array(Glob.brick_array.T, dtype=ctypes.c_int32)
                        pos = (index[1]) * Glob.brick_array.step_x * (temp_array.shape[1]) + index[
                            0] * Glob.brick_array.step_y

                        border = 0
                        if index[0] < Conf.dim_grille[0] - 1:
                            border += 1
                        if index[1] < Conf.dim_grille[1] - 1:
                            border += 2

                        self.shader_handler_brick.bind({"Corrosion": b_xy.material.health if i == 1 else 1,
                                                        "temp_buffer": temp_array.flatten(),
                                                        "brick_dim": [Glob.brick_array.step_x, Glob.brick_array.step_y],
                                                        "grid_pos": pos, "step": Glob.brick_array.nx,
                                                        "border": border})

                        draw_rectangle(x_s + index_c * step, y_s + (Conf.dim_grille[1] - index_l - 1) * h, step, h)

                        self.shader_handler_brick.unbind()

                        # do not update buffer in the next iterations
                        self.shader_handler_brick.fix_buffer()

                        message = "%0.0f °C" % np.mean(Glob.brick_array.get_temp(index[0], index[1]) - 273)
                        # glut_print(x_s + index_c * step + .5 * step - 2.5 * len(message),
                        #            y_s + (Conf.dim_grille[1] - index_l - 1) * h + .5 * h - 5,
                        #            GLUT_BITMAP_HELVETICA_12,
                        #            message, 1, 1, 1)

                    else:
                        draw_rectangle(x_s + index_c * step, y_s + (Conf.dim_grille[1] - index_l - 1) * h,
                                       step, h, 0, 0, 0, a if b_xy.drowned else 1)

                    # draw_rectangle_empty(x_s + index_c * step, y_s + (Conf.dim_grille[1] - index_l - 1) * h,
                    #                      step, h, 0, 0, 0, 1)

    def draw_texture(self, tex_loc: int, _x: int, _y: int, l: int, h: int) -> void:
        glEnable(GL_TEXTURE_2D)
        self.tex_handler.use_texture(tex_loc)
        draw_textured_rectangle(_x, _y, l, h)
        glDisable(GL_TEXTURE_2D)

    @staticmethod
    def draw_frame(image: np.ndarray) -> void:
        """ Draw the frame with OpenGL as a texture in the entire screen"""
        if type(image) != int:
            # draw background
            draw_rectangle(0, 0, Conf.width, Conf.height, 0.2, 0.2, 0.2)

            y0, x0 = Conf.cam_area[0]
            yf, xf = Conf.cam_area[1]

            if Glob.mode == 1:
                draw_rectangle(x0, y0, xf - x0, yf - y0, 0, 0, 0)

    def draw_ui(self, start_button, number) -> void:
        """ Draw user interface and decoration"""

        if number > self.previous_number:
            self.previous_number = number
            self.q += Glob.brick_array.get_capacity()

        # draw board informations
        x_start = 2.5 * Conf.width / 10

        title = strings['title_build']
        subtitle = strings['sub_title_build']
        if Glob.brick_array is not None:
            if Glob.mode == 0:
                if clock() % (2 * Conf.swap_time) <= Conf.swap_time:
                    self.draw_thermal_diffusivity(Conf.cam_area[0][1], 0)
                else:
                    self.draw_resistance_corr(Conf.cam_area[0][1], 0)

            else:
                self.draw_temperatures(Conf.cam_area[0][1], 0, Glob.brick_array)
                if start_button.is_ready():
                    title = strings['title_test']
                    subtitle = strings['sub_title_test1i'] % number
                    subtitle += "   Quantité d'acier: %i tonne%s" % (self.q, "s" if self.q > 1 else "")
                elif start_button.is_triggered:
                    title = "Coulée en cours"
                    subtitle = ""
                else:
                    title = "Evacuation de l'acier liquide"
                    subtitle = ""

        # glut_print(x_start, Conf.height - 30, GLUT_BITMAP_HELVETICA_18,
        #            title + "    %0.0f fps" % (1 / Glob.delta_t), *Conf.text_color)

        glut_print(x_start, Conf.height - 30, GLUT_BITMAP_HELVETICA_18, title, *Conf.text_color)
        glut_print(x_start, Conf.height - 60, GLUT_BITMAP_HELVETICA_18, subtitle, *Conf.text_color)

        y0, x0 = Conf.cam_area[0]
        yf, xf = Conf.cam_area[1]

        # brick map
        if Glob.debug:
            ratio = .2, .2
            self.draw_texture(0, 0, (1 - ratio[1]) * Conf.height, ratio[0] * Conf.width, ratio[1] * Conf.height)

        # draw grid
        if Glob.mode == 0:
            step_i = (xf - x0) / Conf.dim_grille[0]
            step_j = (yf - y0) / Conf.dim_grille[1]
            for i in range(Conf.dim_grille[0]):
                for j in range(Conf.dim_grille[1]):
                    pass
                    draw_rectangle_empty(x0 + i * step_i, y0 + j * step_j, step_i, step_j, 0.2, 0.2, 0.2, 2)

    def draw_text_screen(self):

        texture = np.zeros((Conf.height, Conf.width, 4), np.uint8)
        texture[..., 3] = 128

        text_1 = "Structure defaillante"
        text_2 = "Quantite d'acier : %i tonne%s" % (self.q, 's' if self.q > 0 else '')
        scale = 2
        thickness = 5

        textsize1 = cv2.getTextSize(text_1, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)[0]
        textsize2 = cv2.getTextSize(text_2, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)[0]

        textX_1 = (texture.shape[1] - textsize1[0]) // 2
        textY_1 = (texture.shape[0] + textsize1[1]) // 2
        texture = cv2.putText(texture, text_1, (textX_1, textY_1),
                              cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255, 255), thickness=thickness, lineType=10)

        textX_2 = (texture.shape[1] - textsize2[0]) // 2
        textY_2 = (texture.shape[0] + textsize2[1]) // 2 + 2*textsize1[1]
        texture = cv2.putText(texture, text_2, (textX_2, textY_2),
                              cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255, 255), thickness=thickness, lineType=10)

        glEnable(GL_TEXTURE_2D)
        self.tex_handler.bind_texture(4, cv2.flip(texture, 0), Conf.width, Conf.height)
        self.tex_handler.use_texture(4)
        self.draw_texture(4, 0, 0, Conf.width, Conf.height)


class Camera:
    """ Control webcam and send webcam image"""

    def __init__(self, width: int, height: int) -> None:
        self.capture = cv2.VideoCapture(Conf.cam_number)
        if not self.capture.isOpened():
            raise IOError("Webcam not plugged or wrong webcam index")
        time.sleep(.5)
        self.capture.set(3, width)
        self.capture.set(4, height)
        # self.capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, .25)
        # self.capture.set(cv2.CAP_PROP_EXPOSURE, -2)
        # self.capture.set(cv2.CAP_PROP_BRIGHTNESS, -10)
        for i in range(10):
            _, self.image_raw = self.capture.read()

    def take_frame(self) -> void:
        """ Update current raw frame in BGR format"""
        _, self.image_raw = self.capture.read()
