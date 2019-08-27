from threading import Thread
from time import sleep, clock
from image_recognition.drawing import *
from image_recognition.image_tools import *
from settings.configuration import Config as Conf


class HandButton(Thread):

    def __init__(self, wait_time: float, texture_handler, texloc, hand_area: list, threshold: int) -> void:
        super(HandButton, self).__init__()
        """ Init of one button """
        self.wait_time = wait_time
        self.active = True

        self.remaining_time = wait_time

        self.title = ""

        self.tex_handler = texture_handler
        self.tex_loc = texloc

        self.number = 0

        self.hand_area = hand_area
        self.threshold = threshold
        self.clock = clock()

        self.is_waiting = True
        self.is_triggered = False

        self.frame = None

        self.background = None
        self.crop = None

        y0, x0 = self.hand_area[0]
        yf, xf = self.hand_area[1]
        if self.tex_handler is not None:
            self.tex_handler.bind_texture(self.tex_loc, None, xf - x0, yf - y0)

        self.image = None

    def run(self):
        while True:
            sleep(1)
            if self.image is not None and self.active:
                self.check(self.image)

    def check(self, image: np.ndarray):
        y0, x0 = self.hand_area[0]
        yf, xf = self.hand_area[1]

        if not self.is_triggered:
            self.crop = cv2.cvtColor(crop_zone(image, x0, y0, xf, yf), cv2.COLOR_BGR2RGBA)
            self.crop = cv2.addWeighted(self.crop, 1, self.crop, 0, -20)

        if self.is_triggered:
            self.remaining_time = self.wait_time - (clock() - self.clock)
            if clock() - self.clock > self.wait_time:
                self.is_triggered = False
                self.is_waiting = True
                self.clock = clock()

        elif self.is_waiting:
            # self.remaining_time = self.wait_time + self.wait_time / 4 - (clock() - self.clock)
            if (clock() - self.clock) > max(2, self.wait_time / 4):
                self.is_waiting = False
        else:
            self.is_triggered = self.detect_hand()
            if self.is_triggered:
                self.number += 1
                self.remaining_time = self.wait_time
            self.clock = clock()

    def detect_hand(self) -> bool:
        """ Detect hand in the button area """
        # crop to the button zone
        crop = self.crop
        crop[:, :, 3] = 255.0 * np.ones((crop.shape[0], crop.shape[1]))

        # Luminance analysis is enough
        crop_gray = cv2.cvtColor(crop.copy(), cv2.COLOR_RGBA2GRAY)

        crop_gray = cv2.GaussianBlur(crop_gray, (5, 5), 13)
        crop_gray = cv2.Canny(crop_gray, 40, 40)

        # enlarge shapes
        _kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        crop_gray = cv2.morphologyEx(crop_gray, cv2.MORPH_CLOSE, _kernel)

        triggered = False

        # find contour with a large enough area
        for c in cv2.findContours(crop_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[1]:
            # print(cv2.contourArea(c))
            if cv2.contourArea(c) > self.threshold:
                print(cv2.contourArea(c))
                triggered = True
            cv2.fillPoly(crop, pts=[c], color=(255, 0, 0, 255))

        # if self.tex_handler is not None:
        #     self.tex_handler.bind_update_texture(self.tex_loc, cv2.flip(crop, 0), xf - x0, yf - y0)

        return triggered

    def draw(self):
        y0, x0 = self.hand_area[0]
        yf, xf = self.hand_area[1]
        if self.is_triggered:
            draw_rectangle(x0, Conf.height - yf, xf - x0, yf - y0, 0, 1, 0)
        elif self.is_waiting:
            draw_rectangle(x0, Conf.height - yf, xf - x0, yf - y0, 1, 0, 0)
        else:
            draw_rectangle(x0, Conf.height - yf, xf - x0, yf - y0, 1, 1, 1)

        glut_print(x0 + .5 * (xf - x0) - 6 * len(self.title),
                   Conf.height - y0 + 10, GLUT_BITMAP_HELVETICA_18, self.title, *Conf.text_color)

        if self.tex_handler is not None:
            glEnable(GL_TEXTURE_2D)
            self.tex_handler.use_texture(self.tex_loc)
            draw_textured_rectangle(x0, Conf.height - yf - (yf - y0), xf - x0, yf - y0)
            glDisable(GL_TEXTURE_2D)

        pass

    def is_ready(self):
        return not (self.is_triggered or self.is_waiting)

    def pause(self):
        self.active = False
        self.is_triggered = False

    def unpause(self):
        self.active = True