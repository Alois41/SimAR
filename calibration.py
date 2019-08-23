from OpenGL.GL import *
from OpenGL.GLUT import *
from liquid_equation import *

from OpenGL.GLU import *
from source.configuration import Config as Conf, Globals as Glob
import sys
import os

import time

from multiprocessing import SimpleQueue, Array, freeze_support
from OpenGL.arrays import numpymodule

numpymodule.NumpyHandler.ERROR_ON_COPY = True

screen_number = 1

sys.path += ['.']
OpenGL.ERROR_ON_COPY = True


# import warnings
# warnings.simplefilter("error")


class MainProgram:
    """ Main Class of the program, initiates everything and implements OpenGL environment"""

    def __init__(self):

        # OpenGL setup
        glutInit(sys.argv)
        glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH)
        glutInitWindowSize(Conf.width, Conf.height)
        glutInitWindowPosition(screen_number * 1366 + 10, 0)  # main window dim + 1
        glutCreateWindow("Poche AR")
        glutFullScreen()
        glClearColor(0.0, 0.0, 0.0, 1.0)
        glutDisplayFunc(self.display)
        glutReshapeFunc(self.reshape)
        glutKeyboardFunc(self.keyboard)
        glutIdleFunc(self.idle)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        self.animation_clock = None

        # Memory shared variables
        self.q_activate = SimpleQueue()
        self.liquid_grid = Array(ctypes.c_double, (Conf.dim_grille[0] + 1) * Conf.dim_grille[1])
        self.liquid_im = Array(ctypes.c_double, 10 * (Conf.dim_grille[0] + 1) * 10 * Conf.dim_grille[1] * 4)

        # Liquid control Process
        self.p_liquid = Liquid(self.liquid_im, self.q_activate, self.liquid_grid)
        self.p_liquid.daemon = True
        self.p_liquid.start()

        # Main utility class
        self.augmented_reality = AugmentedReality(Conf.width, Conf.height,
                                                  self.q_activate, self.liquid_im, self.liquid_grid)

        # execute OpenGL loop forever
        glutMainLoop()

    def idle(self):
        """ Opengl routine function, called each loop iteration"""

        Glob.delta_t = clock() - Glob.t_ref
        Glob.t_ref = clock()
        if Glob.mode == 2:
            # reset mode
            Glob.mode = 0
            if clock() - self.animation_clock > 2:
                Glob.mode = 0
            pass

        else:
            # update frame from webcam
            self.augmented_reality.cam.take_frame()

            if Glob.mode == 0:
                # build mode, update bricks
                self.augmented_reality.detect_brick()

            self.augmented_reality.check_buttons()

            if Glob.mode == 1:
                # Testing structure mode
                if Conf.cooling:
                    Conf.t_chamber = Conf.temperature if (self.augmented_reality.buttonStart.is_triggered
                                                          and self.augmented_reality.buttonStart.number) > 0 \
                        else max(Conf.t_chamber - Conf.cooling_factor * Glob.delta_t, 293)
                    Glob.brick_array.update()

                else:
                    Glob.brick_array.update(not self.augmented_reality.buttonStart.is_ready()
                                            and self.augmented_reality.buttonStart.number > 0)

                lost = self.p_liquid.test_loose() or Glob.brick_array.test_loose()
                if (lost and self.augmented_reality.buttonStart.is_waiting) \
                        or self.augmented_reality.buttonReset.is_triggered:
                    print("reset")
                    if self.augmented_reality.buttonReset.is_triggered:
                        print("(from rst button)")

                    self.augmented_reality.buttonStart.is_triggered = False
                    Glob.t_ref, Glob.delta_t = clock(), 0
                    Glob.hand_text = None

                    Glob.updating = False
                    Glob.update_timer = 0

                    self.augmented_reality.buttonReset.is_triggered = False
                    self.augmented_reality.reset()

                    Glob.mode = 2
                    self.animation_clock = clock()
                    self.augmented_reality.buttonStart.is_waiting = True
                    time.sleep(1.5)
                    self.augmented_reality.buttonStart.is_waiting = True

            # update brick grid in liquid simulation
            if Glob.brick_array is not None:
                with self.liquid_grid.get_lock():  # synchronize access
                    arr = np.frombuffer(self.liquid_grid.get_obj())  # no data copying
                    arr[:Conf.dim_grille[0] * Conf.dim_grille[1]] = Glob.brick_array.get_grid().flatten()

        # tell OpenGL to redraw as soon as possible
        glutPostRedisplay()

    def display(self):
        """ Opengl drawing function, called when an update is needed"""
        # Set Projection Matrix
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluOrtho2D(0, Conf.width, 0, Conf.height)

        # Switch to Model View Matrix
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        self.augmented_reality.render()

        glutSwapBuffers()

    @staticmethod
    def reshape(w, h):
        """ OpenGl function, change windows property when window's size changes"""
        w = max(1E-4, w)
        h = max(1E-4, h)
        if h == 0:
            h = 1
        glViewport(0, 0, w, h)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        if w <= h:
            glOrtho(-Glob.nRange, Glob.nRange, -Glob.nRange * h / w, Glob.nRange * h / w, -Glob.nRange, Glob.nRange)
        else:
            glOrtho(-Glob.nRange * w / h, Glob.nRange * w / h, -Glob.nRange, Glob.nRange, -Glob.nRange, Glob.nRange)

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

    def keyboard(self, key, x, y):
        """ Opengl function, add action to keyboard input"""
        if key == b'\x1b':
            # kill program when escape is pressed
            self.p_liquid.terminate()
            self.p_liquid.join()
            os._exit(1)


if __name__ == '__main__':
    from source.augmented_reality import AugmentedReality
    freeze_support()
    try:
        Glob.debug = True
        MainProgram()
    except Exception as e:
        print(e)

