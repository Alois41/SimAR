from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GL.shaders import *
import numpy as np
from Global_tools import get_file_content

width = 1024
height = 768

dim_grille = (7, 4)

color_to_mat = \
    {  # Resistance thermique /corrosion
        "Black": (0, 0),
        "Blue": (0., 0.6),
        "Yellow": (0.5, 1),
        "Magenta": (1, 0.7),
        "Cyan": (0, 1),
        "Orange": (1, 0.2),
        "Red": (1, 0),
        "Green": (0.2, 1)
    }


class Brick:

    def __init__(self, box, color, grid):
        self.xStart, self.yStart = box[0]
        self.length, self.width = box[1]
        self.angle = box[2]
        self.T_out = 293.0
        self.T_in = 293.0
        self.r_th, self.r_cor = color_to_mat[color]

        f_shader = compileShader(get_file_content("TemperatureShader.fs"), GL_FRAGMENT_SHADER)
        v_shader = compileShader(get_file_content("TemperatureShader.vs"), GL_VERTEX_SHADER)
        self.shaderProgram = glCreateProgram()
        glAttachShader(self.shaderProgram, v_shader)
        glAttachShader(self.shaderProgram, f_shader)
        glLinkProgram(self.shaderProgram)
        self.Tinside_location = glGetUniformLocation(self.shaderProgram, "Tinside")
        self.Toutside_location = glGetUniformLocation(self.shaderProgram, "Toutside")
        self.lenght_location = glGetUniformLocation(self.shaderProgram, "lenght")

        self.color = color
        self.xIndex = min(dim_grille[0] - 1, np.rint(dim_grille[0] * (self.xStart - .5 * self.length) / width))
        self.yIndex = min(dim_grille[1] - 1, np.rint(dim_grille[1] * (self.yStart - .5 * self.width) / height))
        grid[int(self.xIndex), int(self.yIndex)] = (self.r_th, self.r_cor)
        if self.length > 1.2 * (width / dim_grille[0]) and self.xIndex < len(grid) - 1:
            grid[int(self.xIndex + 1), int(self.yIndex)] = (self.r_th, self.r_cor)
            print("%i%i grande longueur" % (self.xIndex, self.yIndex))

        if self.width > 1.2 * (height / dim_grille[1]) and self.yIndex < len(grid[0]) - 1:
            grid[int(self.xIndex), int(self.yIndex + 1)] = (self.r_th, self.r_cor)
            print("%i%i grande largeur" % (self.xIndex, self.yIndex))

    def is_almost(self, b):
        ecart = np.sqrt(np.square(self.xStart - b.xStart) + np.square(self.yStart - b.yStart)
                        + np.square(self.angle - b.angle))
        return ecart < 100

    def draw(self):

        # Draw in bottom-left
        x, y = 0.2 * self.xStart, 0.2 * (height - self.yStart)
        l, w = 0.2 * self.length, 0.2 * self.width
        angle = -self.angle

        glUseProgram(self.shaderProgram)
        glUniform1f(self.Tinside_location, self.T_in)
        glUniform1f(self.Toutside_location, self.T_out)
        glUniform1f(self.lenght_location, 10.0)
        glPushMatrix()

        glTranslatef(x, y, 0)
        glRotatef(angle, 0, 0, 1)
        glTranslatef(-.5 * l, -.5 * w, 0)
        glScalef(l, w, 1)

        glBegin(GL_QUADS)  # start drawing a rectangle
        glVertex2f(0, 0)  # bottom left point
        glVertex2f(1, 0)  # bottom right point
        glVertex2f(1, 1)  # top right point
        glVertex2f(0, 1)  # top left point
        glEnd()
        glPopMatrix()
        glutSwapBuffers()
        glUseProgram(0)

    def replace(self, brick):
        self.xStart, self.yStart = brick.xStart, brick.yStart
        self.length, self.width = brick.length, brick.width
        self.angle = brick.angle

    def add_heat(self, t):
        self.T_in = self.T_in + 1000 * t * self.r_th

    def is_inside(self, _x, _y):
        x2 = _x * np.cos(-self.angle) - _y * np.sin(-self.angle)
        y2 = _y * np.cos(-self.angle) + _x * np.sin(-self.angle)

        bx2 = self.xStart * np.cos(-self.angle) - self.yStart * np.sin(-self.angle)
        by2 = self.yStart * np.cos(-self.angle) + self.xStart * np.sin(-self.angle)

        test_x = bx2 <= x2 <= bx2 + self.length
        test_y = by2 <= y2 <= by2 + self.width

        return test_x and test_y
