from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GL.shaders import *
import numpy as np
from Global_tools import get_file_content, Param as p

color_to_mat = \
    {  # color     r_th, r_corr
        "Black": (0, 0),
        "Blue": (0, 0),
        "Yellow": (0.2, 0.1),
        "Magenta": (0.6, 0.078),
        "Cyan": (0, 1),
        "Orange": (0.1, 0.2),
        "Red": (0.6, 0),
        "Green": (0.01, 1),
    }


class Brick:

    def __init__(self, box, color, grid):
        self.xStart, self.yStart = box[0]
        self.length, self.width = box[1]
        self.angle = box[2]
        self.T_out = 293.0
        self.T_in = 293.0
        self.r_th, self.r_cor = color_to_mat[color]

        f_shader = compileShader(get_file_content("./shader/TemperatureShader.fs"), GL_FRAGMENT_SHADER)
        v_shader = compileShader(get_file_content("./shader/TemperatureShader.vs"), GL_VERTEX_SHADER)
        self.shaderProgram = glCreateProgram()
        glAttachShader(self.shaderProgram, v_shader)
        glAttachShader(self.shaderProgram, f_shader)
        glLinkProgram(self.shaderProgram)
        self.t_inside_location = glGetUniformLocation(self.shaderProgram, "Tinside")
        self.t_outside_location = glGetUniformLocation(self.shaderProgram, "Toutside")
        self.length_location = glGetUniformLocation(self.shaderProgram, "lenght")

        self.color = color
        xIndex = int((self.xStart / (p.width / p.dim_grille[0])))
        yIndex = int((self.yStart / (p.height / p.dim_grille[1])))

        self.indexes = []
        self.indexes.append([xIndex, yIndex])

        grid[int(xIndex), int(yIndex)] = (self.r_th, self.r_cor)
        if self.length > 1.2 * (p.width / p.dim_grille[0]) and xIndex < len(grid) - 1:
            grid[int(xIndex + 1), int(yIndex)] = (self.r_th, self.r_cor)
            # print("%i%i grande longueur" % (xIndex, yIndex))
            self.indexes.append([xIndex + 1, yIndex])

        if self.width > 1.2 * (p.height / p.dim_grille[1]) and yIndex < len(grid[0]) - 1:
            grid[int(xIndex), int(yIndex + 1)] = (self.r_th, self.r_cor)
            self.indexes.append([xIndex, yIndex + 1])
            # print("%i%i grande largeur" % (xIndex, yIndex))

    @staticmethod
    def get_brick(brick_array, i, j, prev_brick=None):
        b_ij = next((b for b in brick_array if [i, j] in b.indexes), None)
        if prev_brick is not None:
            if prev_brick == b_ij:
                return None
        return b_ij

    def is_almost(self, b):
        ecart = np.sqrt(np.square(self.xStart - b.xStart) + np.square(self.yStart - b.yStart)
                        + np.square(self.angle - b.angle))
        return ecart < 100 and self.color == b.color

    def draw(self):
        # Draw in bottom-left
        x, y = 0.2 * self.xStart, 0.2 * (p.height - self.yStart)
        l, w = 0.2 * self.length, 0.2 * self.width
        angle = -self.angle

        glUseProgram(self.shaderProgram)
        glUniform1f(self.t_inside_location, self.T_in)
        glUniform1f(self.t_outside_location, self.T_out)
        glUniform1f(self.length_location, 10.0)
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

    def update(self, brick_array, prev_r_th=1, t_out=0, propagate=0):
        if self.indexes[0][0] == 0:
            self.T_in += (t_out - self.T_in) * np.sqrt(self.r_th * prev_r_th) * 0.1
        else:
            self.T_in += (t_out - self.T_in) * np.sqrt(self.r_th * prev_r_th) * 0.1

        b = Brick.get_brick(brick_array, self.indexes[0][0] + 1, self.indexes[0][1])
        if b is not None and b != self:
            b.update(brick_array, self.r_th, self.T_in, self)

        b = Brick.get_brick(brick_array, self.indexes[0][0], self.indexes[0][1] + 1, self)
        if b is not None and b != self and propagate != -1:
            b.update(brick_array, self.r_th, self.T_in, propagate=1)

        b = Brick.get_brick(brick_array, self.indexes[0][0], self.indexes[0][1] - 1, self)
        if b is not None and b != self and propagate != 1:
            b.update(brick_array, self.r_th, self.T_in, propagate=-1)
