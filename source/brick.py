from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GL.shaders import *
import numpy as np
from Global_tools import get_file_content, Param as p


class Brick:

    def __init__(self, box, color, grid):
        self.geometry = BrickGeometry(box)
        self.indexes = self.geometry.compute_indexes()
        self.material = BrickMaterial(color)
        self.is_invalid = False

        for index in self.indexes:
            grid[int(min(p.dim_grille[0], index[0])), int(min(p.dim_grille[1], index[1]))] = \
                (self.material.r_th, self.material.r_cor)

    @staticmethod
    def get_brick(brick_array, i, j, prev_brick=None):
        b_ij = next((b for b in brick_array if [i, j] in b.indexes), None)
        if prev_brick is not None:
            if prev_brick == b_ij:
                return None
        return b_ij

    def is_almost(self, b):
        return self.geometry.compare(b.geometry) and self.material.color == b.material.color

    def replace(self, brick, grid):
        self.is_invalid = False
        self.material.is_broken = False

    def invalidate(self):
        self.is_invalid = True

    def update(self, t_contact, prev_r_th, temp: list = None) -> void:
        self.material.update(t_contact, t_contact, temp)
        if self.material.temperature > 500:
            self.material.is_broken = True


class BrickRenderer:
    def __init__(self):
        f_shader = compileShader(get_file_content("./shader/TemperatureShader.fs"), GL_FRAGMENT_SHADER)
        v_shader = compileShader(get_file_content("./shader/TemperatureShader.vs"), GL_VERTEX_SHADER)
        self.shaderProgram = glCreateProgram()
        glAttachShader(self.shaderProgram, v_shader)
        glAttachShader(self.shaderProgram, f_shader)
        glLinkProgram(self.shaderProgram)
        self.t_inside_location = glGetUniformLocation(self.shaderProgram, "Tinside")
        self.t_outside_location = glGetUniformLocation(self.shaderProgram, "Toutside")
        self.length_location = glGetUniformLocation(self.shaderProgram, "lenght")

    def render(self, b: Brick) -> void:
        # Draw in bottom-left
        x, y = 0.2 * b.geometry.xStart, 0.2 * (p.height - b.geometry.yStart)
        l, w = 0.2 * b.geometry.length, 0.2 * b.geometry.width
        angle = -b.geometry.angle

        glUseProgram(self.shaderProgram)
        glUniform1f(self.t_inside_location, b.material.temperature)
        glUniform1f(self.t_outside_location, b.material.T_out)
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


class BrickGeometry:
    def __init__(self, box):

        if box[2] > -45:
            self.xStart, self.yStart = box[0]
            self.length, self.width = box[1]
            self.angle = box[2]
        elif box[2] <= -45:
            self.xStart, self.yStart = box[0]
            self.width, self.length = box[1]
            self.angle = box[2] + 90

    def compute_indexes(self):
        indexes = []
        x_index = int((self.xStart / (p.width / p.dim_grille[0])))
        y_index = int((self.yStart / (p.height / p.dim_grille[1])))
        indexes.append([x_index, y_index])

        if self.length > 1.2 * (p.width / p.dim_grille[0]) and x_index < p.dim_grille[0] - 1:
            # print("%i%i grande longueur %0.2f" % (x_index, y_index, self.angle))
            indexes.append([x_index + 1, y_index])

        if self.width > 1.2 * (p.height / p.dim_grille[1]) and y_index < p.dim_grille[1] - 1:
            indexes.append([x_index, y_index + 1])
            # print("%i%i grande largeur %0.2f" % (x_index, y_index, self.angle))

        return indexes

    def compare(self, b):
        value = np.sqrt(np.square(self.xStart - b.xStart) + np.square(self.yStart - b.yStart)
                        + np.square(self.angle - b.angle))
        return value < 100


class BrickMaterial:
    def __init__(self, color):
        self.color = color
        self.temperature = 25
        self.r_th, self.r_cor = p.color_to_mat[color]
        self.is_broken = False
        self.N = 5
        self.laplacien = np.zeros(self.N)
        self.T = np.zeros(self.N)

    # noinspection PyUnreachableCode
    def update(self, t_contact, prev_rth, temp: list = None):
        r_th = 1 if self.is_broken else self.r_th
        diff = (t_contact - self.temperature) * r_th * prev_rth * 1E-4
        self.temperature += diff


        return
        # tentative d'eq de la chaleur ( 1D ), revoir les coefficients
        x = np.linspace(0, 1, self.N)
        dx = x[1] - x[0]
        dx2 = dx ** 2

        if temp is not None and temp[0] is not None and temp[1] is not None:
            self.T[0] = temp[0].material.temperature
            self.T[self.N - 1] = temp[1].material.temperature

            for k in range(1, self.N - 1):
                self.laplacien[k] = (self.T[k + 1] - 2 * self.T[k] + self.T[k - 1]) / dx2
            for k in range(1, self.N - 1):
                self.T[k] += 3E-5 * self.r_th * self.laplacien[k]
            self.temperature = self.T[2]


class BrickArray:
    def __init__(self, bricks):
        self.array = np.array([[None] * p.dim_grille[1]] * p.dim_grille[0])
        self.update_array = np.array([[0] * p.dim_grille[1]] * p.dim_grille[0])
        for i in range(p.dim_grille[0]):
            for j in range(p.dim_grille[1]):
                self.array[i][j] = Brick.get_brick(bricks, i, j)

    def get(self, i: int, j: int) -> Brick:
        return self.array[i][j] if 0 <= i < p.dim_grille[0] and 0 <= j < p.dim_grille[1] else None

    def set(self, i: int, j: int, value: Brick or None) -> void:
        self.array[i][j] = value

    def clear(self) -> void:
        self.array = np.array([[None] * p.dim_grille[1]] * p.dim_grille[0])

    def invalidate(self) -> void:
        for column in self.array:
            for brick in column:
                if brick is not None:
                    brick.invalidate()

    def clear_invalid(self) -> void:
        for column in self.array:
            for brick in column:
                if brick is not None and brick.is_invalid:
                    for index in brick.indexes:
                        print("Brick removed: " + str(index) + brick.material.color)
                        self.set(index[0], index[1], None)

    def update(self, t_liquid, heating = True) -> void:

        # update heat for brick in contact to the liquid
        if heating:
            for brick in self.array[0]:
                if brick is not None:
                    brick.update(t_liquid, 1)

        # update heat for brick in contact to the liquid
        for column in self.array:
            for brick in column:
                if brick is not None:
                    for index in brick.indexes:
                        x, y = index[0], index[1]
                        temp = self.get(x - 1, y), self.get(x + 1, y), self.get(x, y + 1), self.get(x, y - 1)
                        for b in temp:
                            if b is not None and b != brick:
                                brick.update(b.material.temperature, b.material.r_th, temp)
                        pass

            pass

    def reset(self) -> void:
        for column in self.array:
            for brick in column:
                if brick is not None:
                    brick.material.T_in = 293
