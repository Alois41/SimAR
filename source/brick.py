from OpenGL.GLUT import *
import numpy as np
import cv2
from source.Global_tools import Config as Conf
from source.heatEquation import HeatEquation


class Brick:

    def __init__(self, box, color):
        self.geometry = BrickGeometry(box)
        self.indexes = self.geometry.compute_indexes()
        self.material = BrickMaterial(color)
        self.is_invalid = False

        # for index in self.indexes:
        #     grid[int(min(Conf.dim_grille[0], index[0])), int(min(Conf.dim_grille[1], index[1]))] = \
        #         (self.material.r_th, self.material.r_cor)

    @staticmethod
    def get_brick(brick_array, i, j, prev_brick=None):
        b_ij = next((b for b in brick_array if [i, j] in b.indexes), None)
        if prev_brick is not None:
            if prev_brick == b_ij:
                return None
        return b_ij

    def is_almost(self, b):
        return self.geometry.compare(b.geometry) and self.material.color_name == b.material.color

    def replace(self, brick):
        self.is_invalid = False
        self.material.is_broken = False

    def invalidate(self):
        self.is_invalid = True

    def update_corrosion(self) -> void:
        self.material.update_corrosion()


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
        x_index = int((self.xStart / (Conf.width / Conf.dim_grille[0])))
        y_index = int((self.yStart / (Conf.height / Conf.dim_grille[1])))
        indexes.append([x_index, y_index])

        if self.length > 1.2 * (Conf.width / Conf.dim_grille[0]) and x_index < Conf.dim_grille[0] - 1:
            # print("%i%i grande longueur %0.2f" % (x_index, y_index, self.angle))
            indexes.append([x_index + 1, y_index])

        if self.width > 1.2 * (Conf.height / Conf.dim_grille[1]) and y_index < Conf.dim_grille[1] - 1:
            indexes.append([x_index, y_index + 1])
            # print("%i%i grande largeur %0.2f" % (x_index, y_index, self.angle))

        return indexes

    def compare(self, b):
        value = np.sqrt(np.square(self.xStart - b.xStart) + np.square(self.yStart - b.yStart)
                        + np.square(self.angle - b.angle))
        return value < 100


class BrickMaterial:
    def __init__(self, color):
        self.color_name = Conf.color_dict[color]
        self.color = cv2.cvtColor(np.uint8([[[color/2, 128, 255]]]), cv2.COLOR_HLS2RGB).flatten() / 255.0
        self.conductivity, self.capacity, self.density, self.r_cor = Conf.color_to_mat[self.color_name]
        self.is_broken = False
        self.health = 1.5
        self.T = [0]  # °K

    def update_corrosion(self):
        self.health = max(0.0, self.health - 0.01 * (1.0 - self.r_cor))

    @property
    def diffusivity(self):
        return self.conductivity / (self.capacity * self.density)

    def break_mat(self):
        self.T[:, :] = Conf.temperature
        self.conductivity, self.capacity, self.density = 200, 0.001, 0.001
        pass


class BrickArray:
    def __init__(self, bricks):
        self.array = np.array([[None] * Conf.dim_grille[1]] * Conf.dim_grille[0])
        for i in range(Conf.dim_grille[0]):
            for j in range(Conf.dim_grille[1]):
                self.array[i][j] = Brick.get_brick(bricks, i, j)

        self.w = .07  # m
        self.h = .04  # m
        self.dx = 0.001  # m / points
        self.dy = 0.001  # m / points
        self.nx, self.ny = int(np.ceil(self.w / self.dx)), int(np.ceil(self.h / self.dy))  # points
        self.T = 293.0 * np.ones((self.ny, self.nx))  # °K
        self.sim_time = 0  # s
        self.heq = None
        self.step_x, self.step_y = 0, 0

    def get(self, i: int, j: int) -> Brick:
        return self.array[i][j] if 0 <= i < Conf.dim_grille[0] and 0 <= j < Conf.dim_grille[1] else None

    def set(self, i: int, j: int, value: Brick or None) -> void:
        self.array[i][j] = value

    def clear(self) -> void:
        self.array = np.array([[None] * Conf.dim_grille[1]] * Conf.dim_grille[0])

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
                        print("Brick removed: " + str(index) + " (%s)" % brick.material.color)
                        self.set(index[0], index[1], None)

    def init_heat_eq(self):

        _conductivity = np.ones((self.ny, self.nx))
        _density = 500 * np.ones((self.ny, self.nx))
        _capacity = 500 * np.ones((self.ny, self.nx))
        for i in range(self.nx):
            for j in range(self.ny):
                index_i = i / (self.nx / Conf.dim_grille[0])
                index_j = j / (self.ny / Conf.dim_grille[1])
                material = self.get(int(index_i), int(index_j)).material
                _conductivity[j, i] = material.conductivity
                _capacity[j, i] = material.capacity
                _density[j, i] = material.density
        self.heq: HeatEquation = HeatEquation(self.T, self.dx, _density, _conductivity, _capacity)
        self.step_x = self.heq.nx / Conf.dim_grille[1]
        self.step_y = self.heq.ny / Conf.dim_grille[0]

    def update(self, heating=True) -> void:
        if self.heq is not None:
            if heating:
                # conditions
                self.heq.temperature[:, 0] = 1500
                # time step
                self.heq.evolve_ts()
                self.sim_time += self.heq.dt

            for i in range(Conf.dim_grille[0]):
                for j in range(Conf.dim_grille[1]):
                    self.get(i, j).material.T = np.flipud(self.T[int(j * self.step_x): int((j + 1) * self.step_x),
                                                                 int(i * self.step_y): int((i + 1) * self.step_y)])

        # update heat for brick in contact to the liquid
        if heating:

            for i in range(Conf.dim_grille[0]):
                brick_i = self.get(i, 0)
                if brick_i is not None:
                    if not brick_i.material.is_broken:
                        brick_i.update_corrosion()
                        if brick_i.material.health <= 0.0:
                            brick_i.material.is_broken = True
                            brick_i.material.break_mat()
                        break
                    else:
                        pass
                        # self.get(i, 1).update_corrosion()

    def reset(self) -> void:
        self.T = 293.0 * np.ones((self.ny, self.nx))
        for column in self.array:
            for brick in column:
                if brick is not None:
                    brick.material.is_broken = False
                    brick.material.health = 1.0

    def is_valid(self) -> bool:
        for i in range(Conf.dim_grille[0]):
            for j in range(Conf.dim_grille[1]):
                if self.get(i, j) is None:
                    return False
        return True
