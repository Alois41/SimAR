from OpenGL.GLUT import *
import numpy as np
import cv2
from settings.configuration import Config as Conf
from physics.heat_equation import HeatEquation


class Brick:
    """  Represent an element of the grid """
    def __init__(self, box=None, color=None, indexes=None):
        self.is_invalid = False   # True when this brick isn't on the grid anymore
        self.drowned = False      # True when liquid can pass through it

        # instantiate brick properties
        if color is None:  # void brick
            self.material = BrickMaterial()
            self.geometry, self.indexes = None, None
            self.indexes = indexes
            self.is_void = True
            self.drowned = True
        else:   # true brick
            self.material = BrickMaterial(color)
            self.geometry = BrickGeometry(box)
            self.indexes = self.geometry.compute_indexes()
            self.is_void = False

    @classmethod
    def void(cls, indexes):
        """ declare a "void" brick, usefull in some functions but just "air" in simulation """
        return cls(indexes=indexes)

    @classmethod
    def new(cls, box, color):
        return cls(box, color)

    @staticmethod
    def get_brick(brick_array, i: int, j: int, prev_brick=None):
        b_ij = next((b for b in brick_array if [i, j] in b.indexes), None)
        if prev_brick is not None:
            if prev_brick == b_ij:
                return None
        return b_ij

    def is_almost(self, b) -> void:
        """ brick comparison """
        if self.is_void or b.is_void:
            return False
        return self.geometry.compare(b.geometry) and np.array_equal(self.material.color_name, b.material.color)

    def replace(self) -> void:
        """ make brick valid and reset its state"""
        self.is_invalid = False
        self.material.is_broken = False

    def invalidate(self) -> void:
        """ make brick invalid to destroy it in the next 'clear_invalid' pass """
        self.is_invalid = True

    def update_corrosion(self, dt) -> void:
        self.material.update_corrosion(dt)


class BrickGeometry:
    """ Geometric properties of a brick"""
    def __init__(self, box):
        # extract values of box, change angle to avoid bad behaviour in drawing
        if box[2] > -45:
            self.xStart, self.yStart = box[0]
            self.length, self.width = box[1]
            self.angle = box[2]
        elif box[2] <= -45:
            self.xStart, self.yStart = box[0]
            self.width, self.length = box[1]
            self.angle = box[2] + 90

    def compute_indexes(self):
        """ compute indexes from brick position and grid parameters"""
        indexes = []
        x_index = int(((self.xStart + .5*self.width) / (Conf.width / Conf.dim_grille[0])))
        y_index = int(((self.yStart + .5*self.length) / (Conf.height / Conf.dim_grille[1])))
        indexes.append([x_index, y_index])

        # # Deprecated atm
        # if self.length > 1.2 * (Conf.width / Conf.dim_grille[0]) and x_index < Conf.dim_grille[0] - 1:
        #     # print("%i%i grande longueur %0.2f" % (x_index, y_index, self.angle))
        #     indexes.append([x_index + 1, y_index])
        #
        # if self.width > 1.2 * (Conf.height / Conf.dim_grille[1]) and y_index < Conf.dim_grille[1] - 1:
        #     indexes.append([x_index, y_index + 1])
        #     # print("%i%i grande largeur %0.2f" % (x_index, y_index, self.angle))

        return indexes

    def compare(self, b):
        """ compare two brick positions"""
        value = np.sqrt(np.square(self.xStart - b.xStart) + np.square(self.yStart - b.yStart)
                        + np.square(self.angle - b.angle))
        return value < 100


class BrickMaterial:
    """ material properties of a brick"""

    def __init__(self, color=-1):
        self.color_name = Conf.color_dict[color]
        self.color = cv2.cvtColor(np.uint8([[[color / 2, 50, 255]]]), cv2.COLOR_HLS2RGB).flatten() / 255.0
        self.conductivity, self.capacity, self.density, self.r_cor = Conf.color_to_mat[self.color_name]
        self.is_broken = False if color >= 0 else True
        self.health = 1.4
        self.T = [0]  # °K

    def update_corrosion(self, dt):
        self.health = max(0.0, self.health - .1 * dt * (1.0 - self.r_cor))

    @property
    def diffusivity(self):
        """ return a more trivial property """
        return self.conductivity / (self.capacity * self.density)


class BrickArray:
    """ Manage all current bricks """

    def __init__(self, bricks, liquid_im):
        self.array = np.array([[None] * Conf.dim_grille[1]] * Conf.dim_grille[0])
        for i in range(Conf.dim_grille[0]):
            for j in range(Conf.dim_grille[1]):
                b = Brick.get_brick(bricks, i, j)
                if b is None:
                    b = Brick.void([[i, j]])
                self.array[i][j] = b

        self.w = Conf.dim_grille[0]/10  # m
        self.h = Conf.dim_grille[1]/10  # m
        self.dx = 0.01  # m / points
        self.dy = 0.01  # m / points
        self.nx, self.ny = int(np.ceil(self.w / self.dx)), int(np.ceil(self.h / self.dy))  # points
        self.T = 293.0 * np.ones((self.ny, self.nx))  # °K
        self.sim_time = 0  # s
        self.heq = None
        self.step_x, self.step_y = 0, 0

        self.liquid_im = liquid_im

    def get(self, i: int, j: int) -> Brick:
        try:
            return self.array[i][j]
        except IndexError:
            return None

    def set(self, i: int, j: int, value: Brick or None) -> void:
        i, j = min(Conf.dim_grille[0]-1, i), min(Conf.dim_grille[1]-1, j)
        self.array[i][j] = value

    def clear(self) -> void:
        self.array = np.array([[None] * Conf.dim_grille[1]] * Conf.dim_grille[0])

    def invalidate(self) -> void:
        """ make all bricks invalid until next detection"""
        for column in self.array:
            for brick in column:
                if brick is not None:
                    brick.invalidate()

    def clear_invalid(self) -> void:
        """ remove not detected bricks"""
        for column in self.array:
            for brick in column:
                if brick is not None and brick.is_invalid:
                    for index in brick.indexes:
                        # print("Brick removed: " + str(index) + " (%s)" % brick.material.color)
                        self.set(index[0], index[1], Brick.void(brick.indexes))

    def update_eq(self):
        """ update heat Equation with brick and Config"""
        _conductivity = np.ones((self.ny, self.nx))
        _density = 500 * np.ones((self.ny, self.nx))
        _capacity = 500 * np.ones((self.ny, self.nx))
        _temperature = self.heq.temperature

        # Read liquid state from memory
        with self.liquid_im.get_lock():
            arr = np.frombuffer(self.liquid_im.get_obj())
            liquid_array = np.resize(arr, (10 * Conf.dim_grille[1], 10 * (Conf.dim_grille[0] + 1), 4))
            liquid_array = np.flip(liquid_array, 0)

        # for each element, set parameters
        index_i, index_j = None, None
        prev_index_i, prev_index_j = None, None
        brick = None
        for i in range(self.nx):
            for j in range(self.ny):
                index_i = int(i / (self.nx / Conf.dim_grille[0]))
                index_j = int(j / (self.ny / Conf.dim_grille[1]))
                if index_i != prev_index_i or index_j != prev_index_j:
                    brick = self.get(index_i, index_j)
                    prev_index_i, prev_index_j = index_i, index_j

                if brick is not None:
                    material = brick.material
                    _conductivity[j, i] = material.conductivity
                    _capacity[j, i] = material.capacity
                    _density[j, i] = material.density
                    if brick.is_void or brick.drowned:
                        conductivity, capacity, density, _ = Conf.color_to_mat["Air"]
                        _conductivity[j, i] = conductivity
                        _capacity[j, i] = capacity
                        _density[j, i] = density

                    # if there's liquid, set temp to liquid temp
                    if liquid_array[..., 0][j, i] > 0.2:
                        _temperature[j, i] = 1600
                        conductivity, capacity, density, _ = Conf.color_to_mat["Molten Steel"]
                        _conductivity[j, i] = conductivity
                        _capacity[j, i] = capacity
                        _density[j, i] = density

                else:
                    conductivity, capacity, density, _ = Conf.color_to_mat["Air"]
                    _conductivity[j, i] = conductivity
                    _capacity[j, i] = capacity
                    _density[j, i] = density

        # Re-create the solver
        self.heq = HeatEquation(_temperature, self.dx, self.dy, _density, _conductivity, _capacity)

    def init_heat_eq(self):
        """ instatiate heat equation solver"""
        _conductivity = np.ones((self.ny, self.nx))
        _density = 500 * np.ones((self.ny, self.nx))
        _capacity = 500 * np.ones((self.ny, self.nx))
        brick = None
        prev_index_i, prev_index_j = None, None
        for i in range(self.nx):
            for j in range(self.ny):
                index_i = int(i / (self.nx / Conf.dim_grille[0]))
                index_j = int(j / (self.ny / Conf.dim_grille[1]))
                if index_i != prev_index_i or index_j != prev_index_j:
                    brick = self.get(index_i, index_j)
                    prev_index_i, prev_index_j = index_i, index_j

                if brick is not None:
                    material = brick.material
                    _conductivity[j, i] = material.conductivity
                    _capacity[j, i] = material.capacity
                    _density[j, i] = material.density
                else:
                    _conductivity[j, i] = 0
                    _capacity[j, i] = 1
                    _density[j, i] = 1
        self.heq: HeatEquation = HeatEquation(self.T, self.dx, self.dy, _density, _conductivity, _capacity)
        self.step_x = self.heq.nx / Conf.dim_grille[1]
        self.step_y = self.heq.ny / Conf.dim_grille[0]

    def get_temp(self, i, j) -> np.ndarray:
        """ Get temperature of a brick """
        return np.flipud(self.T[int(j * self.step_x): int((j + 1) * self.step_x),
                         int(i * self.step_y): int((i + 1) * self.step_y)])

    def update(self, heating=False) -> void:

        # LIQUID UPDATE
        # update brick to fill with steel
        self.get(0, 0).drowned = True

        for i in range(Conf.dim_grille[0]):
            for j in range(Conf.dim_grille[1]):
                b = self.get(i, j)
                # b.drowned = b.is_void

                if b.drowned:
                    indexes = (i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)
                    for index in indexes:
                        b2 = self.get(index[0], index[1])
                        if b2 is not None and b2.material.is_broken:
                            b2.drowned = True
                            b2.material = BrickMaterial(color=-1)
                            self.set(index[0], index[1], b2)
                            continue
        # HEAT UPDATE
        if self.heq is not None:
            if heating:
                # self.heq.temperature[0, 0] = 1500
                # update corrosion

                for brick in self.array.flatten():
                    i, j = brick.indexes[0]
                    neighbors = [self.get(i - 1, j), self.get(i + 1, j), self.get(i, j - 1), self.get(i, j + 1)]
                    for n in neighbors:
                        if n is not None:
                            if len(self.get_temp(*n.indexes[0])) > 0:
                                if n.drowned and np.nanmean(self.get_temp(*n.indexes[0])) > 500:
                                    brick.update_corrosion(self.heq.dt)
                                    if brick.material.health <= 0.0:
                                        brick.material.is_broken = True

                for i in range(Conf.dim_grille[0]):
                    brick_i = self.get(i, 0)
                    if brick_i is not None:
                        if not brick_i.material.is_broken:
                            brick_i.update_corrosion(self.heq.dt)
                            if brick_i.material.health <= 0.0:
                                brick_i.material.is_broken = True
                            break
                # time step
                speed = 10
                for i in range(speed):
                    self.heq.evolve_ts()
                    self.sim_time += self.heq.dt

        self.update_eq()

    def reset(self) -> void:
        """ reset grid state"""
        self.T = 293.0 * np.zeros((self.ny, self.nx))

        with self.liquid_im.get_lock():
            arr = np.frombuffer(self.liquid_im.get_obj())
            arr[:] = np.zeros(arr.shape)

        for brick in self.array.flatten():
            brick.T = [0]
            brick.drowned = False
            if not brick.is_void:
                brick.material.is_broken = False
                brick.material.health = 1.0

        # self.update()
        self.init_heat_eq()

    def is_valid(self) -> bool:
        """ if the Config force the grid tobe full, check for it"""
        if not Conf.force_full:
            return True
        for i in range(Conf.dim_grille[0]):
            for j in range(Conf.dim_grille[1]):
                if self.get(i, j) is None:
                    return False
        return True

    def test_loose(self) -> bool:
        test = False
        for b in self.array.flatten():
            if b.drowned:
                for index in b.indexes:
                    test = True if index[0] >= Conf.dim_grille[0]-1 or index[1] >= Conf.dim_grille[1]-1 else test
        return test

    def get_grid(self):
        grid = np.zeros(self.array.shape)
        for b in self.array.flatten():
            if b.is_void or b.drowned:
                grid[min(Conf.dim_grille[0] - 1, b.indexes[0][0]),
                     min(Conf.dim_grille[1] - 1, b.indexes[0][1])] = 0.0
            else:
                grid[min(Conf.dim_grille[0] - 1, b.indexes[0][0]),
                     min(Conf.dim_grille[1] - 1, b.indexes[0][1])] = -1

        return np.flip(grid, 1)

    def get_capacity(self):
        """ """

        # Read liquid state from memory
        with self.liquid_im.get_lock():
            arr = np.frombuffer(self.liquid_im.get_obj())
            c = len(arr[arr > 0.0])

        return c