import cv2
import numpy as np
from time import clock
from multiprocessing import Process, SimpleQueue
import sys
from source.configuration import Config as Conf

GRAVITY = 9.81
delta_t = .1


class Liquid(Process):
    """ Class with multiprocessing class heritage, control liquid evolution and send liquid image to main process"""

    class LiquidControl:
        """ Liquid control class
            for visual effects, no real physic equation are used
        """
        liquid_grid = None

        def __init__(self, _grid, liquid_grid=None):
            """ Initialise class attributes """
            super().__init__()
            self.grid = _grid
            self.quantity = 0
            self.liquid_grid = liquid_grid

            self.i, self.j = None, None
            self.first_pass = True

        def update(self, image, pouring):
            """ Liquid position update """

            # initialize liquid_grid from the grid given by brick detection
            if self.liquid_grid is None:
                # same shape as the final image but with RGB -> 1 value
                self.liquid_grid = np.zeros(image.shape[:2])
                tmp = np.zeros(image.shape[:2])

                # prepare grid before algorithm
                # iterate over the grid
                for (x, y), element in np.ndenumerate(self.grid):
                    if element == -1:  # -1 for brick pos
                        # fill tmp image according to bricks
                        cv2.rectangle(tmp, (10 * x, 10 * y), (10 * x + 9, 10 * y + 9), 1, thickness=cv2.FILLED)

                # set value to NaN according to bricks
                self.liquid_grid[tmp > 0.0] = np.nan

            # Initialize attributes that control element update order
            if self.i is None:
                self.i = np.linspace(len(self.liquid_grid) - 1, 0, len(self.liquid_grid), dtype=int)
                self.j = np.linspace(0, len(self.liquid_grid[0]) - 1, len(self.liquid_grid[0]),
                                     dtype=int)

            # self.i = self.i[np.argsort(np.nanmean(self.liquid_grid, axis=1))][::-1]
            # Loop over each element and update water quantities
            for x in self.i:
                self.j = self.j[np.argsort(self.liquid_grid[x])]
                for y in self.j:

                    # Pass to an other element if this one is empty
                    quantity = self.liquid_grid[x, y]
                    if np.isnan(quantity) or quantity <= 0:
                        continue

                    to_share = quantity - .2

                    # Initialise vars
                    neighbors = {}
                    up, down, left, right = None, None, None, None
                    this = self.liquid_grid[x, y]
                    _shape = self.liquid_grid.shape

                    # Test if neighbors elements exist
                    if x > 0:
                        down = self.liquid_grid[x - 1, y]
                        if not np.isnan(down):
                            neighbors["DOWN"] = down
                    if y < _shape[1] - 1:
                        left = self.liquid_grid[x, y + 1]
                        if not np.isnan(left):
                            neighbors["LEFT"] = left
                    if y > 0:
                        right = self.liquid_grid[x, y - 1]
                        if not np.isnan(right):
                            neighbors["RIGHT"] = right
                    if x < _shape[0] - 1:
                        up = self.liquid_grid[x + 1, y]
                        if not np.isnan(up):
                            neighbors["UP"] = up
                    if len(neighbors) == 0:
                        continue

                    # Find lowest neighbors
                    minval = min(neighbors.values())
                    res = [k for k, v in neighbors.items() if v == minval]
                    q = to_share / (len(res) + 1)

                    # Update values

                    # Force gravity-like effect
                    if down is not None:
                        if down + 1 <= this:  # or (not fill and d < 4):
                            self.liquid_grid[x - 1, y] += quantity - .2
                            self.liquid_grid[x, y] -= quantity - .2
                            # Re-calculate var
                            continue
                            # to_share = LiquidControl.liquid_grid[x, y] - .2
                            # if to_share < 0:
                            #     continue
                            # q = to_share / (len(res) + 1)

                    # Share water quantity to every lowest neighbor

                    if "DOWN" in res:
                        if down < 2 or (left is not None and down < left) or (right is not None and down < right):
                            self.liquid_grid[x - 1, y] += to_share
                            self.liquid_grid[x, y] -= to_share
                            continue

                    if "LEFT" in res:
                        if left <= this and left < 2:
                            self.liquid_grid[x, y + 1] += q
                            self.liquid_grid[x, y] -= q

                    if "RIGHT" in res:
                        if right <= this and right < 2:
                            self.liquid_grid[x, y - 1] += q
                            self.liquid_grid[x, y] -= q

                    if "UP" in res and pouring:
                        if this > up + 1 and up < 2 and (left is not None and left > up + 1) or (right is not None and right > up + 1):
                            self.liquid_grid[x + 1, y] += .5 * q
                            self.liquid_grid[x, y] -= .5 * q

            # update liquid image
            for (x, y), element in np.ndenumerate(self.liquid_grid):
                if not np.isnan(element):
                    if element > 0.0:
                        image[x, y] = np.array([1, 0.3, 0, max(0.0, min(1.0, element * 100.0))])
                    else:
                        image[x, y] = (0.0, 0.0, 0.0, 0.0)

        def setup(self, pouring, dt):
            """ Change simulation params"""

            if pouring:
                if self.first_pass:
                    # initialize liquid_grid from the grid given by brick detection
                    self.liquid_grid = np.zeros(self.liquid_grid.shape)
                    tmp = np.zeros(self.liquid_grid.shape)
                    for (x, y), element in np.ndenumerate(self.grid):
                        if element == -1:
                            cv2.rectangle(tmp, (10 * x, 10 * y), (10 * x + 9, 10 * y + 9), 1, thickness=cv2.FILLED)
                    self.liquid_grid[tmp > 0.0] = np.nan
                    self.first_pass = False

                # Distribute liquid evenly
                self.liquid_grid[np.isnan(self.liquid_grid)] = -1
                high_val = self.liquid_grid[self.liquid_grid > .5]
                if len(high_val) > 0:
                    high_mean = np.nanmean(high_val)
                    self.liquid_grid[self.liquid_grid > .5] = high_mean
                self.liquid_grid[self.liquid_grid == -1] = np.nan

                # add a liquid source
                self.liquid_grid[-1, 4:6] = self.liquid_grid[-1, 4:6] + dt * 1E9  # top, 5 to 6 pxl right

            else:
                # Get rid of liquid
                self.first_pass = True
                i = np.linspace(len(self.liquid_grid) - 1, 0, len(self.liquid_grid), dtype=int)
                j = np.linspace(0, len(self.liquid_grid[0]) - 1, len(self.liquid_grid[0]), dtype=int)
                for x in i:
                    if np.nansum(self.liquid_grid[x, :]) > 0.0:
                        for y in j:
                            if not np.isnan(self.liquid_grid[x, y]):
                                self.liquid_grid[x, y] = 0
                        break

        def update_grid(self, grid):
            """ change simulation brick grid """
            self.grid = grid
            self.liquid_grid[np.isnan(self.liquid_grid)] = 0.0
            tmp = np.zeros(self.liquid_grid.shape[:2])
            # print(np.resize(self.grid, (Conf.dim_grille[0] + 1, Conf.dim_grille[1])))
            for (x, y), element in np.ndenumerate(self.grid):
                if element == -1:
                    cv2.rectangle(tmp, (10 * x, 10 * y), (10 * x + 9, 10 * y + 9), 1, thickness=cv2.FILLED)
            self.liquid_grid[tmp > 0.0] = np.nan

    def __init__(self, liquid_im, q_active: SimpleQueue, rst: SimpleQueue, lost: SimpleQueue, liquid_grid):
        """ initialisation of Processes objects shared with other processes"""
        super().__init__()
        # Process objects, shared between main Process and this one
        self.liquid_im = liquid_im      # Array buffer ~ C array
        self.liquid_grid = liquid_grid  # Array buffer ~ C array
        self.q_active = q_active        # Queue
        self.rst = rst                  # Queue
        self.lost = lost                # Queue

        self.time = 0         # normal attribute

    def run(self) -> None:
        """ main method of the Thread, <processObj>.start() execute it once"""

        time = clock()
        while True:
            # Initialise method vars
            grid, new_grid = None, None
            level, image = None, None
            is_pouring = False
            rst = False
            stopped = False

            as_reach_limit = False

            # Make a memory link between liquid_grid and new_grid
            with self.liquid_grid.get_lock():  # wait for obj to be readable
                new_grid = np.frombuffer(self.liquid_grid.get_obj())

            # Process Loop, shouldn't stop
            while not rst:

                # If Queue has an object waiting, read it
                if not self.q_active.empty():
                    is_pouring = self.q_active.get()

                if not self.rst.empty():
                    print("rst")
                    rst = self.rst.get()

                # make the sim running a little bit after lose and wait for reset
                if as_reach_limit:
                    if stopped:
                        continue
                    elif clock() - self.time >= 1:
                        self.lost.put(True)
                        stopped = True
                        continue

                # If the grid has changed
                if not np.array_equal(new_grid, grid):
                    grid = new_grid.copy()   # set the grid to the new one without memory link

                    # Create liquid Class object if needed
                    if level is None:
                        level = self.LiquidControl(np.resize(grid, (Conf.dim_grille[0] + 1, Conf.dim_grille[1])))
                        # extend width by one for liquid leak
                        image = np.zeros((10 * Conf.dim_grille[1], 10 * (Conf.dim_grille[0] + 1), 4), dtype=float)

                    # Else, update the Class object with the new grid
                    else:
                        level.update_grid(np.resize(grid, (Conf.dim_grille[0] + 1, Conf.dim_grille[1])))

                # Continue liquid simulation
                if level is not None:
                    level.update(image, is_pouring)
                    time = clock()
                    level.setup(is_pouring, clock() - time)

                # Update to main Process : write in liquid image memory location
                with self.liquid_im.get_lock():
                    arr = np.frombuffer(self.liquid_im.get_obj())
                    arr[:] = image.flatten()

                # Update losing state to main process
                if not as_reach_limit:
                    with self.liquid_im.get_lock():
                        arr = np.frombuffer(self.liquid_im.get_obj())
                        arr = np.resize(arr, (10 * Conf.dim_grille[1], 10 * (Conf.dim_grille[0] + 1), 4))
                        if np.max(arr[6, :]) >= 1.0:
                            self.time = clock()
                            as_reach_limit = True
                            print("limit")


