import cv2
import numpy as np
from time import clock
from multiprocessing import Process, SimpleQueue
import sys
from source.configuration import Config as Conf

GRAVITY = 9.81
delta_t = .1


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
        LiquidControl.liquid_grid = liquid_grid

        self.i, self.j = None, None
        self.first_pass = True

    def update(self, image, pouring=True):
        """ Liquid position update """

        # initialize liquid_grid from the grid given by brick detection
        if LiquidControl.liquid_grid is None:
            # same shape as the final image but with RGB -> 1 value
            LiquidControl.liquid_grid = np.zeros(image.shape[:2])
            tmp = np.zeros(image.shape[:2])

            # prepare grid before algorithm
            # iterate over the grid
            for (x, y), element in np.ndenumerate(self.grid):
                if element == -1:  # -1 for brick pos
                    # fill tmp image according to bricks
                    cv2.rectangle(tmp, (10 * x, 10 * y), (10 * x + 9, 10 * y + 9), 1, thickness=cv2.FILLED)

            # set value to NaN according to bricks
            LiquidControl.liquid_grid[tmp > 0.0] = np.nan

        # Initialize attributes that control element update order
        if self.i is None:
            self.i = np.linspace(len(LiquidControl.liquid_grid) - 1, 0, len(LiquidControl.liquid_grid), dtype=int)
            self.j = np.linspace(0, len(LiquidControl.liquid_grid[0]) - 1, len(LiquidControl.liquid_grid[0]), dtype=int)

        # Loop over each element and update water quantities
        for x in self.i:
            for y in self.j:

                # Pass to an other element if this one is empty
                quantity = LiquidControl.liquid_grid[x, y]
                if np.isnan(quantity) or quantity <= 0:
                    continue

                to_share = quantity - .2

                # Initialise vars
                neighbors = {}
                up, down, left, right = None, None, None, None
                this = LiquidControl.liquid_grid[x, y]
                _shape = LiquidControl.liquid_grid.shape

                # Test if neighbors elements exist
                if x > 0:
                    down = LiquidControl.liquid_grid[x - 1, y]
                    if not np.isnan(down):
                        neighbors["DOWN"] = down
                if y < _shape[1] - 1:
                    left = LiquidControl.liquid_grid[x, y + 1]
                    if not np.isnan(left):
                        neighbors["LEFT"] = left
                if y > 0:
                    right = LiquidControl.liquid_grid[x, y - 1]
                    if not np.isnan(right):
                        neighbors["RIGHT"] = right
                if x < _shape[0] - 1:
                    up = LiquidControl.liquid_grid[x + 1, y]
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
                        LiquidControl.liquid_grid[x - 1, y] += quantity - .2
                        LiquidControl.liquid_grid[x, y] -= quantity - .2
                        # Re-calculate var
                        to_share = LiquidControl.liquid_grid[x, y] - .2
                        if to_share < 0:
                            continue
                        q = to_share / (len(res) + 1)

                # Share water quantity to every lowest neighbor
                if "RIGHT" in res:
                    if right <= this and right < 2:
                        LiquidControl.liquid_grid[x, y - 1] += q
                        LiquidControl.liquid_grid[x, y] -= q

                if "LEFT" in res:
                    if left <= this and left < 2:
                        LiquidControl.liquid_grid[x, y + 1] += q
                        LiquidControl.liquid_grid[x, y] -= q

                if "UP" in res and pouring:
                    if this > up + 1 and up < 2:
                        LiquidControl.liquid_grid[x + 1, y] += .5*q
                        LiquidControl.liquid_grid[x, y] -= .5*q

                if "DOWN" in res:
                    if down < 2:
                        LiquidControl.liquid_grid[x - 1, y] += LiquidControl.liquid_grid[x, y]
                        LiquidControl.liquid_grid[x, y] = 0

        # update liquid image
        for (x, y), element in np.ndenumerate(LiquidControl.liquid_grid):
            if not np.isnan(element):
                if element > 0.0:
                    image[x, y] = np.array([1, 0.3, 0, max(0.0, min(1.0, element*100.0))])
                else:
                    image[x, y] = (0.0, 0.0, 0.0, 0.0)

    def setup(self, pouring=True):
        """ Change simulation params"""

        # self.liquid_grid[4:-2, -9:-1] = 0
        # self.liquid_grid[-1, -10:-1] = 0

        if pouring:

            if self.first_pass:
                # initialize liquid_grid from the grid given by brick detection
                LiquidControl.liquid_grid = np.zeros(LiquidControl.liquid_grid.shape)
                tmp = np.zeros(self.liquid_grid.shape)
                for (x, y), element in np.ndenumerate(self.grid):
                    if element == -1:
                        cv2.rectangle(tmp, (10 * x, 10 * y), (10 * x + 9, 10 * y + 9), 1, thickness=cv2.FILLED)
                LiquidControl.liquid_grid[tmp > 0.0] = np.nan
                self.first_pass = False

            # Distribute liquid evenly
            high_mean = np.nanmean(LiquidControl.liquid_grid[LiquidControl.liquid_grid > 0.5])
            LiquidControl.liquid_grid[LiquidControl.liquid_grid > 0.5] = high_mean

            # add a liquid source
            LiquidControl.liquid_grid[-1, 4] = LiquidControl.liquid_grid[-1, 4] + 20   # top, 5 pxl right
            # self.liquid_grid[-1, 5] = self.liquid_grid[-1, 5] + 10 # top, 6 pxl right

        else:
            # Get rid of liquid
            self.first_pass = True
            i = np.linspace(len(LiquidControl.liquid_grid) - 1, 0, len(LiquidControl.liquid_grid), dtype=int)
            j = np.linspace(0, len(LiquidControl.liquid_grid[0]) - 1, len(LiquidControl.liquid_grid[0]), dtype=int)
            for x in i:
                if np.nansum(LiquidControl.liquid_grid[x, :]) > 0.0:
                    for y in j:
                        if not np.isnan(LiquidControl.liquid_grid[x, y]):
                            LiquidControl.liquid_grid[x, y] = 0
                    break

    def update_grid(self, grid):
        """ change simulation brick grid """
        self.grid = grid
        LiquidControl.liquid_grid[np.isnan(LiquidControl.liquid_grid)] = 0.0
        tmp = np.zeros(LiquidControl.liquid_grid.shape[:2])
        # print(np.resize(self.grid, (Conf.dim_grille[0] + 1, Conf.dim_grille[1])))
        for (x, y), element in np.ndenumerate(self.grid):
            if element == -1:
                cv2.rectangle(tmp, (10 * x, 10 * y), (10 * x + 9, 10 * y + 9), 1, thickness=cv2.FILLED)
        LiquidControl.liquid_grid[tmp > 0.0] = np.nan


class Liquid(Process):
    """ Class with multiprocessing class heritage, control liquid evolution and send liquid image to main process"""

    def __init__(self, liquid_im, q_active: SimpleQueue, liquid_grid):
        """ initialisation of Processes objects shared with other processes"""
        super().__init__()
        # Process objects, shared between main Process and this one
        self.liquid_im = liquid_im      # Array buffer ~ C array
        self.liquid_grid = liquid_grid  # Array buffer ~ C array
        self.q_active = q_active        # Queue

    def run(self) -> None:
        """ main method of the Thread, <processObj>.start() execute it once"""

        # Initialise method vars
        grid, new_grid = None, None
        level, image = None, None
        is_pouring = False

        # Make a memory link between liquid_grid and new_grid
        with self.liquid_grid.get_lock():  # wait for obj to be readable
            new_grid = np.frombuffer(self.liquid_grid.get_obj())

        # Process Loop, shouldn't stop
        while True:

            # If Queue has an object waiting, read it
            if not self.q_active.empty():
                is_pouring = self.q_active.get()

            # If the grid has changed
            if not np.array_equal(new_grid, grid):
                grid = new_grid.copy()   # set the grid to the new one without memory link

                # Create liquid Class object if needed
                if level is None:
                    level = LiquidControl(np.resize(grid, (Conf.dim_grille[0] + 1, Conf.dim_grille[1])))
                    # extend width by one for liquid leak
                    image = np.zeros((10 * Conf.dim_grille[1], 10 * (Conf.dim_grille[0] + 1), 4), dtype=float)

                # Else, update the Class object with the new grid
                else:
                    level.update_grid(np.resize(grid, (Conf.dim_grille[0] + 1, Conf.dim_grille[1])))

            # Continue liquid simulation
            if level is not None:
                level.update(image, is_pouring)
                level.setup(is_pouring)

            # Update to main Process : write in liquid image memory location
            with self.liquid_im.get_lock():
                arr = np.frombuffer(self.liquid_im.get_obj())
                arr[:] = image.flatten()

    def test_loose(self):
        """ return True if the liquid has reached forbidden areas """
        with self.liquid_im.get_lock():
            arr = np.frombuffer(self.liquid_im.get_obj())
            arr = np.resize(arr, (10 * Conf.dim_grille[1], 10 * (Conf.dim_grille[0] + 1), 4))
            if arr[0, -1, 1] > 0.2:
                return True


