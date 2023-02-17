import numpy as np
import numpy.random as random
from math import sqrt, pi, cos, sin
from gl_vars import gl


def drop_IAB(seed):
    """
    Randomly generates IAB nodes positions
    withing the given radius
    """
    random.seed(seed)
    R = gl.cell_radius_m
    for n in range(0, gl.n_IAB):
        r = R * sqrt(random.random())
        theta = random.random() * 2 * pi
        x = r * cos(theta)
        y = r * sin(theta)
        gl.IAB_pos[n, :] = [x, y, gl.iab_height]


def drop_DgNB(seed):
    """
    Randomly generates the DgNB position
    on a circle
    """
    random.seed(seed)
    R = gl.cell_radius_m
    for n in range(0, 1):
        theta = random.random() * 2 * pi
        x = R * cos(theta)
        y = R * sin(theta)
        gl.DgNB_pos[n, :] = [x, y, gl.DgNB_height]

