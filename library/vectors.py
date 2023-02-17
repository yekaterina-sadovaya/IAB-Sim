import warnings
from math import sqrt, copysign, log
import numpy as np


def vector_rot90(v):
    """rotates a 3-vector around z axis counter-clockwise"""
    v[:] = v[1], -v[0], v[2]
    return v


def norm(v: np.ndarray) -> [np.ndarray, float]:
    return np.sqrt((v * v).sum(axis=0))


def vector_normalize(v: np.ndarray) -> np.ndarray:
    """Limit the vector's length to 1, keeping direction intact"""
    n = norm(v)
    assert n > 0, "Can not normalize null vector!"
    return v / norm(v)


def distance(p1: np.ndarray, p2: np.ndarray) -> float:
    """
    Compute distance between two points
    :param p1: N-vector point
    :param p2: N-vector point
    :return:
    """
    return norm(p1 - p2)


def distance_multipoint(base: np.ndarray, points: np.ndarray) -> np.ndarray:
    """
    Compute distance between base and multiple points. Dimensions must agree.

    :param base: Base point, N-vector
    :param points: Test locations, NxM matrix
    :return: M-vector of distances
    """
    return norm((points - base).T)


def vector_project(v: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Project point p onto vector v"""
    return p * np.dot(p, v) / (norm(p) ** 2)


def insert_zero_coefficients(s_np_zero_elemnts, vector_1, vector_2):

    if len(s_np_zero_elemnts) != 0:
        for zero_index in s_np_zero_elemnts:
            vector_1 = np.insert(vector_1, 0, zero_index)
            vector_2 = np.insert(vector_2, 0, zero_index)

    return vector_1, vector_2


