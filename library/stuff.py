from math import pi
from typing import Union

import numpy as np

speed_of_light = 299792458


def sign(x: float):
    """
    :param x: input arg, must be scalar
    :return: Returns sign of value x. If x is zero, returns zero.
    """
    if x:
        return np.copysign(1.0, x)
    else:
        return 0


def DB2RATIO(d):
    """
    Convert a value / array of values into linear scale

    Accepts scalars and numpy arrays
    :param d: dB scale value[s]
    :return: linear scale value[s]
    """
    return 10.0 ** (d / 10.0)


def RATIO2DB(x):
    """
    Convert a value / array of values into dB scale.

    Accepts scalars and numpy arrays
    :param x: input
    :return: dB scale of input
    """
    return 10.0 * np.log10(x)


def cart2sph(x, y, z):
    hxy = np.hypot(x, y)
    r = np.hypot(hxy, z)
    el = np.arctan2(z, hxy)
    az = np.arctan2(y, x)
    return az, el, r


def shannon_capacity(BW_Hz, lin_SINR):
    """Return Shannon's capacity for bandwidth BW_Hz and SNR equal to SINR"""
    return BW_Hz * np.log2(1 + lin_SINR)


def shannon_required_lin_SINR(BW_Hz, data_rate):
    """Return required linear SNR to achieve a given rate in bps over bandwidth BW_Hz"""
    return 2 ** (data_rate / BW_Hz) - 1


def free_space_path_loss(dist: float, frequency_Hz: float):
    return 20 * np.log10(dist) + 20 * np.log10(frequency_Hz) - 147.55


def friis_path_loss_dB(dist: [float, np.ndarray], frequency_Hz: float, n: float = 2.0) -> Union[float, np.ndarray]:
    """Return the path loss in dB according to Friis formula.

    n may be adjusted unlike free_space_path_loss
    Normally, returned path loss will be negative (as per the textbook formula definition)

    :param dist: distance in meters. Can be array of distances.
    :param frequency_Hz: carrier
    :param n: propagation constant (normally 2)
    """
    return RATIO2DB(np.power(speed_of_light / (frequency_Hz * 4 * pi * dist), n))


def friis_range(path_loss_dB: Union[np.ndarray, float], frequency_Hz: float, n: float = 2.0) -> Union[
    np.ndarray, float]:
    """Return the range according to Friis formula given path loss in dB. n may be adjusted

    If the path loss is negative, an absolute value is taken.

    :param path_loss_dB: path loss in dB
    :param frequency_Hz: carrier
    :param n: propagation constant (normally 2)
    """
    return speed_of_light / (np.power(DB2RATIO(-np.abs(path_loss_dB)), 1 / n) * 4 * pi * frequency_Hz)


def dic_parse(s: str, sep1: str = ' ', sep2: str = '_'):
    """Parse a string of form "A_4.3 B_3 C_0" into a python dictionary.
     No conversions are made to the values, i.e. the mapping is str to str. """
    d = {}
    try:
        fields = s.split(sep1)
        for f in fields:
            if f:
                n, v = f.split(sep2)
                d[n] = v
    finally:
        return d


