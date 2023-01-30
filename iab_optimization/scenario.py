from iab_optimization.pathloss import UMa_los_prob, UMa_los, UMi_street_canyon_los, UMi_street_canyon_los_prob
from iab_optimization.lib.utilities import DB2RATIO, toss_coin
from math import exp, log2, log10
import numpy as np


def spectral_efficiency_UMi(dist2d, dist3d, f_carrier, h_bs, h_ut, P_t, B):
    pathloss = UMi_street_canyon_los(dist2d, dist3d, f_carrier, h_bs, h_ut)
    snr = P_t + 15 - pathloss + 10 + 174 - 10 * log10(B) - 13
    i = 3
    sinr_lin = DB2RATIO(snr - i)
    s = log2(1 + sinr_lin)
    return s


def spectral_efficiency_UMa(dist2d, dist3d, f_carrier, h_bs, h_ut, P_t, B):
    pathloss = UMa_los(dist2d, dist3d, f_carrier, h_bs, h_ut)
    snr = P_t + 15 - pathloss + 10 + 174 - 10 * log10(B) - 13
    i = 3
    sinr_lin = DB2RATIO(snr - i)
    s = log2(1 + sinr_lin)
    return s


def human_blockage(d, lambda_b, r_b, h_b, h_u, h_w):
    p = 1 - exp(-2 * lambda_b * r_b * (d * (h_b - h_u) / (h_w - h_u) + r_b))
    return p


def joint_blockage_UMi(d, lambda_b, r_b, h_b, h_u, h_w):
    p_b = 1 - UMi_street_canyon_los_prob(d)
    p_h = human_blockage(d, lambda_b, r_b, h_b, h_u, h_w)
    p = 1 - (1 - p_b) * (1 - p_h)
    return p


def joint_blockage_UMa(d, lambda_b, r_b, h_b, h_u, h_w):
    p_b = 1 - UMa_los_prob(d)
    p_h = human_blockage(d, lambda_b, r_b, h_b, h_u, h_w)
    p = 1 - (1 - p_b) * (1 - p_h)
    return p


def calculate_jain_index(x):
    jain_index = (np.sum(x))**2 / (x.shape[0] * np.sum(np.square(x)))
    return jain_index


