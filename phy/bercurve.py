from gl_vars import gl
from library.stat_container import st

import numpy as np

BLERCURVE_PREFIX = "BLER_CURVE"
BLERCURVE_SUFFIX = ".interp"
BLERCURVE_TOLERANCE = 0.01


class PHYparams:
    def __init__(self, M, R, MCS):
        self.code_rate = R
        self.mod_order = M
        self.MCS = MCS


def snr2mcs(snr, BERs):
    y = np.zeros(BERs.__len__())
    for i in range(0, BERs.__len__()):
        # 24 corresponds to the minimum possible transport block
        y[i] = BERs[i](snr)     # * gl.MAX_CB_size_bits

    y_min = np.where(y == np.amin(y))
    y_min = y_min[0]

    if len(y_min) > 1:
        y_min = np.array(y_min)
        MCS = np.max(y_min)

    elif y[y_min] > gl.target_BLER:
        MCS = -1

    else:
        MCS = y_min[0]

    return MCS


def set_params_PHY(SNR, BERs):
    MCS = snr2mcs(SNR, BERs)
    params = {-1: PHYparams(0, 0, MCS),
              0: PHYparams(2, 120/1024, MCS),
              1: PHYparams(2, 157/1024, MCS),
              2: PHYparams(2, 193/1024, MCS),
              3: PHYparams(2, 251/1024, MCS),
              4: PHYparams(2, 308/1024, MCS),
              5: PHYparams(2, 379 / 1024, MCS),
              6: PHYparams(2, 449/1024, MCS),
              7: PHYparams(2, 526 / 1024, MCS),
              8: PHYparams(2, 602/1024, MCS),
              9: PHYparams(2, 679 / 1024, MCS),
              10: PHYparams(4, 340 / 1024, MCS),
              11: PHYparams(4, 378/1024, MCS),
              12: PHYparams(4, 434 / 1024, MCS),
              13: PHYparams(4, 490/1024, MCS),
              14: PHYparams(4, 553 / 1024, MCS),
              15: PHYparams(4, 616/1024, MCS),
              16: PHYparams(4, 658 / 1024, MCS),
              17: PHYparams(6, 438 / 1024, MCS),
              18: PHYparams(6, 466/1024, MCS),
              19: PHYparams(6, 517 / 1024, MCS),
              20: PHYparams(6, 567/1024, MCS),
              21: PHYparams(6, 616 / 1024, MCS),
              22: PHYparams(6, 666/1024, MCS),
              23: PHYparams(6, 719 / 1024, MCS),
              24: PHYparams(6, 772/1024, MCS),
              25: PHYparams(6, 822 / 1024, MCS),
              26: PHYparams(6, 873 / 1024, MCS),
              27: PHYparams(6, 910 / 1024, MCS),
              28: PHYparams(6, 948 / 1024, MCS)
              }
    try:
        return params[MCS]
    except KeyError:
        return params[0]



