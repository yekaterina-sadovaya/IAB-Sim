from math import exp
from numpy import log10
from scipy.constants import speed_of_light

# Maximal pathloss that kills any signal
PL_MAX = 300.0


def UMa_los_prob(d):
    if d <= 18:
        return 1.0
    else:
        p = 18 / d + exp(-d / 63) * (1 - 18 / d)
    return p


def UMi_street_canyon_los_prob(d):
    if d <= 18:
        return 1.0
    else:
        p = 18 / d + exp(-d / 36) * (1 - 18 / d)
    return p


def UMa_los(dist2d, dist3d, f_carrier, h_bs, h_ut):
    d_bp = 4.0 * (h_bs - 1.0) * (h_ut - 1.0) * f_carrier / speed_of_light
    if dist2d > 5000.0:
        return PL_MAX
    if dist2d < 10.0:
        dist2d = 10.0 + (10.0 - dist2d)
    if dist2d < d_bp:
        pl = 32.4 + 20 * log10(dist3d) + 20 * log10(f_carrier / 1e9)
    else:
        pl = 32.4 + 40.0 * log10(dist3d) + 20.0 * log10(f_carrier / 1e9) - 10.0 * log10(d_bp ** 2 + (h_bs - h_ut) ** 2)
    return pl


def UMa_nlos(dist2d, dist3d, f_carrier, h_bs, h_ut):
    pl_nlos = 13.54 + 39.08 * log10(dist3d) + 20 * log10(f_carrier / 1e9) - 0.6 * (h_ut - 1.5)
    if dist2d > 5000.0:
        return PL_MAX
    if dist2d < 10.0:
        dist2d = 10.0 + (10.0 - dist2d)
    pl_los = UMa_los(dist2d, dist3d, f_carrier, h_bs, h_ut)
    pl = max(pl_los, pl_nlos)
    return pl


def UMi_street_canyon_los(dist2d, dist3d, f_carrier, h_bs, h_ut):
    d_bp = 4.0 * (h_bs - 1.0) * (h_ut - 1.0) * f_carrier / speed_of_light
    if dist2d > 5000.0:
        return PL_MAX
    if dist2d < 10.0:
        dist2d = 10.0 + (10.0 - dist2d)

    if dist2d < d_bp:
        pl = 32.4 + 21 * log10(dist3d) + 20 * log10(f_carrier / 1e9)
    else:
        pl = 32.4 + 40.0 * log10(dist3d) + 20.0 * log10(f_carrier / 1e9) - 9.5 * log10(d_bp ** 2 + (h_bs - h_ut) ** 2)
    return pl


def UMi_street_canyon_nlos(dist2d, dist3d, f_carrier, h_bs, h_ut):
    pl_nlos = 35.3 * log10(dist3d) + 22.4 + 21.3 * log10(f_carrier / 1e9) - 0.3 * (h_ut - 1.5)
    if dist2d > 5000.0:
        return PL_MAX
    if dist2d < 10.0:
        dist2d = 10.0 + (10.0 - dist2d)
    pl_los = UMi_street_canyon_los(dist2d, dist3d, f_carrier, h_bs, h_ut)
    pl = max(pl_los, pl_nlos)
    return pl


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    from lib.vectors import norm
    from lib.utilities import toss_coin
    from lib.plot_utilities import matplotlib_nikita_style

    matplotlib_nikita_style()

    # Case 1
    h_bs = 25
    h_ut = 10
    src_pos = np.array([0, 0, h_bs])
    x_coord = np.arange(1, 350, 1)
    dist3d = []
    dist2d = []
    for i in x_coord:
        dst_pos = np.array([0, i, h_ut])
        d = dst_pos[0:2] - src_pos[0:2]
        d = np.hypot(*d)
        dist2d.append(d)
        dv = dst_pos - src_pos
        d = norm(dv)
        dist3d.append(d)

    pathloss_nlos = []
    # NLOS pathloss
    for i, j in zip(dist2d, dist3d):
        pl = UMa_nlos(i, j, f_carrier=30e9, h_bs=h_bs, h_ut=h_ut)
        pathloss_nlos.append(pl)
    pathloss_los = []
    # LOS pathloss
    for i, j in zip(dist2d, dist3d):
        pl = UMa_los(i, j, f_carrier=30e9, h_bs=h_bs, h_ut=h_ut)
        pathloss_los.append(pl)
    # pathloss
    pathloss = []
    for i, j in zip(dist2d, dist3d):
        p = UMa_los_prob(i)
        if toss_coin(p):
            pl = UMa_los(i, j, f_carrier=30e9, h_bs=h_bs, h_ut=h_ut)
        else:
            pl = UMa_nlos(i, j, f_carrier=30e9, h_bs=h_bs, h_ut=h_ut)
        pathloss.append(pl)

    plt.plot(dist2d, pathloss_nlos, 'r*', label="NLOS")
    plt.plot(dist2d, pathloss_los, 'b*', label="LOS")
    plt.plot(dist2d, pathloss, 'y+', label="PL")
    plt.xlabel('2D distance, m')
    plt.ylabel('Path loss, dB')
    plt.title('Path loss between IAB donor and IAB node')
    plt.legend()

    ##
    plt.show()
