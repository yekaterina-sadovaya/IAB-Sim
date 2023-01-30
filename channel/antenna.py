import matplotlib.pyplot as plt
import numpy as np
from gl_vars import gl
from math import sqrt, cos, sin, pi, log10
from cmath import exp
import pickle5 as pickle

wavelength = 299792458/gl.carrier_frequency_Hz
d_h = wavelength/2
d_v = d_h


def calculate_3gpp_antenna(aoa_az, aoa_el, N):
    N_hor = N
    N_vert = N
    # element pattern of each antenna elements
    Am = 30   # the front-back ratio
    SLA = 30  # the lower limit
    G = 8   # maximum gain of an antenna element
    ang_3db = 65*pi/180
    A_eh = -min(12 * ((aoa_az / ang_3db) ** 2), Am)
    A_ev = -min(12 * ((aoa_el / ang_3db) ** 2), SLA)
    # magnitude of an element pattern
    P_E = G - min(-(A_eh + A_ev), Am)
    P_e = 10**(P_E/20)

    a = []
    # calculate phase shift and weighting factor for all array elements
    for m in range(0,N_hor):
        for n in range(0, N_vert):
            v = exp(-2j*pi*((n-1)*cos(aoa_el)*(d_v/wavelength) + (m-1)*sin(aoa_el)*sin(aoa_az)*(d_h/wavelength) ))
            w = (1/sqrt(N_hor*N_vert))*exp(2j*pi*((n-1)*cos(aoa_el)*(d_v/wavelength) +
                                                  (m-1)*cos(aoa_el)*sin(aoa_az)*(d_h/wavelength)))
            E_mn = P_e*v*w
            a.append(E_mn)
    # calculate the gain as the superposition of the elements
    G = 20*log10(abs(sum(a)))
    return G


def antenna3gpp(N_ant):
    filename = 'channel/' + str(N_ant) + 'x' + str(N_ant) + '.pickle'
    with open(filename, 'rb') as handle:
        interpolating_function = pickle.load(handle)

    return interpolating_function


def attenuation_in_direction(phi_i, theta_i, ang_3db):
    SLh = 20
    SLv = 20
    SL = 30

    A_eh = -min(12 * ((phi_i / ang_3db) ** 2), SLh)
    A_ev = -min(12 * ((theta_i / ang_3db) ** 2), SLv)

    return 15-min(-(A_eh + A_ev), SL)


def beam_split(active_ues):
    if len(active_ues) >= 8:
        gl.IAB_tx_power_dBm = 23.9794
        gl.DgNB_tx_power_dBm = 33.9794
    elif 8 > len(active_ues) >= 6:
        gl.DgNB_tx_power_dBm = 35.2287874528
        gl.IAB_tx_power_dBm = 25.2287874528
    elif 6 > len(active_ues) >= 4:
        gl.DgNB_tx_power_dBm = 36.9897
        gl.IAB_tx_power_dBm = 26.9897
    else:
        gl.DgNB_tx_power_dBm = 40
        gl.IAB_tx_power_dBm = 30


if __name__ == "__main__":
    plt.rc('font', **{'family': 'serif'})
    plt.rcParams['pdf.fonttype'] = 42

    plt.style.use('YS_plot_style.mplstyle')

    N_elements = 2
    size = 1000
    XX = np.linspace(-pi, pi, size)
    YY = np.linspace(-pi, pi, size)
    gain2 = np.zeros(size)
    gain4 = np.zeros(size)
    gain8 = np.zeros(size)
    gain16 = np.zeros(size)
    for i, aoa_az in enumerate(XX):
        # for j, aoa_el in enumerate(YY):
        gain2[i] = calculate_3gpp_antenna(aoa_az, 0, N_elements)
        gain4[i] = calculate_3gpp_antenna(aoa_az, 0, 4)
        gain8[i] = calculate_3gpp_antenna(aoa_az, 0, 8)
        gain16[i] = calculate_3gpp_antenna(aoa_az, 0, 16)
    print(max(gain2))
    XX = XX*180/pi
    plt.plot(XX, gain16)
    plt.plot(XX, gain8)
    plt.plot(XX, gain4)
    plt.plot(XX, gain2)
    plt.xlabel('$\phi$, $\degree$')
    plt.ylabel('Antenna Gain, dBi')
    plt.legend(['16 × 16', '8 × 8', '4 × 4', '2 × 2'])
    plt.xlim([-180, 180])
    plt.ylim([-60, 35])
    plt.show()
