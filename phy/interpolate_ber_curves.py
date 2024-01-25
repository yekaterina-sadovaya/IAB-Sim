import csv
from scipy import interpolate
import matplotlib.pyplot as plt
import pickle
import numpy as np

N_MCS = 29


def load_BER():
    ALL_BER_CS = {}
    for j in range(0, N_MCS):
        with open('phy/MCS_CURVES/MCS' + str(j) + '.pickle', 'rb') as handle:
            interpolating_function = pickle.load(handle)
        ALL_BER_CS[j] = interpolating_function
    return ALL_BER_CS


def interpolate_BER():
    plt.figure()
    for i in range(0, N_MCS):

        data = {}
        with open('MCS' + str(i) + '.csv', newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
            for n_row, row in enumerate(reader):
                data[n_row] = row

        f = interpolate.interp1d(data[0], data[1], fill_value="extrapolate")

        with open('MCS' + str(i) + '.pickle', 'wb') as handle:
            pickle.dump(f, handle, protocol=pickle.HIGHEST_PROTOCOL)

        plt.plot(data[0], data[1])

    plt.title('Original figure')
    plt.grid()
    plt.xlabel('SINR, dB')
    plt.ylabel('BER')


def plot_interpolation():
    plt.figure()
    x = np.arange(-20, 30, 0.1)
    for j in range(0, N_MCS):
        with open('MCS' + str(j) + '.pickle', 'rb') as handle:
            interpolating_function = pickle.load(handle)
            y = interpolating_function(x)
        plt.plot(x, y)

    plt.title('Restored figure')
    plt.grid()
    plt.xlabel('SINR, dB')
    plt.ylabel('BER')
    plt.show()


if __name__ == "__main__":
    interpolate_BER()
    plot_interpolation()
