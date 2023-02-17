import numpy as np
import matplotlib.pyplot as plt
from math import pi
from matplotlib import cm
from channel.antenna import calculate_3gpp_antenna
from mpl_toolkits.mplot3d import Axes3D
plt.style.use('seaborn')
from scipy import interpolate
import pickle

# This script computes and saves antenna pattern with given configurations
# to save time for re-computing it during simulations

N_elements = 16
filename = str(N_elements)+'x' + str(N_elements) + '.pickle'

size = 1000
XX = np.linspace(-pi, pi, size)
YY = np.linspace(-pi, pi, size)
gain = np.zeros([size, size])
for i, aoa_az in enumerate(XX):
    for j, aoa_el in enumerate(YY):
        gain[i, j] = calculate_3gpp_antenna(aoa_az, aoa_el, N_elements)

X, Y = np.meshgrid(XX, YY)

fig = plt.figure()
ax = fig.gca(projection='3d')

surf = ax.plot_surface(X, Y, gain)
plt.title('Original')

interpolating_function0 = interpolate.interp2d(XX, YY, gain, kind='cubic')

with open(filename, 'wb') as handle:
    pickle.dump(interpolating_function0, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(filename, 'rb') as handle:
    b = pickle.load(handle)

interpolating_function = b

fig2 = plt.figure()
ax2 = fig2.gca(projection='3d')

f = np.zeros([size,size])
for i, aoa_az in enumerate(XX):
    for j, aoa_el in enumerate(YY):
        f[i, j] = interpolating_function(aoa_az, aoa_el)
ax2.plot_surface(X, Y, f, cmap=cm.coolwarm)
plt.title('Restored')

plt.show()
