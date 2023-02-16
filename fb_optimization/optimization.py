import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from channel.propagation import spectral_efficiency_UMa, spectral_efficiency_UMi
from fb_optimization.baseclass import *
from numpy.linalg import norm
import os


class Optimization(BaseClass):
    """
    Optimization of time allocation for
    the full-buffer traffic
    """
    def __init__(self, params):
        BaseClass.__init__(self, params)

    def optimize_single_link(self, s_np_iab_DL, s_np_iab_UL, s_np_bs_DL, s_np_bs_UL, s_m_DL, s_m_UL):

        if gl.print_Flag is True:
            print('Optimization starts...')

        optimization_output = BaseClass.optimizae_single_link(self, s_np_iab_DL, s_np_iab_UL, s_np_bs_DL,
                                                              s_np_bs_UL, s_m_DL, s_m_UL)
        if gl.print_Flag is True:
            print('Finnished optimizing.')

        if optimization_output is not None:
            y_1 = optimization_output[0]
            y_2 = optimization_output[1]
            y_3 = optimization_output[2]
            y_4 = optimization_output[3]
            x_1 = optimization_output[4]
            x_2 = optimization_output[5]
            x_3 = optimization_output[6]
            x_4 = optimization_output[7]
            y_b1 = optimization_output[8]
            y_1b = optimization_output[9]
            eps_1 = optimization_output[10]
            eps_2 = optimization_output[11]
            eps_3 = optimization_output[12]
            eps_4 = optimization_output[13]
            return BaseClass.post_process_results(self, y_1, y_2, y_3, y_4, x_1, x_2, x_3, x_4, y_b1, y_1b,
                                                  s_np_iab_DL, s_np_iab_UL, s_np_bs_DL, s_np_bs_UL,
                                                  eps_1, eps_2, eps_3, eps_4)
        else:
            return None

    def plot_allocations(self, optimization_output):
        h = optimization_output[0]
        eps = optimization_output[1]
        y = optimization_output[2]
        x = optimization_output[3]
        backhaul = optimization_output[4]

        # epsilon
        fg = plt.figure()
        eps = np.around(eps, 2)
        barWidth = 0.4
        r1 = np.arange(1, eps.shape[0] + 1)
        plt.bar(r1, eps, width=barWidth, edgecolor='black', align='center', alpha=0.5, color='b')
        for i in range(eps.shape[0]):
            plt.text(i+1, eps[i], str(eps[i]))
        plt.xlabel('Timeslot')
        plt.ylabel('Timeslot duration')
        plt.ylim([0, 0.8])
        ax = fg.gca()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        # plt.title('Max Min fair resource allocation')

        # allocations
        fg = plt.figure()
        y_pos = np.arange(1, h.shape[0] + 1)
        plt.bar(y_pos, h, align='center', alpha=0.5)
        for i in range(h.shape[0]//2):
            plt.text(i+1, h[i], str('D'))
        for i in range(h.shape[0]//2):
            plt.text(h.shape[0]//2+i+1, h[i], str('U'))
        plt.xlabel('UE data rates')
        plt.ylabel('$h_{n}, Mbps$')
        ax = fg.gca()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        # plt.ylim([0, 70])

        # time slots 1
        fg = plt.figure()
        barWidth = 0.4
        r1 = np.arange(1, y.shape[0] + 1)
        plt.bar(r1, y, width=barWidth, edgecolor='black', label='UE allocations', hatch='//')
        plt.bar(r1, backhaul, width=barWidth, edgecolor='black', label='Backhaul allocations', hatch='\\', bottom=y)
        plt.xlabel('Timeslots')
        plt.ylabel('Value')
        ax = fg.gca()
        plt.legend(loc='best')
        plt.ylim([0, 0.8])
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        # time slots 2
        fg = plt.figure()
        non_used = y + backhaul - x
        r1 = np.arange(1, x.shape[0] + 1)
        plt.bar(r1, x, width=barWidth, edgecolor='black', label='UE allocations', hatch='//')
        plt.bar(r1, non_used, width=barWidth, edgecolor='black', color='grey', label='Not used', hatch='\\', bottom=x)
        plt.xlabel('Timeslots')
        plt.ylabel('Value')
        ax = fg.gca()
        plt.legend(loc='best')
        plt.ylim([0, 0.8])
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        plt.show()


if __name__ == "__main__":

    os.chdir("..")
    directory = os.path.abspath(os.curdir)
    plt.style.use(directory + '\\post_processing\\YS_plot_style.mplstyle')

    n_IAB = 1
    n_UE = 30
    B_backhaul = 400e6
    B_access = 400e6
    ue_pos = np.zeros([n_UE, 3])
    x = np.random.uniform(low=0.0, high=gl.cell_radius_m, size=n_UE)
    y = np.random.uniform(low=0.0, high=gl.cell_radius_m, size=n_UE)
    ue_pos[:,0] = x
    ue_pos[:,1] = y
    ue_pos[:,2] = np.ones(n_UE)*1.5
    ue_bs = ue_pos[0:10]
    ue_iab = ue_pos[10::]
    st.closest_bs_indices[10::] = np.ones([20,1])

    iab_pos = np.array([[0, 0, 20]])
    bs_pos = np.array([[100, 100, 20]])

    params = OptimizationParams(ue_pos, ue_bs, ue_iab)
    optimization = Optimization(params)
    s_np_iab = np.empty([ue_iab.shape[0] * iab_pos.shape[0], 0])
    for i in ue_iab:
        for j in iab_pos:
            dist2d = norm(i[0:2] - j[0:2])
            dist3d = norm(i - j)
            se = spectral_efficiency_UMi(dist2d, dist3d, 28e9, 20, 1.5, 33, 400e6)
            s_np_iab = np.append(s_np_iab, [se])

    s_np_bs = np.empty([ue_bs.shape[0] * (iab_pos.shape[0] + 1), 0])
    for i in ue_bs:
        for j in bs_pos:
            dist3d = norm(i - j)
            se = spectral_efficiency_UMa(dist3d, 28e9, 30, 1.5, 33, 400e6)
            s_np_bs = np.append(s_np_bs, [se])

    s_m = np.array([7.5])
    optimization_output = optimization.optimize_single_link(s_np_iab, s_np_iab,
                                                              s_np_bs, s_np_bs, s_m, s_m)
    optimization.plot_allocations(optimization_output)

