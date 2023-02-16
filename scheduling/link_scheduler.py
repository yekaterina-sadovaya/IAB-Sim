from gl_vars import gl
from stat_container import st
from phy.bercurve import set_params_PHY
from fb_optimization.optimization import Optimization, OptimizationParams
from library.calc_params import insert_zero_coefficients

import os
import matplotlib.pyplot as plt
import numpy as np
directory = os.getcwd()
plt.style.use(directory + '\\post_processing\\YS_plot_style.mplstyle')


class LinkScheduler:
    """
    Non-optimal link scheduler
    """
    def __init__(self):
        self.current_state = 1
        self.C = {1: 0.5, 2: 0, 3: 0, 4: 0.5}
        self.active_ues = {}

    def divide_frame(self, *args):
        """
        Computes frame division coefficients
        """
        active_UEs_DL = st.active_UEs['DL']
        active_UEs_UL = st.active_UEs['UL']
        self.active_ues['DL'] = active_UEs_DL
        self.active_ues['UL'] = active_UEs_UL
        if gl.frame_division_policy == 'PF':
            if (active_UEs_UL.__len__() + active_UEs_DL.__len__()) != 0:
                self.C[1] = active_UEs_DL.__len__() / (active_UEs_UL.__len__() + active_UEs_DL.__len__())
                self.C[4] = active_UEs_UL.__len__() / (active_UEs_UL.__len__() + active_UEs_DL.__len__())


class LinkSchedulerOptimal(LinkScheduler):
    """
    Optimal link scheduler
    """
    def __init__(self):
        LinkScheduler.__init__(self)
        self.DgNB_OPT_BH_coefficients = [0.25, 0, 0, 0.25]
        self.C = {1: 0.25, 2: 0.25, 3: 0.25, 4: 0.25}

    def divide_frame(self, *args):
        """
        Computes frame division coefficients using fb_optimization
        """

        # when using fb_optimization + simulator, the traffic must be symmetric, i.e. the same UEs
        # are active at a time
        self.active_ues['DL'] = st.active_UEs['DL']
        self.active_ues['UL'] = st.active_UEs['UL']

        if st.simulation_time_tics == 0:
            UE_positions_tr = args[0]
            PL_DgNB_IAB = args[1]
            BER_CURVES = args[2]
            ue_pos = UE_positions_tr

            ues_belong_to_iab_nodes, ues_belong_to_DgNB = [], []
            spect_eff_ue_iab_DL, spect_eff_ue_bs_DL = np.array([]), np.array([])
            spect_eff_ue_iab_UL, spect_eff_ue_bs_UL = np.array([]), np.array([])
            for ue_i in range(0, gl.n_UEs):
                if st.closest_bs_indices[ue_i] != 0:
                    ues_belong_to_iab_nodes.append(ue_i)
                    node_name = 'I' + str(st.closest_bs_indices[ue_i][0])
                    if gl.use_average is True:
                        spect_eff_ue_iab_DL = np.append(spect_eff_ue_iab_DL,
                                                        [st.PHY_params_average[node_name]['DL'][ue_i].mod_order *
                                                         st.PHY_params_average[node_name]['DL'][ue_i].code_rate])
                        spect_eff_ue_iab_UL = np.append(spect_eff_ue_iab_UL,
                                                    [st.PHY_params_average[node_name]['UL'][ue_i].mod_order *
                                                     st.PHY_params_average[node_name]['UL'][ue_i].code_rate])
                    else:
                        spect_eff_ue_iab_DL = np.append(spect_eff_ue_iab_DL,
                                                        [st.PHY_params[node_name]['DL'][ue_i].mod_order *
                                                         st.PHY_params[node_name]['DL'][ue_i].code_rate])
                        spect_eff_ue_iab_UL = np.append(spect_eff_ue_iab_UL,
                                                        [st.PHY_params[node_name]['UL'][ue_i].mod_order *
                                                         st.PHY_params[node_name]['UL'][ue_i].code_rate])
                else:
                    ues_belong_to_DgNB.append(ue_i)
                    node_name = 'D'
                    if gl.use_average is True:
                        spect_eff_ue_bs_DL = np.append(spect_eff_ue_bs_DL,
                                                       st.PHY_params_average[node_name]['DL'][ue_i].mod_order *
                                                       st.PHY_params_average[node_name]['DL'][ue_i].code_rate)
                        spect_eff_ue_bs_UL = np.append(spect_eff_ue_bs_UL,
                                                       st.PHY_params_average[node_name]['UL'][ue_i].mod_order *
                                                       st.PHY_params_average[node_name]['UL'][ue_i].code_rate)
                    else:
                        spect_eff_ue_bs_DL = np.append(spect_eff_ue_bs_DL,
                                                       st.PHY_params[node_name]['DL'][ue_i].mod_order *
                                                       st.PHY_params[node_name]['DL'][ue_i].code_rate)
                        spect_eff_ue_bs_UL = np.append(spect_eff_ue_bs_UL,
                                                       st.PHY_params[node_name]['UL'][ue_i].mod_order *
                                                       st.PHY_params[node_name]['UL'][ue_i].code_rate)

            spect_eff_DgNB_IAB = np.array([])
            for iab_i in range(0, gl.n_IAB):
                RSRP = gl.DgNB_tx_power_dBm - PL_DgNB_IAB[iab_i] - st.noise_power_DL - gl.interference_margin_dB
                params_DgNB_IAB = set_params_PHY(RSRP, BER_CURVES)
                spect_eff_DgNB_IAB = np.append(spect_eff_DgNB_IAB, params_DgNB_IAB.mod_order*params_DgNB_IAB.code_rate)

            ue_iab = ue_pos[ues_belong_to_iab_nodes]
            ue_bs = ue_pos[ues_belong_to_DgNB]

            # first, we need to get rid of zeros but they should be accounted when processing the output
            s_np_iab_DL_zero_elemnts = np.where(spect_eff_ue_iab_DL == 0)
            s_np_iab_UL_zero_elemnts = np.where(spect_eff_ue_iab_UL == 0)
            s_np_bs_DL_zero_elemnts = np.where(spect_eff_ue_bs_DL == 0)
            s_np_bs_UL_zero_elemnts = np.where(spect_eff_ue_bs_UL == 0)
            spect_eff_ue_iab_DL = spect_eff_ue_iab_DL[spect_eff_ue_iab_DL != 0]
            spect_eff_ue_iab_UL = spect_eff_ue_iab_UL[spect_eff_ue_iab_UL != 0]
            spect_eff_ue_bs_DL = spect_eff_ue_bs_DL[spect_eff_ue_bs_DL != 0]
            spect_eff_ue_bs_UL = spect_eff_ue_bs_UL[spect_eff_ue_bs_UL != 0]

            params = OptimizationParams(ue_pos, ue_bs, ue_iab)
            optimization = Optimization(params)
            optimization_output = optimization.optimize_single_link(spect_eff_ue_iab_DL, spect_eff_ue_iab_UL,
                                                                    spect_eff_ue_bs_DL, spect_eff_ue_bs_UL,
                                                                    spect_eff_DgNB_IAB, spect_eff_DgNB_IAB)

            if optimization_output is not None:

                h = optimization_output[0]
                eps = optimization_output[1]
                y = optimization_output[2]
                x = optimization_output[3]
                backhaul = optimization_output[4]

                if gl.plot_Flag is True:
                    optimization.plot_allocations(optimization_output)

                y_1 = optimization_output[5][0]
                y_2 = optimization_output[5][1]
                y_3 = optimization_output[5][2]
                y_4 = optimization_output[5][3]

                x_1 = optimization_output[6][0]
                x_2 = optimization_output[6][1]
                x_3 = optimization_output[6][2]
                x_4 = optimization_output[6][3]

                # reordering; including zero spectral efficiencies
                # donor
                y_1, y_3 = insert_zero_coefficients(s_np_bs_DL_zero_elemnts[0], y_1, y_3)
                y_2, y_4 = insert_zero_coefficients(s_np_bs_UL_zero_elemnts[0], y_2, y_4)
                # iab node
                x_1, x_2 = insert_zero_coefficients(s_np_iab_UL_zero_elemnts[0], x_1, x_2)
                x_3, x_4 = insert_zero_coefficients(s_np_iab_DL_zero_elemnts[0], x_3, x_4)

                self.DgNB_OPT_BH_coefficients[0] = backhaul[0]
                self.DgNB_OPT_BH_coefficients[3] = backhaul[3]

                if len(h) != gl.n_UEs*2:
                    N = gl.n_UEs*2 - len(h)
                    h = np.insert(h, np.zeros(N/2), len(h)/2)
                    h = np.insert(h, np.zeros(N/2), len(h))

                self.C[1] = eps[0]
                self.C[2] = eps[1]
                self.C[3] = eps[2]
                self.C[4] = eps[3]

                if len(h) != 0:
                    st.optimal_throughput['DL'] = np.append(st.optimal_throughput['DL'], h[0:gl.n_UEs])
                    st.optimal_throughput['UL'] = np.append(st.optimal_throughput['DL'], h[gl.n_UEs:gl.n_UEs*2])
                    if gl.plot_Flag is True:
                        import matplotlib.pyplot as plt
                        from matplotlib.ticker import MaxNLocator
                        fig, axs = plt.subplots(2, 2)
                        eps = np.around(eps, 2)
                        barWidth = 0.4
                        r1 = np.arange(1, eps.shape[0] + 1)
                        axs[0, 0].bar(r1, eps, width=barWidth, edgecolor='black', align='center', alpha=0.5, color='b')
                        for i in range(eps.shape[0]):
                            axs.flat[0].text(i+1, eps[i], str(eps[i]))
                        axs.flat[0].set(xlabel='Timeslot',ylabel='Timeslot duration')
                        axs.flat[0].set(ylim=[0, 0.8])
                        axs.flat[0].xaxis.set_major_locator(MaxNLocator(integer=True))

                        y_pos = np.arange(1, h.shape[0] + 1)
                        axs[0, 1].bar(y_pos, h, align='center', alpha=0.5)
                        for i in range(h.shape[0]//2):
                            axs.flat[1].text(i+1, h[i], str('D'))
                        for i in range(h.shape[0]//2):
                            axs.flat[1].text(h.shape[0]//2+i+1, h[h.shape[0]//2+i], str('U'))
                        axs.flat[1].set(xlabel='UE data rates',ylabel='$h_{n}, Mbps$')
                        axs.flat[1].xaxis.set_major_locator(MaxNLocator(integer=True))

                        barWidth = 0.4
                        r1 = np.arange(1, y.shape[0] + 1)
                        non_used = eps - (y+backhaul)
                        axs[1, 0].bar(r1, y, width=barWidth, edgecolor='black', label='UE allocations', hatch='//')
                        axs[1, 0].bar(r1, backhaul, width=barWidth, edgecolor='black', label='Backhaul allocations', hatch='\\', bottom=y)
                        axs[1, 0].bar(r1, non_used, width=barWidth, edgecolor='black', color='grey', label='Not used', hatch='\\', bottom=y+backhaul)
                        axs.flat[2].set(xlabel='Timeslots',ylabel='Value')
                        axs.flat[2].set(ylim=[0, 0.8])
                        axs.flat[2].xaxis.set_major_locator(MaxNLocator(integer=True))
                        axs[1, 0].legend()

                        non_used = eps - x
                        r1 = np.arange(1, x.shape[0] + 1)
                        axs[1, 1].bar(r1, x, width=barWidth, edgecolor='black', label='UE allocations', hatch='//')
                        axs[1, 1].bar(r1, non_used, width=barWidth, edgecolor='black', color='grey', label='Not used', hatch='\\', bottom=x)
                        axs.flat[3].set(xlabel='Timeslots',ylabel='Value')
                        axs.flat[3].set(ylim=[0, 0.8])
                        axs.flat[3].xaxis.set_major_locator(MaxNLocator(integer=True))
                        axs[1, 1].legend()

                # gather weighting coefficients
                if len(y) != 0:
                    ue_bs = np.where(st.closest_bs_indices == 0)[0]
                    for ki, k in enumerate(ue_bs):
                        st.optimal_weights['D'][k, 0] = y_1[ki]
                        st.optimal_weights['D'][k, 1] = y_2[ki]
                        st.optimal_weights['D'][k, 2] = y_3[ki]
                        st.optimal_weights['D'][k, 3] = y_4[ki]
                if len(x) != 0:
                    ue_iab = np.where(st.closest_bs_indices != 0)[0]
                    y_b1_per_ue = backhaul[0]/len(ue_iab)
                    y_1b_per_ue = backhaul[3]/len(ue_iab)
                    for ki, k in enumerate(ue_iab):
                        st.optimal_weights['I1'][k, 0] = x_1[ki]
                        st.optimal_weights['I1'][k, 1] = x_2[ki]
                        st.optimal_weights['I1'][k, 2] = x_3[ki]
                        st.optimal_weights['I1'][k, 3] = x_4[ki]

                        st.optimal_weights['D'][k, 0] = y_b1_per_ue
                        st.optimal_weights['D'][k, 3] = y_1b_per_ue