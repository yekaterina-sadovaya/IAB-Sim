from library.stat_container import st
from gl_vars import gl
from phy.abstractions import set_params_OFDM

import roundrobin
import numpy as np
from math import floor

OFDM_params = set_params_OFDM(gl.numerology_num)
number_of_RBs = floor(gl.bandwidth / OFDM_params.SCS_Hz)

N_BS = gl.n_IAB + 1


def find_multipliers(number):
    possible_dividers = np.array([])
    # check if the number is prime
    for m in range(2, number):
        if number % m == 0:
            possible_dividers = np.append(possible_dividers, m)
    if number == 2:
        possible_dividers = np.array([1, 2])
    return possible_dividers


def RoundRobin_metric(active_ues, time_fraction_number, node_name):
    """
    Recompute the time each user
    was last served
    :param active_ues: UEs, which have traffic in their buffers
    :param time_fraction_number: frame part number
    :param node_name: number (id) of IAB node or donor
    :return: re-ordered list of users
    """

    if len(active_ues) != 0:
        DIR = st.allowed_transmissions[time_fraction_number][node_name]
        RR_metric = np.array([])
        for ue in active_ues:
            current_time = st.simulation_time_s
            last_time_served = st.last_time_served[DIR][ue]
            if last_time_served:
                RR_metric = np.append(RR_metric, current_time - last_time_served)
            else:
                RR_metric = np.append(RR_metric, current_time)

        ids = np.argsort(RR_metric)[::-1]
        active_ues = [active_ues[i] for i in ids]

    return active_ues


def PF_metric(active_ues, symb_total, time_fraction_number, node_name,
              alpha=0.01, betta=1):
    """
    Computes PF metric
    :param active_ues: UEs, which have traffic in their buffers
    :param symb_total: number of available symbols
    :param time_fraction_number: frame part number
    :param node_name: number (id) of IAB node or donor
    :param alpha: prioritization coefficient
    :param betta: prioritization coefficient
    :return: re-ordered list of users
    """

    transmitted_bits_estimate = np.array([])
    if len(active_ues) != 0:
        PF_metric = np.array([])
        DIR = st.allowed_transmissions[time_fraction_number][node_name]
        for ue in active_ues:
            if st.PHY_params[node_name][DIR][ue]:
                spect_eff_i = \
                    st.PHY_params[node_name][DIR][ue].mod_order * \
                    st.PHY_params[node_name][DIR][ue].code_rate
            else:
                spect_eff_i = 0
            estimated_throughput = spect_eff_i * number_of_RBs * symb_total
            transmitted_bits_estimate = np.append(transmitted_bits_estimate, estimated_throughput)
            estimated_throughput = estimated_throughput / 1e-3
            # compute PF metric
            if np.any(st.past_throughput[DIR][ue]):
                st.past_throughput[DIR][ue] = \
                    np.mean(st.past_throughput[DIR][ue])
                m = (estimated_throughput**alpha) / (st.past_throughput[DIR][ue]**betta)
                PF_metric = np.append(PF_metric, m)
            else:
                PF_metric = np.append(PF_metric, 1)

        ids = np.argsort(PF_metric)[::-1]
        active_ues = [active_ues[i] for i in ids]
    return active_ues, transmitted_bits_estimate


def WPF_metric(active_ues, symb_total, time_fraction_number, node_name, coefficient_eps,
               alpha=0.01, betta=1):
    """
    WPF is used in a combination with optimization,
    PF metric is scaled by the optimal coefficients
    """

    if len(active_ues) != 0:
        WPF_metric = np.array([])
        DIR = st.allowed_transmissions[time_fraction_number][node_name]
        max_w = np.max(st.optimal_weights[node_name])
        for ue in active_ues:
            # estimate achieved throughput
            if st.PHY_params[node_name][DIR][ue]:
                spect_eff_i = \
                    st.PHY_params[node_name][DIR][ue].mod_order * \
                    st.PHY_params[node_name][DIR][ue].code_rate
            else:
                spect_eff_i = 0
            # estimated_throughput = (spect_eff_i * number_of_RBs * symb_total)/st.simulation_time_s
            estimated_throughput = spect_eff_i * number_of_RBs * symb_total
            # account for optimal weights
            weight = st.optimal_weights[node_name][ue, time_fraction_number-1]
            # normalize
            # weight = weight/max_w
            weight = weight/coefficient_eps
            # compute PF metric
            if np.any(st.past_throughput[DIR][ue]):
                if gl.traffic_type == 'FTP':
                    st.past_throughput[DIR][ue] = \
                        np.mean(st.past_throughput[DIR][ue])
                WPF_i = (estimated_throughput**alpha) / (st.past_throughput[DIR][ue]**betta)
                WPF_i = WPF_i * weight
                WPF_metric = np.append(WPF_metric, WPF_i)
            else:
                WPF_metric = np.append(WPF_metric, 1e8)

        ids = np.argsort(WPF_metric)[::-1]
        active_ues = [active_ues[i] for i in ids]
    return active_ues


def WFQ_metric(active_ues, coefficient_eps, time_fraction_number, node_name):
    """
    WFQ is used in a combination with optimization,
    RR metric is scaled by the optimal coefficients
    """

    DIR = st.allowed_transmissions[time_fraction_number][node_name]
    max_w = np.max(st.optimal_weights[node_name])
    # Weighted Fair Queuing initializes RR metric with the weights,
    # which are computed via fb_optimization
    if len(active_ues) != 0:

        current_time = st.simulation_time_s
        WFQ_metric = np.array([])
        for ue in active_ues:

            weight = st.optimal_weights[node_name][ue, time_fraction_number-1]
            weight = weight/coefficient_eps
            # normalize
            # weight = weight/max_w
            last_time_served = st.last_time_served[DIR][ue]

            if last_time_served:
                WFQ_metric = np.append(WFQ_metric, (current_time - last_time_served)*weight*1e5)
                # WFQ_metric = np.append(WFQ_metric, 1/(st.packets_counter[DIR][ue]))
            else:
                WFQ_metric = np.append(WFQ_metric, 1e8)
                # WFQ_metric = np.append(WFQ_metric, current_time*weight)

        ids = np.argsort(WFQ_metric)[::-1]
        active_ues = [active_ues[i] for i in ids]

    return active_ues


class Scheduler:
    """
    Scheduler class
    """
    def __init__(self):
        self.schedules = {'I' + str(k1): {'DL': [], 'UL': []} for k1 in range(1, gl.n_IAB + 1)}
        self.schedules['D'] = {'DL': [], 'UL': []}

    def run_scheduler(self, link_scheduler, topology, UE_positions):

        if np.any(link_scheduler.active_ues['DL']) or np.any(link_scheduler.active_ues['UL']):

            current_slot_fraction = link_scheduler.current_state
            C = link_scheduler.C[current_slot_fraction]
            if gl.frame_division_policy == 'OPT':
                self.BH_coefficients = link_scheduler.DgNB_OPT_BH_coefficients
            self.create_schedules(link_scheduler, C, current_slot_fraction)
            self.allocate_resources(link_scheduler, current_slot_fraction, topology, UE_positions)

    def create_schedules(self, link_scheduler, C, time_fraction_number):

        # empty schedules first
        for node_name in st.node_names:
            DIR = st.allowed_transmissions[time_fraction_number][node_name]
            self.schedules[node_name][DIR] = []

            active_ues = link_scheduler.active_ues[DIR]
            time_slots_total = (C * gl.FRAME_DURATION_S) / OFDM_params.RB_time_s
            time_slots_total = np.round(time_slots_total)
            symbols_total = time_slots_total * 14
            st.symbols_per_ue[time_fraction_number] = symbols_total

            if gl.scheduler == 'RR':
                active_ues = RoundRobin_metric(active_ues, time_fraction_number, node_name)
            elif gl.scheduler == 'PF':
                active_ues, bits_estimate = PF_metric(active_ues, symbols_total, time_fraction_number, node_name)
            elif gl.scheduler == 'WFQ':
                active_ues = WFQ_metric(active_ues, C, time_fraction_number, node_name)
            elif gl.scheduler == 'WPF':
                active_ues = WPF_metric(active_ues, symbols_total, time_fraction_number, node_name, C)
            else:
                raise ValueError

            if len(active_ues) != 0:

                for ue_to_schedule in active_ues:

                    if time_fraction_number == 2 or time_fraction_number == 3:
                        LINK = 'AC'
                        st.allowed_links[time_fraction_number][node_name] = LINK
                        IfTraffic = len(st.packet_traffic[node_name][LINK][DIR][ue_to_schedule]) != 0
                        if IfTraffic and len(self.schedules[node_name][DIR]) == 0:
                            self.schedules[node_name][DIR] = [[ue_to_schedule]]

                    else:
                        if len(self.schedules[node_name][DIR]) == 0:
                            if st.BH_counter[time_fraction_number][0] > 0:
                                proportion_BH =\
                                    st.BH_counter[time_fraction_number][1]/st.BH_counter[time_fraction_number][0]
                            else:
                                proportion_BH = 0

                            if gl.frame_division_policy == 'OPT':
                                fraction_BH = self.BH_coefficients[time_fraction_number -
                                                                   1]/link_scheduler.C[time_fraction_number]
                                Cs = link_scheduler.C[time_fraction_number]
                                Ns = st.symbols_per_ue[time_fraction_number]
                                Cs_sim = Ns/14
                                difference = (Cs_sim - Cs)/Cs
                            else:
                                fraction_BH = 0.5

                            if proportion_BH < fraction_BH: #+difference+0.15:    # np.sum(self.BH_coefficients)
                                if node_name == 'D':
                                    LINK = 'BH'
                                else:
                                    LINK = 'AC'
                                st.allowed_links[time_fraction_number][node_name] = LINK
                                IfTraffic = len(st.packet_traffic[node_name][LINK][DIR][ue_to_schedule]) != 0
                                if IfTraffic:
                                    self.schedules[node_name][DIR] = [[ue_to_schedule]]
                                    if node_name == 'D':
                                        st.BH_counter[time_fraction_number][1] = st.BH_counter[time_fraction_number][1] + 1
                                        st.BH_counter[time_fraction_number][0] = st.BH_counter[time_fraction_number][0] + 1

                            else:
                                LINK = 'AC'
                                st.allowed_links[time_fraction_number][node_name] = LINK
                                IfTraffic = len(st.packet_traffic[node_name][LINK][DIR][ue_to_schedule]) != 0
                                if IfTraffic:
                                    self.schedules[node_name][DIR] = [[ue_to_schedule]]
                                    if node_name == 'D':
                                        st.BH_counter[time_fraction_number][2] = st.BH_counter[time_fraction_number][2] + 1
                                        st.BH_counter[time_fraction_number][0] = st.BH_counter[time_fraction_number][0] + 1

    def allocate_resources(self, link_scheduler, current_slot_fraction, topology, UE_positions):

        N_symbols = st.symbols_per_ue[current_slot_fraction]

        for node_name in st.node_names:
            DIR = st.allowed_transmissions[current_slot_fraction][node_name]
            LINK = st.allowed_links[current_slot_fraction][node_name]
            UEs_active = link_scheduler.active_ues[DIR]
            N_active = len(UEs_active)
            if N_active != 0:
                for ue_id in UEs_active:
                    IfTraffic = len(st.packet_traffic[node_name][LINK][DIR][ue_id]) != 0
                    if IfTraffic:
                        # print(st.PHY_params[node_name][DIR][ue_id])
                        if not st.PHY_params[node_name][DIR][ue_id]:
                            topology.update_channel(UE_positions[ue_id], ue_id, DIR, LINK)
                        spect_eff_i = \
                            st.PHY_params[node_name][DIR][ue_id].mod_order * \
                            st.PHY_params[node_name][DIR][ue_id].code_rate

                        st.N_info[node_name][DIR][ue_id, 0] = N_symbols * spect_eff_i * number_of_RBs

    def define_allowed_transmissions(self):
        for n1 in range(0, N_BS):
            if n1 == 0:
                s = 'D'
                st.allowed_transmissions[1][s] = 'DL'
                st.allowed_transmissions[4][s] = 'UL'

                st.allowed_transmissions[2][s] = 'UL'
                st.allowed_transmissions[3][s] = 'DL'

            else:
                s = 'I' + str(n1)
                st.allowed_transmissions[1][s] = 'UL'
                st.allowed_transmissions[4][s] = 'DL'

                st.allowed_transmissions[2][s] = 'UL'
                st.allowed_transmissions[3][s] = 'DL'

