from gl_vars import gl
from stat_container import st
from channel.calculations import calc_transmission
from channel.propagation import time_blocked
from phy.bercurve import set_params_PHY

import numpy as np
import networkx as nx
import random
from math import log10, floor


class TopologyCreator:
    def __init__(self, BERs):
        self.bs_positions = np.vstack((gl.DgNB_pos, gl.IAB_pos))
        self.DgNB_IAB_distances = np.linalg.norm(self.bs_positions[0, :] - self.bs_positions[1:, :], axis=1)
        self.PL_bw_IAB = np.zeros([gl.IAB_pos.shape[0], gl.IAB_pos.shape[0]])
        self.backhaul_rsrp = {'DL': {}, 'UL': {}}
        self.BERs = BERs

    def backhaul_routing(self, ue_position):
        n_bs = self.bs_positions[:, 0:3].shape[0]
        pos = list(self.bs_positions[:, 0:3])

        G = nx.complete_graph(n_bs + 1)
        pos.append(ue_position)

        paths, dist_weights, rsrps_UL, rsrps_DL = ([] for i in range(4))
        for path in nx.all_simple_paths(G, source=n_bs, target=0):
            paths.append(path)
            weight, rsrp_UL, rsrp_DL = ([] for i in range(3))
            for i, el in enumerate(path):
                try:
                    d = np.linalg.norm(pos[el] - pos[path[i + 1]])
                    weight.append(d)
                    # if distance is not UE-DgNB
                    if el != n_bs:
                        if path[i + 1] == 0:
                            tx_pow_DL = gl.DgNB_tx_power_dBm
                            pl = self.PL_bw_DgNB_IAB[i-1]
                        else:
                            tx_pow_DL = gl.IAB_tx_power_dBm
                            pl = self.PL_bw_IAB[i, i + 1]
                        rsrp_DL.append(tx_pow_DL - pl - st.noise_power_DL - gl.interference_margin_dB)
                        rsrp_UL.append(gl.IAB_tx_power_dBm - pl - st.noise_power_DL - gl.interference_margin_dB)
                except IndexError:
                    continue
            dist_weights.append(weight)
            rsrps_UL.append(rsrp_UL)
            rsrps_DL.append(rsrp_DL)

        max_values = []
        for weighted_path in dist_weights:
            max_values.append(max(weighted_path))

        if gl.ue_associations == 'best_rsrp_maxmin':
            ind = max_values.index(min(max_values))
        elif gl.ue_associations == 'rand':
            ind = random.randint(0, len(max_values)-1)
        elif gl.ue_associations == 'min_hops':
            ind = 0
        else:
            raise ValueError
        best_path_UL = paths[ind][1:]
        best_path_DL = paths[ind][1:][::-1]
        # n_bs + 1 is an identifier for the end of the DL transmission
        best_path_DL.append(n_bs + 1)
        # for the UL this means the beginning of the transmission from the UE
        best_path_UL.insert(0, n_bs + 1)

        backhaul_paths = {'DL':best_path_DL, 'UL':best_path_UL}

        self.backhaul_rsrp['DL'][tuple(backhaul_paths['DL'])] = rsrps_DL[ind][::-1]
        self.backhaul_rsrp['UL'][tuple(backhaul_paths['UL'])] = rsrps_UL[ind]

        return backhaul_paths

    def determine_initial_associations(self, ue_positions):
        """This method calculates association points for each EU
        if they are not given as an input parameters"""

        self.PL_bw_DgNB_IAB = calc_transmission(self.bs_positions[1:, :], self.bs_positions[0],
                                                self.DgNB_IAB_distances, np.zeros([self.bs_positions[1:, :].shape[0],
                                                                                   2]),
                                                Nant_tx=gl.N_antenna_elements_DgNB,
                                                Nant_rx=gl.N_antenna_elements_IAB)
        for i_iab, iab_pos in enumerate(gl.IAB_pos):

            current_ids = np.array(range(gl.IAB_pos.shape[0])) != i_iab
            dist_bw_nodes = np.linalg.norm(iab_pos - gl.IAB_pos[current_ids, :], axis=1)
            self.PL_bw_IAB[i_iab, current_ids] = calc_transmission(gl.IAB_pos[current_ids, :], iab_pos,
                                                 dist_bw_nodes, np.zeros([gl.IAB_pos[1:, :].shape[0], 2]),
                                                 Nant_tx=gl.N_antenna_elements_IAB,
                                                 Nant_rx=gl.N_antenna_elements_IAB)

        for i, ue_pos in enumerate(ue_positions):

            best_path = self.backhaul_routing(ue_pos)
            st.backhaul_routes[i] = best_path

            for j in range(1, gl.multi_con_degree + 1):

                st.closest_bs_indices[i, j - 1] = best_path['UL'][1]
                st.closest_bs_distances[i, j - 1] = np.linalg.norm(ue_pos - self.bs_positions[best_path['UL'][1], 0:3])

                st.UE_blockage_condition[i, 2 * j - 1] = time_blocked(st.UE_blockage_condition[i, j - 1],
                                                                      gl.blockers_density,
                                                                      st.closest_bs_distances[i, j - 1],
                                                                      gl.UE_average_speed)
                if st.closest_bs_indices[i, j - 1] == 0:
                    p_tx = gl.DgNB_tx_power_dBm
                    n_ant = gl.N_antenna_elements_DgNB
                    node_name = 'D'
                else:
                    p_tx = gl.IAB_tx_power_dBm
                    n_ant = gl.N_antenna_elements_IAB
                    node_name = 'I' + str(st.closest_bs_indices[i, j - 1])
                    # if gl.num_interfaces_IAB > 1:
                    #     node_name = node_name + '_1'

                PL, PL_average = calc_transmission([ue_pos], self.bs_positions[st.closest_bs_indices[i, j - 1]],
                                       [st.closest_bs_distances[i, j - 1]], [st.UE_blockage_condition[i, :2]],
                                       Nant_tx=gl.N_antenna_elements_UE, Nant_rx=n_ant)
                st.PL_in_all_links[node_name][i, j - 1] = PL
                st.current_rsrp[node_name]['UL'][i, j - 1] = gl.UE_tx_power_dBm - PL - st.noise_power_UL - \
                                                  gl.interference_margin_dB
                st.PHY_params[node_name]['UL'][i] = set_params_PHY(st.current_rsrp[node_name]['UL'][i, j - 1], self.BERs)
                st.current_rsrp[node_name]['DL'][i, j - 1] = p_tx - PL - st.noise_power_DL - gl.interference_margin_dB
                st.PHY_params[node_name]['DL'][i] = set_params_PHY(st.current_rsrp[node_name]['DL'][i, j - 1], self.BERs)
                if gl.use_average is True:
                    st.current_rsrp_average[node_name]['UL'][i, j - 1] = gl.UE_tx_power_dBm - PL_average - \
                                                                          st.noise_power_UL - gl.interference_margin_dB
                    st.PHY_params_average[node_name]['UL'][i] = set_params_PHY(
                        st.current_rsrp_average[node_name]['UL'][i, j - 1], self.BERs)
                    st.current_rsrp_average[node_name]['DL'][i, j - 1] = p_tx - PL_average - \
                                                                         st.noise_power_DL - gl.interference_margin_dB
                    st.PHY_params_average[node_name]['DL'][i] = set_params_PHY(
                        st.current_rsrp_average[node_name]['DL'][i, j - 1], self.BERs)

    def update_channel(self, ue_pos, ue_num, DIR, LINK):

        if DIR == 'DL':
            if st.closest_bs_indices[ue_num, 0] == 0:
                p_tx = gl.DgNB_tx_power_dBm
                n_ant_tx = gl.N_antenna_elements_DgNB
            else:
                p_tx = gl.IAB_tx_power_dBm
                n_ant_tx = gl.N_antenna_elements_IAB
            noise_power = st.noise_power_DL
            n_ant_rx = gl.N_antenna_elements_UE
        else:
            if st.closest_bs_indices[ue_num, 0] == 0:
                n_ant_rx = gl.N_antenna_elements_DgNB
            else:
                n_ant_rx = gl.N_antenna_elements_IAB
            p_tx = gl.UE_tx_power_dBm
            n_ant_tx = gl.N_antenna_elements_UE
            noise_power = st.noise_power_UL

        if np.any(np.array(st.active_UEs[DIR]) == ue_num):
            for node_name in st.node_names:
                IfTraffic = len(st.packet_traffic[node_name][LINK][DIR][ue_num]) != 0
                if IfTraffic:
                    # PL = calc_transmission([ue_pos], self.bs_positions[st.closest_bs_indices[ue_num, 0]],
                    #                        [st.closest_bs_distances[ue_num, 0]],
                    #                        [st.UE_blockage_condition[ue_num, :2]],
                    #                        Nant_tx=n_ant_tx, Nant_rx=n_ant_rx,
                    #                        diversity=gl.antenna_diversity)
                    current_hop = st.packet_traffic[node_name][LINK][DIR][ue_num][0].current_hop
                    assoc_point = st.closest_bs_indices[ue_num]
                    IfFirstHop = current_hop > gl.n_IAB + 1 or current_hop == assoc_point
                    if IfFirstHop:
                        PL, PL_average = calc_transmission([ue_pos], self.bs_positions[st.closest_bs_indices[ue_num, 0]],
                                               [st.closest_bs_distances[ue_num, 0]],
                                               [st.UE_blockage_condition[ue_num, :2]],
                                               Nant_tx=n_ant_tx, Nant_rx=n_ant_rx)

                    elif current_hop != assoc_point and current_hop != 0:
                        first_index = current_hop
                        node_id = np.where(np.array(st.backhaul_routes[ue_num][DIR]) == current_hop)[0]
                        second_index = st.backhaul_routes[ue_num][DIR][int(node_id) + 1]
                        # we need to subtract one because PL IAB is [n_iab x n_iab]
                        PL = self.PL_bw_IAB[first_index - 1, second_index - 1]
                    elif current_hop != assoc_point and current_hop == 0:
                        ap_ind = st.backhaul_routes[ue_num][DIR][-2] - 1
                        PL = self.PL_bw_DgNB_IAB[ap_ind]
                    else:
                        assert ValueError
                        print('Something went wrong. Next hop, assoc. point:', current_hop, assoc_point)

                    st.PL_in_all_links[node_name][ue_num, 0] = PL
                    st.current_rsrp[node_name][DIR][ue_num, 0] = p_tx - PL - noise_power - gl.interference_margin_dB
                    st.PHY_params[node_name][DIR][ue_num] = set_params_PHY(st.current_rsrp[node_name][DIR][ue_num, 0],
                                                                           self.BERs)
                    if IfFirstHop and gl.use_average is True:
                        st.current_rsrp_average[node_name][DIR][ue_num, 0] = p_tx - PL_average - noise_power - gl.interference_margin_dB
                        st.PHY_params_average[node_name][DIR][ue_num] = set_params_PHY(st.current_rsrp_average[node_name][DIR][ue_num, 0],
                                                                           self.BERs)
