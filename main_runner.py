from mobility.configure_mobility import set_mobility_model
from gl_vars import gl
from stat_container import st
from generate_traffic import ftp3_traffic
from topology_formation import TopologyCreator
from packet.packets_operations import packetize_data, transmit_blocks, calc_phy_throughput_FB
from phy.abstractions import set_params_OFDM
from scheduling.link_scheduler import LinkScheduler, LinkSchedulerOptimal
from scheduling.packet_scheduler import Scheduler
from library.calc_params import calc_throughput
from library.random_drop import drop_DgNB, drop_IAB
from phy.interpolate_ber_curves import load_BER
from channel.antenna import beam_split
from post_processing.gather_results import combine_metrics

import numpy as np
import pickle
import matplotlib.pyplot as plt

plt.style.use('YS_plot_style.mplstyle')

subframe_duration_s = 1e-3


def packet_sim(session_intensity, SIM_SEED):
    gl.FTP_parameter_lambda_DL = session_intensity
    gl.FTP_parameter_lambda_UL = session_intensity

    BER_CURVES = load_BER()

    # Drop the DgNB on the edge
    drop_DgNB(SIM_SEED)
    drop_IAB(SIM_SEED)

    # Calculate parameters, which do not change during the calculations
    OFDM_params = set_params_OFDM(gl.numerology)
    # periodicity = subframe_duration_s / OFDM_params.RB_time_s
    periodicity = gl.time_stop_tics

    UE_mobility_model = set_mobility_model(SIM_SEED, subframe_duration_s)
    topology = TopologyCreator(BER_CURVES)
    if gl.frame_division_policy != 'OPT':
        link_scheduler = LinkScheduler()
    else:
        link_scheduler = LinkSchedulerOptimal()
    packet_scheduler = Scheduler()

    st.__init__()

    # Containers to store data
    TPT = {'DL': np.array([]), 'UL': np.array([])}
    delay = {'DL': np.array([]), 'UL': np.array([])}
    num_active_UEs = {'DL': np.array([]), 'UL': np.array([])}
    number_of_hops_DL = np.array([])

    SEC_COUNT = 0

    for tic in range(0, gl.time_stop_tics):

        if tic % periodicity * 2 == 0:
            UE_positions = next(UE_mobility_model)
            # Transform according to the cell size
            x = UE_positions[:, 0] - gl.cell_radius_m
            y = UE_positions[:, 1] - gl.cell_radius_m
            UE_positions_tr = [x, y, UE_positions[:, 2]]
            UE_positions_tr = np.transpose(UE_positions_tr)

        if tic == 0:
            # generate traffic (demands) for each UE initially
            ftp3_traffic('UL', tic, SIM_SEED)
            ftp3_traffic('DL', tic, SIM_SEED)
            topology.determine_initial_associations(UE_positions_tr)
            if gl.plot_Flag is True:
                indices1 = np.where(st.closest_bs_indices == 0)[0]
                UE_positions_DgNB = UE_positions_tr[indices1, :]
                indices2 = np.where(st.closest_bs_indices == 1)[0]
                UE_positions_IAB = UE_positions_tr[indices2, :]
                plt.plot(UE_positions_DgNB[:, 0], UE_positions_DgNB[:, 1], 'o', label='UEs connected to DgNB',
                         markersize=5, color='k')
                plt.plot(UE_positions_IAB[:, 0], UE_positions_IAB[:, 1], 'o', label='UEs connected to IAB',
                         markersize=5, color='b')
                plt.plot(gl.DgNB_pos[:, 0], gl.DgNB_pos[:, 1], 'o', label='DgNB', markersize=10)
                plt.plot(gl.IAB_pos[:, 0], gl.IAB_pos[:, 1], '^', label='IAB node', markersize=10)
                plt.legend()
                plt.xlabel('x, m')
                plt.ylabel('y, m')
                plt.xticks(fontsize=12)
                plt.yticks(fontsize=12)
                plt.show()
        else:
            # At this stage, traffic flow is converted to packets
            packetize_data('DL')
            packetize_data('UL')

            if gl.frame_division_policy != 'OPT':
                link_scheduler.divide_frame()
            else:
                link_scheduler.divide_frame(UE_positions_tr, topology.PL_bw_DgNB_IAB, BER_CURVES)

            C = link_scheduler.C[link_scheduler.current_state]
            st.simulation_time_tics = tic

            # C is the coefficient for each time duration. There are 4 in general but when
            # optimization framework turned off, only the 1st and 4th slots are used (backhaul is never disabled)
            if C != 0:

                packet_scheduler.define_allowed_transmissions()
                packet_scheduler.run_scheduler(link_scheduler, topology, UE_positions_tr)

                if np.any(link_scheduler.active_ues['DL']) or np.any(link_scheduler.active_ues['UL']):
                    transmit_blocks(link_scheduler, packet_scheduler, OFDM_params, BER_CURVES)
                    # thr_per_subframe = calc_thr_per_subframe(link_scheduler.current_state)

                if any(st.mean_throughput['DL']):
                    # print(link_scheduler.C)
                    st.mean_throughput['DL'] = np.mean(st.mean_throughput['DL'])
                    st.mean_delay['DL'] = np.mean(st.mean_delay['DL'])

                ftp3_traffic('DL', tic, SIM_SEED)
                ftp3_traffic('UL', tic, SIM_SEED)

                TPT, delay, num_active_UEs, number_of_hops_DL = \
                    combine_metrics(link_scheduler, tic, TPT, delay, num_active_UEs, number_of_hops_DL)
                if gl.division_unit == 'slot':
                    st.simulation_time_s = st.simulation_time_s + OFDM_params.RB_time_s * C
                    # st.simulation_time_s = st.simulation_time_s + \
                    # st.symbols_per_ue[link_scheduler.current_state]*OFDM_params.symbol_duration
                elif gl.division_unit == 'subfr':
                    st.simulation_time_s = st.simulation_time_s + subframe_duration_s * C

                if tic % gl.channel_update_periodicity_tics == 0:
                    for UE_num, UE_position in enumerate(UE_positions_tr):
                        topology.update_channel(UE_position, UE_num, 'DL', 'AC')
                        topology.update_channel(UE_position, UE_num, 'DL', 'BH')
                        topology.update_channel(UE_position, UE_num, 'UL', 'AC')
                        topology.update_channel(UE_position, UE_num, 'UL', 'BH')

            if link_scheduler.current_state == 4:
                link_scheduler.current_state = 1
            else:
                link_scheduler.current_state += 1

            # if st.simulation_time_tics % 32000 == 0:
            #     SEC_COUNT = SEC_COUNT + 1
            #     data_opt = dict(sim_time_s=st.simulation_time_s,
            #                     opt_weights=st.optimal_weights,
            #                     trans_time=st.time_transmitted,
            #                     assoc_points=st.closest_bs_indices)
            #     print('save data...')
            #     with open('FB_static2_seed10_'+str(SEC_COUNT)+'s.pickle', 'wb') as handle:
            #         pickle.dump(data_opt, handle, protocol=pickle.HIGHEST_PROTOCOL)

    if gl.traffic == 'full':
        calc_phy_throughput_FB()

    data_opt = dict(sim_time_s=st.simulation_time_s,
                    opt_weights=st.optimal_weights,
                    trans_time=st.time_transmitted,
                    assoc_points=st.closest_bs_indices)
    print('save data...')
    if gl.frame_division_policy == '50/50':
        SF = '50'
    else:
        SF = gl.frame_division_policy
    if gl.scheduler == 'WFQ':
        SCHD = 'RR'
    else:
        SCHD = gl.scheduler
    if gl.use_average is True:
        AVG = '_AVG_'
    else:
        AVG = '_'

    folder = 'C:/Users/sadov/OneDrive/Документы/Работа/Intel/2021 IAB/Packet-sim-git/Data/PF_R'+str(gl.cell_radius_m)+'m_staticUEs_' + SCHD + '/'
    with open(folder + 'alloc_FB_static_' + SF + '_' + gl.scheduler + AVG +str(SIM_SEED)+'_'+str(SEC_COUNT)+'s.pickle', 'wb') as handle:
        pickle.dump(data_opt, handle, protocol=pickle.HIGHEST_PROTOCOL)

    data = dict(throughput_per_packet_DL=TPT['DL'],
                throughput_per_packet_UL=TPT['UL'],
                throughput_per_burst_DL=st.perceived_throughput['DL'],
                throughput_per_burst_UL=st.perceived_throughput['UL'],
                time_pt=st.time_pt_calculated,
                delay_DL=delay['DL'],
                delay_UL=delay['UL'],
                number_of_hops_DL=number_of_hops_DL,
                active_ues_DL=num_active_UEs['DL'],
                active_ues_UL=num_active_UEs['UL'],
                # rsrp_DL=st.rsrp_in_time['DL'],
                # rsrp_UL=st.rsrp_in_time['UL'],
                optimal_rate=st.optimal_throughput)

    with open(folder + str(gl.n_IAB)+'_NODE_' + SF + '_' + gl.scheduler + '_INT' + AVG + str(gl.FTP_parameter_lambda_UL) + '_' +
              str(gl.FTP_parameter_lambda_DL) + '_' + str(SIM_SEED) + '.pickle', 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    # packet_sim(0.1, 8)
    packet_sim(50, 7)
