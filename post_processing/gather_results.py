from library.stat_container import st
from gl_vars import gl

import numpy as np
import pickle


def combine_metrics(link_scheduler, tic, TPT, delay, num_active_UE, number_of_hops_DL):
    """
    Updates and gather statistics
    """

    if tic % 1000 == 0 and gl.print_Flag is True:
        print('Time: ' + str(st.simulation_time_s) + '; Tic: ' + str(tic))
        if link_scheduler.active_ues['DL']:
            num_active_UE['DL'] = np.append(num_active_UE['DL'],
                                            len(link_scheduler.active_ues['DL']))
        elif link_scheduler.active_ues['UL']:
            num_active_UE['UL'] = np.append(num_active_UE['UL'],
                                            len(link_scheduler.active_ues['UL']))

        if any(st.perceived_throughput['DL']):
            for u in link_scheduler.active_ues['DL']:
                number_of_hops_DL = np.append(number_of_hops_DL, len(st.backhaul_routes[u]['DL']) - 1)
            print('Mean UE perc. throughput in ' + 'DL (' + str(gl.FTP_parameter_lambda_DL) + '): ' +
                  str(np.mean(st.perceived_throughput['DL']) / 1e6) + ' Mbps')

    st.actual_throughput = {'DL': {k: np.array([]) for k in range(gl.n_UEs)},
                            'UL': {k: np.array([]) for k in range(gl.n_UEs)}}
    if st.simulation_time_s >= 1 and tic % gl.channel_update_periodicity_tics == 0:
        TPT['DL'] = np.append(TPT['DL'], st.mean_throughput['DL'])
        TPT['UL'] = np.append(TPT['UL'], st.mean_throughput['UL'])
        delay['DL'] = np.append(delay['DL'], st.mean_delay['DL'])
        delay['UL'] = np.append(delay['UL'], st.mean_delay['UL'])

    st.mean_throughput['DL'] = np.array([])
    st.mean_throughput['UL'] = np.array([])
    st.mean_delay['DL'] = np.array([])
    st.mean_delay['UL'] = np.array([])

    return TPT, delay, num_active_UE, number_of_hops_DL


def save_data(per_packet_throughput, packet_delay, num_active_UEs, number_of_hops_DL):
    """
    Saves calculation results
    """

    print('saving data...')

    if gl.frame_division_policy == '50/50':
        frame_coeff = '50'
    else:
        frame_coeff = gl.frame_division_policy

    data = dict(throughput_per_packet_DL=per_packet_throughput['DL'],
                throughput_per_packet_UL=per_packet_throughput['UL'],
                throughput_per_burst_DL=st.perceived_throughput['DL'],
                throughput_per_burst_UL=st.perceived_throughput['UL'],
                time_pt=st.time_pt_calculated,
                delay_DL=packet_delay['DL'],
                delay_UL=packet_delay['UL'],
                number_of_hops_DL=number_of_hops_DL,
                active_ues_DL=num_active_UEs['DL'],
                active_ues_UL=num_active_UEs['UL'],
                optimal_rate=st.optimal_throughput)

    if gl.frame_division_policy == 'OPT':
        if gl.use_average is True:
            channel_avg = '_AVG_'
        else:
            channel_avg = '_NOAVG_'

        # save true optimal continuous values
        data_opt = dict(sim_time_s=st.simulation_time_s,
                        opt_weights=st.optimal_weights,
                        trans_time=st.time_transmitted,
                        assoc_points=st.closest_bs_indices)
        with open(frame_coeff + channel_avg + str(gl.SIM_SEED) + '.pickle', 'wb') as handle:
            pickle.dump(data_opt, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open('achieved_' + str(gl.SIM_SEED) + '.pickle', 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        if gl.traffic_type == 'full':
            file_name = 'res_' + frame_coeff + '_FB_' + str(gl.SIM_SEED) + '.pickle'
        else:
            file_name = 'res_' + frame_coeff + '_' + str(gl.FTP_parameter_lambda_UL) + '_' + \
                        str(gl.FTP_parameter_lambda_DL) + '_' + str(gl.SIM_SEED) + '.pickle'
        with open(file_name, 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def calc_throughput(link_scheduler):
    """
    Computes mean throughput for all packets
    """

    THP_av = []
    for i in st.actual_throughput[link_scheduler.current_state]:
        if st.actual_throughput[link_scheduler.current_state][i].any():

            N = st.actual_throughput[link_scheduler.current_state][i].size
            per_packet_tp = np.sum(st.actual_throughput[link_scheduler.current_state][i])/N
            st.actual_throughput[link_scheduler.current_state][i] = per_packet_tp
            THP_av.append(st.actual_throughput[link_scheduler.current_state][i])

    if THP_av:
        st.mean_throughput[link_scheduler.current_state] = np.append(st.mean_throughput[link_scheduler.current_state],
                                                                     np.mean(THP_av))
