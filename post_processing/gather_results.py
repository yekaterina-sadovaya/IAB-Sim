from stat_container import st
from gl_vars import gl

import numpy as np


def combine_metrics(link_scheduler, tic, TPT, delay, num_active_UE, number_of_hops_DL):
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
    if st.simulation_time_s >= 2 and tic % gl.channel_update_periodicity_tics == 0:
        TPT['DL'] = np.append(TPT['DL'], st.mean_throughput['DL'])
        TPT['UL'] = np.append(TPT['UL'], st.mean_throughput['UL'])
        delay['DL'] = np.append(delay['DL'], st.mean_delay['DL'])
        delay['UL'] = np.append(delay['UL'], st.mean_delay['UL'])

    st.mean_throughput['DL'] = np.array([])
    st.mean_throughput['UL'] = np.array([])
    st.mean_delay['DL'] = np.array([])
    st.mean_delay['UL'] = np.array([])

    return TPT, delay, num_active_UE, number_of_hops_DL