import numpy as np
from stat_container import st
from gl_vars import gl


def calc_throughput(link_scheduler):
    THP_av = []
    for i in st.actual_throughput[link_scheduler.current_state]:
        if st.actual_throughput[link_scheduler.current_state][i].any():

            # transmitted_bits = np.sum(st.actual_throughput[link_scheduler.current_state][i])
            # st.actual_throughput[link_scheduler.current_state][i] = transmitted_bits/gl.frame_duration_s
            # THP_av.append(st.actual_throughput[link_scheduler.current_state][i])

            N = st.actual_throughput[link_scheduler.current_state][i].size
            per_packet_tp = np.sum(st.actual_throughput[link_scheduler.current_state][i])/N
            st.actual_throughput[link_scheduler.current_state][i] = per_packet_tp
            THP_av.append(st.actual_throughput[link_scheduler.current_state][i])

    if THP_av:
        st.mean_throughput[link_scheduler.current_state] = np.append(st.mean_throughput[link_scheduler.current_state],
                                                                 np.mean(THP_av))


def insert_zero_coefficients(s_np_zero_elemnts, vector_1, vector_2):

    if len(s_np_zero_elemnts) != 0:
        for zero_index in s_np_zero_elemnts:
            vector_1 = np.insert(vector_1, 0, zero_index)
            vector_2 = np.insert(vector_2, 0, zero_index)

    return vector_1, vector_2
