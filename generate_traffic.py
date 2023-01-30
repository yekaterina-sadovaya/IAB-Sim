from gl_vars import gl
import numpy as np
import matplotlib.pyplot as plt
from stat_container import st


def ftp3_traffic(LOG_DIR, tic, seed):
    np.random.seed(seed)

    if LOG_DIR == 'DL':
        session_intensity = gl.FTP_parameter_lambda_DL
    elif LOG_DIR == 'UL':
        session_intensity = gl.FTP_parameter_lambda_UL
    else:
        print("Logical directions were specified incorrectly.")
        raise ValueError

    # if gl.traffic == 'FTP':
    #     inds = np.where(st.ue_associated_traffic_bytes[LOG_DIR][:, 1] < st.simulation_time_s)
    #     inds = inds[0]
    # else:
    #     inds = range(0, gl.n_UEs)
    inds = np.where(st.ue_associated_traffic_bytes[LOG_DIR][:, 1] < st.simulation_time_s)
    inds = inds[0]

    for id in inds:
        if tic == 0:
            st.ue_associated_traffic_bytes[LOG_DIR][id, 1] = st.ue_associated_traffic_bytes[LOG_DIR][id, 1] + \
                                                             np.random.exponential(1 / session_intensity)
        else:
            node_number = st.closest_bs_indices[id][0]
            if node_number == 0:
                node_name = 'D'
                LINK = 'AC'
            else:
                if LOG_DIR == 'DL':
                    node_name = 'D'
                    LINK = 'BH'
                else:
                    node_name = 'I' + str(node_number)
                    LINK = 'AC'

            if len(st.packet_traffic[node_name][LINK][LOG_DIR][id]) <= 5000:
                st.ue_associated_traffic_bytes[LOG_DIR][id, 0] = st.ue_associated_traffic_bytes[LOG_DIR][id, 0]\
                                                                 + gl.file_size_bytes
                st.ue_associated_traffic_bytes[LOG_DIR][id, 1] = st.ue_associated_traffic_bytes[LOG_DIR][id, 1] +\
                                                                 np.random.exponential(1 / session_intensity)


if __name__ == '__main__':

    session_arrival_times = []
    for t in range(1, 600):
        st.simulation_time_s = t / 10
        ftp3_traffic('DL', t, 1)
        session_arrival_times.append(st.ue_associated_traffic_bytes['DL'][0, 1])

    plt.plot([0, 70], [0, 0], color='blue')
    for point in session_arrival_times:
        plt.vlines(session_arrival_times, 0, gl.file_size_bytes / 1e6, color='blue')
    plt.grid()
    plt.xlabel('time, s')
    plt.ylabel('file size, Mbytes')
    plt.show()
