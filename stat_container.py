import numpy as np
from gl_vars import gl


class STAT_container_class:
    def __init__(self):

        self.simulation_time_s = 0
        self.simulation_time_tics = 0
        self.active_UEs = {'DL': [], 'UL': []}
        # for each UE store information about how much data (in bits) corresponds to each UE in DL and UL
        # 1 column: amount of data in bits; 2 column: time when new session will arrive
        self.ue_associated_traffic_bytes = {'DL': np.zeros([gl.n_UEs, 2]), 'UL': np.zeros([gl.n_UEs, 2])}
        keys = ['D']
        # if gl.num_interfaces_IAB > 1:
        #     keys = keys + ['I'+str(n1)+'_'+str(n2) for n1 in range(1, gl.n_IAB+1)
        #                    for n2 in range(1, gl.num_interfaces_IAB+1)]
        # else:
        keys = keys + ['I'+str(n1) for n1 in range(1, gl.n_IAB+1)]
        self.node_names = keys
        self.allowed_transmissions = {k: {key: [] for key in keys} for k in range(1, 5)}
        self.allowed_links = {k: {key: [] for key in keys} for k in range(1, 5)}
        # in this data, DL packets correspond to DgNB buffer; UL packets to UEs; buffers of IAB nodes are
        # realized separately
        self.packet_traffic = {key: {'BH': {'DL': {k: [] for k in range(gl.n_UEs)}, 'UL': {k: [] for k in range(gl.n_UEs)}},
                                     'AC': {'DL': {k: [] for k in range(gl.n_UEs)}, 'UL': {k: [] for k in range(gl.n_UEs)}}}
                               for key in keys}
        # self.packet_traffic = {'DL': {k: [] for k in range(gl.n_UEs)},
        #                        'UL': {k: [] for k in range(gl.n_UEs)}}
        # there is a separate buffer for those packets, which were fragmented. This does not influence the results
        # but rather is made for calculations simplicity
        self.fragmented_packets = {'DL': {k: [] for k in range(gl.n_UEs)}, 'UL': {k: [] for k in range(gl.n_UEs)}}
        # if blocked - first column; duration of blockage - second column
        self.UE_blockage_condition = np.zeros([gl.n_UEs, 2 * gl.multi_con_degree])

        self.noise_power_UL = gl.thermal_noise_density + gl.NF_BS_dB + 10 * np.log10(gl.bandwidth)
        self.noise_power_DL = gl.thermal_noise_density + gl.NF_UE_dB + 10 * np.log10(gl.bandwidth)

        self.closest_bs_distances = np.zeros([gl.n_UEs, gl.multi_con_degree])
        self.closest_bs_indices = np.zeros([gl.n_UEs, gl.multi_con_degree])
        self.closest_bs_indices = self.closest_bs_indices.astype('int32')
        self.next_hop = {'DL': np.zeros([gl.n_UEs, gl.multi_con_degree]),
                         'UL': np.zeros([gl.n_UEs, gl.multi_con_degree])}

        self.current_rsrp = {key: {'DL': np.zeros([gl.n_UEs, gl.multi_con_degree]),
                                   'UL': np.zeros([gl.n_UEs, gl.multi_con_degree])} for key in keys}
        self.rsrp_in_time = {key: {'DL': {k: np.array([]) for k in range(gl.n_UEs)},
                                   'UL': {k: np.array([]) for k in range(gl.n_UEs)}} for key in keys}
        self.PL_in_all_links = {key: np.zeros([gl.n_UEs, gl.multi_con_degree]) for key in keys}

        self.backhaul_routes = {k: {} for k in range(gl.n_UEs)}

        self.resource_allocations = {}

        self.PHY_params = {key: {'DL': {k: np.array([]) for k in range(gl.n_UEs)},
                                 'UL': {k: np.array([]) for k in range(gl.n_UEs)}} for key in keys}
        if gl.use_average is True:
            self.PHY_params_average = {key: {'DL': {k: np.array([]) for k in range(gl.n_UEs)},
                                             'UL': {k: np.array([]) for k in range(gl.n_UEs)}} for key in keys}
            self.current_rsrp_average = {key: {'DL': np.zeros([gl.n_UEs, gl.multi_con_degree]),
                                       'UL': np.zeros([gl.n_UEs, gl.multi_con_degree])} for key in keys}

        self.N_info = {key: {'DL': np.zeros([gl.n_UEs, gl.multi_con_degree]),
                       'UL': np.zeros([gl.n_UEs, gl.multi_con_degree])} for key in keys}

        # self.N_info = {key: {'BH': {'DL': np.zeros([gl.n_UEs, gl.multi_con_degree]),
        #                             'UL': np.zeros([gl.n_UEs, gl.multi_con_degree])},
        #                      'AC': {'DL': np.zeros([gl.n_UEs, gl.multi_con_degree]),
        #                             'UL': np.zeros([gl.n_UEs, gl.multi_con_degree])}}
        #                for key in keys}

        self.TBS = {'DL': np.array([]), 'UL': np.array([])}

        self.actual_throughput = {'DL': {k: np.array([]) for k in range(gl.n_UEs)},
                                  'UL': {k: np.array([]) for k in range(gl.n_UEs)}}

        self.bits_tr = {'DL': {k: 0 for k in range(gl.n_UEs)},
                                  'UL': {k: 0 for k in range(gl.n_UEs)}}

        self.mean_throughput = {'DL': np.array([]), 'UL': np.array([])}
        self.mean_delay = {'DL': np.array([]), 'UL': np.array([])}
        self.perceived_throughput = {'DL': np.array([]), 'UL': np.array([])}
        self.optimal_throughput = {'DL':np.array([]), 'UL':np.array([])}
        self.time_pt_calculated = np.array([])

        self.packets_counter = {'DL': {k: 0 for k in range(gl.n_UEs)},
                                'UL': {k: 0 for k in range(gl.n_UEs)}}

        self.per_packet_delay_per_TTI = {'DL': {k: np.array([]) for k in range(gl.n_UEs)},
                                         'UL': {k: np.array([]) for k in range(gl.n_UEs)}}

        self.per_packet_time_served_in_burst = {'DL': {k: np.array([]) for k in range(gl.n_UEs)},
                                                'UL': {k: np.array([]) for k in range(gl.n_UEs)}}

        # self.symbols_per_ue = {'DL': {k: 0 for k in range(gl.n_UEs)},
        #                        'UL': {k: 0 for k in range(gl.n_UEs)}}     # for packet scheduling
        self.symbols_per_ue = {k: 0 for k in range(1, 5)}   # for packet scheduling

        self.past_throughput = {'DL': {k: np.array([]) for k in range(gl.n_UEs)},
                                'UL': {k: np.array([]) for k in range(gl.n_UEs)}}     # for packet PF scheduling

        self.last_time_served = {'DL': {k: np.array([]) for k in range(gl.n_UEs)},
                                 'UL': {k: np.array([]) for k in range(gl.n_UEs)}}     # for RR scheduling

        self.optimal_weights = {'D': np.zeros([gl.n_UEs, 4]), 'I1': np.zeros([gl.n_UEs, 4])}     # for WFQ scheduling

        self.resource_blocks_per_ue = {'DL': {k: 0 for k in range(gl.n_UEs)},
                                       'UL': {k: 0 for k in range(gl.n_UEs)}}     # for packet scheduling

        self.BH_counter = {1: np.zeros(3), 2: np.zeros(3),
                           3: np.zeros(3), 4: np.zeros(3)}     # counter required to satisfy optimal allocations at the backhaul
                                                                   # format [# timeslots total; # allocated to backhaul]

        self.time_transmitted = {'D':np.zeros([gl.n_UEs, 4]),'I1':np.zeros([gl.n_UEs,4])}



st = STAT_container_class()
