from packet.form_packets import TDataPacket
from stat_container import st
from gl_vars import gl
from packet.arq_block import ARQ_block

from math import log2
import collections
import numpy as np

N_BS = gl.n_IAB + 1

TBS_SIZE_DEF = 8448

TBS_for_less_than_3824 = np.linspace(24, 192, 22)
TBS_for_less_than_3824 = np.append(TBS_for_less_than_3824, np.linspace(208, 384, 12))
TBS_for_less_than_3824 = np.append(TBS_for_less_than_3824, np.linspace(408, 576, 8))
TBS_for_less_than_3824 = np.append(TBS_for_less_than_3824, np.linspace(608, 1672, 26))
TBS_for_less_than_3824 = np.append(TBS_for_less_than_3824, np.linspace(1736, 2856, 18))
TBS_for_less_than_3824 = np.append(TBS_for_less_than_3824, np.array([2976, 3104, 3240, 3368, 3496, 3624, 3752, 3824]))


def packetize_data(LOG_DIR):
    """
    Converts data flow from traffic model to
    integer number of packets
    """
    st.active_UEs[LOG_DIR] = []
    for id_of_UE in range(0, gl.n_UEs):
        number_of_packets = st.ue_associated_traffic_bytes[LOG_DIR][id_of_UE, 0] / gl.packet_size_bytes
        st.ue_associated_traffic_bytes[LOG_DIR][id_of_UE, 0] = 0
        number_of_packets = int(number_of_packets)
        new_packets = []
        # in case of full buffer, packets should be also generated at intermediate hops to satisfy
        # optimal allocations, i.e., align the assumptions
        additional_packets_fb = []
        if number_of_packets != 0:
            for i in range(0, number_of_packets):
                IP_pkt = TDataPacket(id_of_UE, st.backhaul_routes[id_of_UE][LOG_DIR][-1], gl.packet_size_bytes,
                                     current_hop=st.backhaul_routes[id_of_UE][LOG_DIR][0])

                if i == 0:
                    IP_pkt.first_in_a_burst = True
                if i == number_of_packets - 1:
                    IP_pkt.last_in_a_burst = True

                new_packets.append(IP_pkt)

        if st.closest_bs_indices[id_of_UE][0] == 0 or LOG_DIR == 'DL':
            node_name = 'D'
            if st.closest_bs_indices[id_of_UE][0] == 0:
                LINK = 'AC'
            else:
                LINK = 'BH'
        else:
            node_name = 'I' + str(st.closest_bs_indices[id_of_UE][0])
            LINK = 'AC'

        if number_of_packets != 0:
            st.packet_traffic[node_name][LINK][LOG_DIR][id_of_UE] = st.packet_traffic[node_name][LINK][LOG_DIR][
                                                                  id_of_UE] + new_packets
        for node_name in st.node_names:

            if len(st.packet_traffic[node_name]['AC'][LOG_DIR][id_of_UE]) != 0 or \
                    len(st.packet_traffic[node_name]['BH'][LOG_DIR][id_of_UE]) != 0:
                st.active_UEs[LOG_DIR].append(id_of_UE)

    st.active_UEs[LOG_DIR] = set(st.active_UEs[LOG_DIR])
    st.active_UEs[LOG_DIR] = list(st.active_UEs[LOG_DIR])


def det_block_size(DIR, ue_num, node_name):
    """
    Transport block size determination according
    to 3GPP
    """

    N_info = st.N_info[node_name][DIR][ue_num]
    if N_info > 0:
        if N_info <= 3824:
            n = np.max([3, np.floor(log2(N_info)) - 6])
            N_info_ = np.max([24, (2 ** n) * np.floor(N_info / (2 ** n))])
            d = TBS_for_less_than_3824 - N_info_
            d = abs(d)
            ind = np.where(d == np.min(d))
            if len(ind[0]) > 1:
                ind = ind[0][-1]
            TBS = TBS_for_less_than_3824[ind]
        else:
            n = log2(N_info - 24) - 5
            N_info_ = (2 ** n) * np.round((N_info - 24) / (2 ** n))
            R = st.PHY_params[node_name][DIR][ue_num].code_rate
            if R <= 0.25:
                C = np.ceil((N_info_ + 24) / 3816)
                TBS = 8 * C * np.ceil((N_info_ + 24) / (8 * C)) - 24
            else:
                if N_info_ >= 8424:
                    C = np.ceil((N_info_ + 24) / 8424)
                    TBS = 8 * C * np.ceil((N_info_ + 24) / (8 * C)) - 24
                else:
                    TBS = 8 * np.ceil((N_info_ + 24) / 8) - 24
    else:
        TBS = 0
    return TBS


def deliver_packet_FTP(DIR, ue_in_TTI, pkt_id, OFDM_params, TTI):
    """
    Simulates packet transmission for FRP traffic,
    computes packet parameters
    :param DIR: Logical direction (UL or DL)
    :param ue_in_TTI: UE number
    :param pkt_id: packet number
    :param OFDM_params: OFDM parameters class
    """

    if st.packet_traffic[DIR][ue_in_TTI][pkt_id].current_hop != \
            st.packet_traffic[DIR][ue_in_TTI][pkt_id].destination:

        st.packet_traffic[DIR][ue_in_TTI][pkt_id].current_hop = \
            st.backhaul_routes[ue_in_TTI][DIR][st.packet_traffic[DIR][ue_in_TTI][pkt_id].hops_number]

    else:
        st.last_time_served[DIR][ue_in_TTI] = st.simulation_time_s
        st.packet_traffic[DIR][ue_in_TTI][pkt_id].service_enter_time = \
            st.simulation_time_s + ((OFDM_params.RB_time_s / 14) * st.symbols_per_ue[DIR][ue_in_TTI]) * (TTI + 1)
        delay = st.packet_traffic[DIR][ue_in_TTI][pkt_id].service_enter_time - \
                st.packet_traffic[DIR][ue_in_TTI][pkt_id].arrival_time
        st.per_packet_delay_per_TTI[DIR][ue_in_TTI] = np.append(st.per_packet_delay_per_TTI[DIR][ue_in_TTI], delay)
        bits_transmitted = st.packet_traffic[DIR][ue_in_TTI][pkt_id].size_bytes * 8
        if st.packet_traffic[DIR][ue_in_TTI][pkt_id].first_in_a_burst:
            st.per_packet_time_served_in_burst[DIR][ue_in_TTI] = \
                np.append(st.per_packet_time_served_in_burst[DIR][ue_in_TTI],
                          st.packet_traffic[DIR][ue_in_TTI][pkt_id].arrival_time)
        # calculate throughput before the packet is removed from the buffer
        st.actual_throughput[DIR][ue_in_TTI] = np.append(st.actual_throughput[DIR][ue_in_TTI], bits_transmitted)
        st.packets_counter[DIR][ue_in_TTI] = st.packets_counter[DIR][ue_in_TTI] + 1

        st.bits_tr[DIR][ue_in_TTI] = st.bits_tr[DIR][ue_in_TTI] + 8424

        # calculate UE perceived throughput for the dynamic traffic
        if st.packet_traffic[DIR][ue_in_TTI][pkt_id].last_in_a_burst:
            st.per_packet_time_served_in_burst[DIR][ue_in_TTI] = \
                np.append(st.per_packet_time_served_in_burst[DIR][ue_in_TTI],
                          st.packet_traffic[DIR][ue_in_TTI][pkt_id].service_enter_time)
            t_del = st.per_packet_time_served_in_burst[DIR][ue_in_TTI][1] - \
                    st.per_packet_time_served_in_burst[DIR][ue_in_TTI][0]
            p_t = (((st.packets_counter[DIR][ue_in_TTI] - 1) * 8424) / t_del)
            st.perceived_throughput[DIR] = np.append(st.perceived_throughput[DIR], p_t)
            st.time_pt_calculated = np.append(st.time_pt_calculated, st.simulation_time_s)
            if gl.scheduler == 'PF' or gl.scheduler == 'WPF':
                st.past_throughput[DIR][ue_in_TTI] = np.append(st.past_throughput[DIR][ue_in_TTI], p_t)
            st.per_packet_time_served_in_burst[DIR][ue_in_TTI] = np.array([])

            st.packets_counter[DIR][ue_in_TTI] = 0

        del st.packet_traffic[DIR][ue_in_TTI][pkt_id]


def deliver_packet_FB(DIR, ue_in_TTI, pkt_id, OFDM_params, node_name, TTI, scheduled_direction, LINK):
    """
    Simulates packet transmission for full-buffer traffic,
    computes packet parameters
    """

    st.packet_traffic[node_name][LINK][DIR][ue_in_TTI][pkt_id].hops_number = \
        st.packet_traffic[node_name][LINK][DIR][ue_in_TTI][pkt_id].hops_number + 1
    current_hop_index = st.packet_traffic[node_name][LINK][DIR][ue_in_TTI][pkt_id].hops_number
    current_hop = st.backhaul_routes[ue_in_TTI][DIR][current_hop_index]
    st.packet_traffic[node_name][LINK][DIR][ue_in_TTI][pkt_id].current_hop = current_hop

    if current_hop != st.packet_traffic[node_name][LINK][DIR][ue_in_TTI][pkt_id].destination:

        next_hop = st.backhaul_routes[ue_in_TTI][DIR][current_hop_index + 1]

        if len(st.backhaul_routes[ue_in_TTI][DIR]) == 2 or next_hop == 0:
            next_node_name = 'D'
            if len(st.backhaul_routes[ue_in_TTI][DIR]) == 2:
                NEXT_LINK = 'AC'
            else:
                NEXT_LINK = 'BH'

        elif next_hop == N_BS + 1:
            next_node_name = 'I' + str(current_hop)
            NEXT_LINK = 'AC'
        else:
            next_node_name = 'I' + str(next_hop)
            NEXT_LINK = 'AC'

        if node_name == 'I1' and next_hop == 1 and DIR == 'UL':
            next_node_name = 'D'
            NEXT_LINK = 'BH'

        st.packet_traffic[next_node_name][NEXT_LINK][DIR][ue_in_TTI].append(st.packet_traffic[node_name][LINK][DIR][ue_in_TTI][pkt_id])
        del st.packet_traffic[node_name][LINK][DIR][ue_in_TTI][pkt_id]

    else:

        st.packet_traffic[node_name][LINK][DIR][ue_in_TTI][pkt_id].service_enter_time = \
            st.simulation_time_s + ((OFDM_params.RB_time_s / 14) * st.symbols_per_ue[scheduled_direction]) * (TTI + 1)
        st.packets_counter[DIR][ue_in_TTI] = st.packets_counter[DIR][ue_in_TTI] + 1
        st.bits_tr[DIR][ue_in_TTI] = st.bits_tr[DIR][ue_in_TTI] + 8424
        st.last_time_served[DIR][ue_in_TTI] = st.simulation_time_s

        if gl.scheduler == 'PF' or gl.scheduler == 'WPF':
            st.past_throughput[DIR][ue_in_TTI] = st.bits_tr[DIR][ue_in_TTI]     # /st.simulation_time_s

        del st.packet_traffic[node_name][LINK][DIR][ue_in_TTI][pkt_id]


def calc_phy_throughput_FB():
    """
    Computes physical throughput for the
    full-buffer traffic
    """

    for ue_number in range(0, gl.n_UEs):

        t_del = st.simulation_time_s

        p_t = ((st.packets_counter['DL'][ue_number] * 8424) / t_del)
        st.perceived_throughput['DL'] = np.append(st.perceived_throughput['DL'], p_t)

        p_t = ((st.packets_counter['UL'][ue_number]* 8424) / t_del)
        st.perceived_throughput['UL'] = np.append(st.perceived_throughput['UL'], p_t)

        print('UE throughput ' + str(st.perceived_throughput['DL'][ue_number] / 1e6) + ' Mbps')


def transmit_blocks(link_scheduler, packet_scheduler, OFDM_params, BLERs):
    """
    Block transmission
    :param link_scheduler: link scheduler class
    :param packet_scheduler: packet scheduler class
    :param OFDM_params: OFDM parameters class
    :param BLERs: BLER curves
    """

    time_fraction_number = link_scheduler.current_state

    for node_name in st.node_names:

        DIR = st.allowed_transmissions[time_fraction_number][node_name]
        LINK = st.allowed_links[time_fraction_number][node_name]

        if DIR:

            current_schd = packet_scheduler.schedules[node_name][DIR]

            for TTI, UEs_in_TTI in enumerate(current_schd):

                for ue_in_TTI in UEs_in_TTI:

                    st.packet_traffic[node_name][LINK][DIR][ue_in_TTI].sort(key=lambda x: x.arrival_time, reverse=False)
                    transport_block_size = det_block_size(DIR, ue_in_TTI, node_name)
                    max_codeblock_size_bits = gl.max_codeblock_size_bytes * 8
                    codeblocks_in_TB = transport_block_size / max_codeblock_size_bits
                    if codeblocks_in_TB < 1:
                        codeblocks_per_UE_in_tti = -1
                    else:
                        codeblocks_per_UE_in_tti = np.floor(codeblocks_in_TB)

                    if len(st.packet_traffic[node_name][LINK][DIR][ue_in_TTI]) > 0:

                        for n_CB in range(0, int(np.ceil(codeblocks_in_TB))):

                            try:
                                current_packet = st.packet_traffic[node_name][LINK][DIR][ue_in_TTI][0]
                            except IndexError:
                                break

                            current_block = ARQ_block(TTI)
                            current_block.add_packet(0)

                            if node_name[0] == 'D':
                                n1 = 0
                            else:
                                n1 = int(node_name[1])
                            if current_packet.current_hop == n1 and transport_block_size > 0:
                                if codeblocks_per_UE_in_tti < 0:
                                    current_block.size_bits = transport_block_size
                                    st.packet_traffic[node_name][LINK][DIR][ue_in_TTI][0].size_bytes = \
                                        st.packet_traffic[node_name][LINK][DIR][ue_in_TTI][0].size_bytes - transport_block_size / 8
                                else:
                                    current_block.next_hop = n1

                            MCS_ID = st.PHY_params[node_name][DIR][ue_in_TTI].MCS

                            if MCS_ID == -1:
                                outcome = 0
                            else:

                                BLER = BLERs[MCS_ID](st.current_rsrp[node_name][DIR][ue_in_TTI])

                                if BLER[0] > 1:
                                    outcome = 0
                                elif BLER[0] < 0:
                                    BLER[0] = gl.target_BLER
                                    outcome = np.random.choice([1, 0], p=[1 - BLER[0], BLER[0]])
                                else:
                                    outcome = np.random.choice([1, 0], p=[1 - BLER[0], BLER[0]])

                            current_block.correctness_flag = outcome
                            if outcome == 1:

                                if n_CB == 0 and gl.optimization_stats is True:
                                    if node_name == 'D' and len(st.backhaul_routes[ue_in_TTI][DIR]) == 2:
                                        st.time_transmitted[node_name][ue_in_TTI, time_fraction_number-1] = \
                                            st.time_transmitted[node_name][ue_in_TTI, time_fraction_number-1] + st.symbols_per_ue[time_fraction_number]*OFDM_params.symbol_duration
                                    elif node_name != 'D':
                                        packet0 = st.packet_traffic[node_name][LINK][DIR][ue_in_TTI][0]
                                        if DIR == 'DL' and packet0.current_hop == st.backhaul_routes[ue_in_TTI][DIR][-2]:
                                            st.time_transmitted[node_name][ue_in_TTI, time_fraction_number-1] = \
                                                st.time_transmitted[node_name][ue_in_TTI, time_fraction_number-1] + st.symbols_per_ue[time_fraction_number]*OFDM_params.symbol_duration
                                        elif DIR == 'UL' and packet0.current_hop == st.backhaul_routes[ue_in_TTI][DIR][0]:
                                            st.time_transmitted[node_name][ue_in_TTI, time_fraction_number-1] = \
                                                st.time_transmitted[node_name][ue_in_TTI, time_fraction_number-1] + st.symbols_per_ue[time_fraction_number]*OFDM_params.symbol_duration

                                for pkt_id in sorted(current_block.packets, reverse=True):

                                    packet_size = st.packet_traffic[node_name][LINK][DIR][ue_in_TTI][pkt_id].size_bytes

                                    if packet_size == gl.packet_size_bytes:
                                        # In the following function,
                                        # for whole packets, next hop is assigned or they
                                        # are delivered to the destination
                                        if gl.traffic_type == 'FTP':
                                            deliver_packet_FTP(DIR, ue_in_TTI, pkt_id, OFDM_params, TTI)
                                        elif gl.traffic_type == 'full':
                                            deliver_packet_FB(DIR, ue_in_TTI, pkt_id, OFDM_params, node_name, TTI,
                                                              time_fraction_number, LINK)

                                    else:
                                        # For packets, which were divided,
                                        # a separate buffer is created and their integrity will be
                                        # verified every time until it becomes whole
                                        st.fragmented_packets[DIR][ue_in_TTI].append(current_block)
                                        s_check = collections.defaultdict(list)
                                        fragments_numbers = []
                                        for f_num, fragm in enumerate(st.fragmented_packets[DIR][ue_in_TTI]):
                                            if fragm.packets.__len__() > 1:
                                                raise ValueError
                                            if fragm.size_bits:
                                                # first fragment of packet has size in fragm
                                                s_check[fragm.packets[0]].append(fragm.size_bits)
                                                fragments_numbers.append(f_num)
                                            else:
                                                # if the number of TBS was sufficient to transmit the rest
                                                # of a packet, its remaining part is ss (packet.syze_bytes)
                                                s_check[fragm.packets[0]].append(packet_size)
                                                fragments_numbers.append(f_num)
                                            if np.sum(s_check[fragm.packets[0]]) >= packet_size:  # gl.ip_packet_size_bytes * 8:
                                                # remove delivered fragments from buffers
                                                for ff in fragments_numbers[::-1]:
                                                    del st.fragmented_packets[DIR][ue_in_TTI][ff]
                                                if gl.traffic_type == 'FTP':
                                                    deliver_packet_FTP(DIR, ue_in_TTI, pkt_id, OFDM_params, TTI)
                                                elif gl.traffic_type == 'full':
                                                    deliver_packet_FB(DIR, ue_in_TTI, pkt_id, OFDM_params, node_name,
                                                                      TTI, time_fraction_number, LINK)

                            else:
                                current_block.times_retransmitted += 1

                        if np.any(st.actual_throughput[DIR][ue_in_TTI]):
                            bits_per_tti = np.sum(st.actual_throughput[DIR][ue_in_TTI])
                            mean_delay = np.mean(st.per_packet_delay_per_TTI[DIR][ue_in_TTI])
                            st.mean_throughput[DIR] = np.append(st.mean_throughput[DIR], bits_per_tti / mean_delay)
                            st.mean_delay[DIR] = np.append(st.mean_delay[DIR], mean_delay)

                        st.actual_throughput[DIR][ue_in_TTI] = np.array([])
                        st.per_packet_delay_per_TTI[DIR][ue_in_TTI] = np.array([])
