from mobility.configure_mobility import set_mobility_model
from gl_vars import gl
from stat_container import st
from generate_traffic import ftp3_traffic
from topology_formation import TopologyCreator
from packet.packets_operations import packetize_data, transmit_blocks, calc_phy_throughput_FB
from phy.abstractions import set_params_OFDM
from scheduling.link_scheduler import LinkScheduler, LinkSchedulerOptimal
from scheduling.packet_scheduler import Scheduler
from library.random_drop import drop_DgNB, drop_IAB
from phy.interpolate_ber_curves import load_BER
from post_processing.gather_results import combine_metrics, save_data

import numpy as np
import matplotlib.pyplot as plt

plt.style.use('YS_plot_style.mplstyle')


def packet_sim():
    """
    This functions starts the simulations with
    the configured parameters
    """

    BER_CURVES = load_BER()

    # Drop the DgNB and IAB nodes
    drop_DgNB(gl.SIM_SEED)
    drop_IAB(gl.SIM_SEED)

    # Calculate parameters, which do not change during the calculations
    OFDM_params = set_params_OFDM(gl.numerology_num)
    if gl.UE_mobility_pattern == 'stable':
        # periodicity determines the frequency of channel updates
        periodicity = gl.sim_time_tics
    else:
        periodicity = gl.FRAME_DURATION_S / OFDM_params.RB_time_s

    UE_mobility_model = set_mobility_model(gl.SIM_SEED, gl.FRAME_DURATION_S)
    topology = TopologyCreator(BER_CURVES)
    if gl.frame_division_policy != 'OPT':
        link_scheduler = LinkScheduler()
    else:
        link_scheduler = LinkSchedulerOptimal()
    packet_scheduler = Scheduler()

    st.__init__()

    # Containers to store data
    per_packet_throughput = {'DL': np.array([]), 'UL': np.array([])}
    packet_delay = {'DL': np.array([]), 'UL': np.array([])}
    num_active_UEs = {'DL': np.array([]), 'UL': np.array([])}
    number_of_hops_DL = np.array([])

    for tic in range(0, gl.sim_time_tics):

        if tic % periodicity * 2 == 0:
            UE_positions = next(UE_mobility_model)
            # Transform according to the cell size
            x = UE_positions[:, 0] - gl.cell_radius_m
            y = UE_positions[:, 1] - gl.cell_radius_m
            UE_positions_tr = [x, y, UE_positions[:, 2]]
            UE_positions_tr = np.transpose(UE_positions_tr)

        if tic == 0:
            # generate traffic (demands) for each UE initially
            ftp3_traffic('UL', tic, gl.SIM_SEED)
            ftp3_traffic('DL', tic, gl.SIM_SEED)
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

                if any(st.mean_throughput['DL']):
                    st.mean_throughput['DL'] = np.mean(st.mean_throughput['DL'])
                    st.mean_delay['DL'] = np.mean(st.mean_delay['DL'])

                ftp3_traffic('DL', tic, gl.SIM_SEED)
                ftp3_traffic('UL', tic, gl.SIM_SEED)

                per_packet_throughput, packet_delay, num_active_UEs, number_of_hops_DL = \
                    combine_metrics(link_scheduler, tic, per_packet_throughput,
                                    packet_delay, num_active_UEs, number_of_hops_DL)

                # update simulation time
                st.simulation_time_s = st.simulation_time_s + gl.FRAME_DURATION_S * C

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

    if gl.traffic_type == 'full':
        calc_phy_throughput_FB()

    save_data(per_packet_throughput, packet_delay, num_active_UEs, number_of_hops_DL)


if __name__ == "__main__":
    packet_sim()
