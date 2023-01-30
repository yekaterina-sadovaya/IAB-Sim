import numpy as np


class GL_container_class:
    def __init__(self):
        # 1 simulation tic corresponds to a transmission of half (or another coefficient) of the frame (UL or DL)
        self.time_stop_tics = 120000
        self.carrier_frequency_Hz = 28e9
        self.cell_radius_m = 2000
        # 1 Mbyte = 1e6 byte
        self.file_size_bytes = 10*1.995*1e6
        # Max IP packet size for 5G = 1500
        self.ip_packet_size_bytes = 8424/8
        self.n_UEs = 30
        self.n_IAB = 1
        self.n_DgNB = 1
        self.FTP_parameter_lambda_DL = 0.1
        self.FTP_parameter_lambda_UL = 0.5
        self.multihop_flag = True
        self.multihop_option = 'best_rsrp_maxmin'       # min_hops, best_rsrp_maxmin, rand
        self.multi_con_degree = 1
        self.blockers_density = 0.0001
        self.antenna_diversity = False
        self.MAX_ARQ_RETRIES = 3

        self.bandwidth: float = 400e6
        self.thermal_noise_density: float = -173.93
        self.NF_BS_dB = 7
        self.NF_UE_dB = 13
        self.interference_margin_dB = 3

        self.DgNB_height = 25
        self.iab_height = 10
        self.UE_height = 1.5
        # set mobility pattern to RDM or RPGM
        self.UE_mobility_pattern = 'RDM'
        self.UE_min_speed = 0.000082*4      # m/s
        self.UE_average_speed = 0.000084*4  # m/s
        # self.UE_min_speed = 11.1*4      # m/s
        # self.UE_average_speed = 11.2*4  # m/s
        self.UE_tx_power_dBm = 23
        self.DgNB_tx_power_dBm = 40
        self.IAB_tx_power_dBm = 30

        self.DgNB_pos = np.zeros([self.n_DgNB, 3])
        self.IAB_pos = np.zeros([self.n_IAB, 3])
        self.N_antenna_elements_DgNB = 16
        self.N_antenna_elements_IAB = 16
        self.N_antenna_elements_UE = 4

        self.PL_calculation_option = 'simple'   # has 2 options: "simple" and "cluster"
                                                # by "simple" it is assumed that only PL formula is used
        self.shadow_fading = True               # turn this on to enable SF for the simple PL calculations
        self.antenna_downtilt_deg = 3
        self.loss_from_blockage_dB = 20
        self.channel_update_periodicity_tics = 5000
        self.rsrp_statistics = False            # turn ths on to gather RSRP data in time (does not impact
                                                # simulations), enable for post-processing if needed

        # OFDM-related parameters
        self.numerology = 3     # default numerology is 0
        self.Bandwidth = 400e6

        # subfr or slot
        # practically, we cannot divide anything rather then slots
        self.division_unit = 'slot'

        # The size of code block block in bits
        self.MAX_CB_size_bits = 8424

        self.target_BLER = 0.1
        # choose from 'FTP' or 'full'
        self.traffic = 'full'

        # RA_division flag sets, which resources are divided; possible values are 'time', 'freq', 'time-freq'
        self.RA_division = 'time'

        # 50/50, PF, OPT
        self.frame_division_policy = 'OPT'
        # if this flag is enable, average values of spectral efficiencies will be used in the optimization
        self.use_average = True
        # Choose optimization 'MAXMIN' or 'PF'
        self.optimization_type = 'MAXMIN'

        self.scheduler = 'WFQ'      # PF, RR, WFQ, WPF

        self.print_Flag = True
        self.plot_Flag = False
        self.optimization_stats = True


gl = GL_container_class()
