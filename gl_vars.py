import numpy as np


class GL_container_class:
    """
    This class combines all parameters required to configure
    the simulation environment
    """
    def __init__(self):

        # ------------------------Generic simulation parameters------------------------
        self.SIM_SEED = 1
        self.sim_time_tics = 1000                         # 1 tic equals 1 DL or UL interval in a frame

        # ------------------------Deployment configuration----------------------------
        self.carrier_frequency_Hz = 28e9
        self.bandwidth = 400e6
        self.cell_radius_m = 2000
        self.n_UEs = 30
        self.n_IAB = 1
        self.DgNB_height = 25                           # height in meters
        self.iab_height = 10
        self.UE_height = 1.5
        self.FTP_parameter_lambda_DL = 0.1              # traffic intensity in DL
        self.FTP_parameter_lambda_UL = 0.5              # traffic intensity in UL
        self.multihop_flag = True                       # if disabled, it is still possible to have 2 hops
                                                        # from UE to donor but it disables
                                                        # connections with larger number of hops
        self.ue_associations = 'best_rsrp_maxmin'       # min_hops, best_rsrp_maxmin, rand
        self.multi_con_degree = 1                       # 1 or 2 (splits traffic from UEs to multiple nodes)
                                                        # do not use with multihop_flag = True
        self.traffic_type = 'full'                           # choose from 'FTP' or 'full'

        # ------------------------Scheluling and Optimization--------------------------------
        self.frame_division_policy = 'PF'              # available: '50/50', 'PF', 'OPT'

        self.use_average = True                         # if this flag is enable, average values
                                                        # of spectral efficiencies will be used in the fb_optimization
                                                        # if disable, instant variables will be used
        self.optimization_type = 'MAXMIN'               # choose fb_optimization: 'MAXMIN' or 'PF'
        self.scheduler = 'PF'                          # PF, RR, WFQ, WPF
                                                        # PF and RR are non-parametrized schedulers
                                                        # (use without fb_optimization)
                                                        # WPF and WFQ use coefficients from the fb_optimization

        # ------------------------Channel calculation and blockage---------------------------
        self.channel_update_periodicity_tics = 5000
        self.PL_calculation_option = 'simple'           # has 2 options: "simple" and "cluster"
                                                        # by "simple" it is assumed that only PL formula is used
        self.shadow_fading = True                       # turn this on to enable SF for the simple PL calculations
        self.loss_from_blockage_dB = 20
        self.blockers_density = 0.0001                  # density of blockers per square meter

        # ------------------------Physical parameters of UEs and nodes/DgNB-----------------
        self.thermal_noise_density = -173.93
        self.NF_BS_dB = 7                               # noise figure of IAB nodes and donor is the same
        self.NF_UE_dB = 13
        self.interference_margin_dB = 3                 # average interference margin
        self.N_antenna_elements_DgNB = 16
        self.N_antenna_elements_IAB = 16
        self.N_antenna_elements_UE = 4
        self.antenna_downtilt_deg = 3
        self.UE_tx_power_dBm = 23
        self.DgNB_tx_power_dBm = 40
        self.IAB_tx_power_dBm = 30

        # ------------------------Packet transmission parameters-----------------------------
        self.burst_size_bytes = 10 * 1.995 * 1e6        # file size for full-buffer traffic should be set
                                                        # to a large number, which will not be all transmitted;
                                                        # for FTP traffic, 1 Mbyte = 1e6 bytes is the standard value
        self.packet_size_bytes = 1053                   # packet size and block size should be put equal
                                                        # to enable faster calculations without fragmentations
        self.max_codeblock_size_bytes = 1053
        self.max_ARQ_retries = 3                        # maximum number of retransmissions allowed

        # ------------------------UE mobility configurations---------------------------------
        self.UE_mobility_pattern = 'stable'             # RDM, RPGM, stable
                                                        # for optimal scheduling, set to stable
        self.UE_min_speed = 0.05                        # m/s
        self.UE_average_speed = 1                       # m/s

        # ------------------------OFDM-related parameters----------------------------------
        self.numerology_num = 3                         # default numerology is 0
        self.FRAME_DURATION_S = 10e-3
        self.target_BLER = 0.1

        # ------------------------Printing, Plotting, and Saving-----------------------------
        self.print_Flag = True
        self.plot_Flag = True
        self.optimization_stats = True                  # enable to gather fb_optimization statistics, e.g.,
                                                        # allocations and coefficients

        # initialize positions of IAB and donor nodes to empty (used in the calculations)
        self.DgNB_pos = np.zeros([1, 3])
        self.IAB_pos = np.zeros([self.n_IAB, 3])


gl = GL_container_class()
