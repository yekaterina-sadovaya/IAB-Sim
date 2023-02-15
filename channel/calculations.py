import numpy as np
import copy

from library.stuff import cart2sph
from channel.propagation import MP_Propagation_Model, UMa_LoS, antenna3gpp
from channel.channel_config import MP_chan_params_Cluster
from gl_vars import gl

ch_params = MP_chan_params_Cluster()
antenna_downtilt_rad = gl.antenna_downtilt_deg*np.pi/180


def calc_transmission(txs, rx, distances, blockage, Nant_tx: int, Nant_rx: int):

    class MainTransmission:
        def __init__(self):
            self.tx_main_directions = []
            self.rx_main_directions = []

    main_transmission = MainTransmission()
    if type(txs) is list:
        s = txs.__len__()
    elif type(txs) is np.ndarray:
        s = txs.shape[0]
    else:
        raise TypeError

    PL = np.zeros(s)
    PL_average = np.zeros(s)
    for tx_index, tx_i in enumerate(txs):
        min_distance = distances[tx_index]
        is_blocked = blockage[tx_index]

        transmitting_direction = (rx - tx_i) / min_distance

        azimuth_angle_tx, zenith_angle_tx, r = cart2sph(transmitting_direction[0], transmitting_direction[1],
                                                      transmitting_direction[2])
        tx_ant = [azimuth_angle_tx, zenith_angle_tx]
        main_transmission.tx_main_directions.append(tx_ant)

        azimuth_angle_rx, zenith_angle_rx, r = cart2sph(-transmitting_direction[0], -transmitting_direction[1],
                                                   transmitting_direction[2])

        rx_ant = [azimuth_angle_rx, zenith_angle_rx]
        main_transmission.rx_main_directions.append(rx_ant)

        if gl.PL_calculation_option == 'cluster':
            ch_prop_model = MP_Propagation_Model(ch_params, Nant_tx, Nant_rx)
            Ch_params_clusters = ch_prop_model.get_Cluster_channel_mmWave(min_distance, transmitting_direction,
                                                                          tx_ant, rx_ant, 'LOS')
            pl = Ch_params_clusters.PL
            # set gains to zero because they are accounted for each MPC in channel model calculations
            G_tx = 0
            G_rx = 0
        elif gl.PL_calculation_option == 'simple':
            pl, pl_average = UMa_LoS(min_distance, f_carrier=gl.carrier_frequency_Hz, h_bs=rx[2], h_ms=tx_i[2])
            antenna_tx = antenna3gpp(Nant_tx)
            antenna_rx = antenna3gpp(Nant_rx)
            G_tx = antenna_tx(0, zenith_angle_tx - antenna_downtilt_rad)
            G_rx = antenna_rx(0, zenith_angle_rx)
        else:
            print('PL calculation option is specified incorrectly')
            raise ValueError

        if is_blocked[0]:
            pl = pl + gl.loss_from_blockage_dB

        PL[tx_index] = pl - G_tx - G_rx
        PL_average[tx_index] = pl_average - G_tx - G_rx

    return PL, PL_average
