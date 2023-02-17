import numpy as np
from scipy.constants import speed_of_light
from library.additional_functions import friis_path_loss_dB


class MP_chan_params(object):
    """Config class for the multipath channel"""
    def __init__(self):
        self.crash_on_near_field = True #Crash if link is in near-field conditions. Will issue errors anyway.
        self.max_power = 23  # maximal power in dBm/subchannel
        self.max_gain = 15  # maximal single antenna gain dB
        self.subchannel_BW = 150e3  # Subchannel bandwidth (for OFDM FFT results), set to None to avoid FFT calculation
        self.sys_BW = 1e6  # System Bandwidth in Hz
        self.max_SINR = 25  # dB
        self.min_SINR = -5  # dB
        self.cache_time = 1 # time to cache channel state data
        self.carrier = 1e9  # Carrier frequency in Hz
        self.coverage_n = 2.0  # Approximate path loss exponent for quick coverage tests
        self.max_subchannels = int(self.sys_BW / self.subchannel_BW)
        self.symbol_duration = 1 / self.sys_BW  # The symbol duration in the channel in seconds
        self.subchannel_gain = 10 * np.log10(self.subchannel_BW)  # subchannel gain
        self.sensitivity_level = -130 + self.subchannel_gain  # dBm/subchannel
        self.MCL = self.max_power + self.max_gain - self.sensitivity_level #Max coupling loss of a link
        self.guard_interval_duration = 1 / self.sys_BW / 128 # FIXME: What does this do?
        self.prop_loss_function = friis_path_loss_dB


class MP_chan_params_LOS(MP_chan_params):
    def __init__(self):
        MP_chan_params.__init__(self)
        self.prop_loss_function = friis_path_loss_dB


from math import log10
from gl_vars import gl


class MP_chan_params_Cluster(MP_chan_params):
    def __init__(self, sys_bw=400e6, max_power=23):
        MP_chan_params.__init__(self)
        self.prop_loss_function = friis_path_loss_dB
        self.N_clust = 12  # UMa LoS
        self.N_rays = 20  # rays per cluster
        self.delay_scaling = max(0.25, 6.5622-3.4084*log10(gl.carrier_frequency_Hz))  # delay scaling parameter 3
        self.ds_var = 0.66  # large scale parameter from paper
        self.ds_mean = -6.955 - 0.0963*log10(gl.carrier_frequency_Hz)  # large scale parameter from paper
        self.per_clust_sh = 3  # sf parameter from paper
        self.ricean_fact_mean = 9  # ricean factor from paper
        self.ricean_fact_var = 3.5
        self.sys_BW = sys_bw
        self.sensitivity_level = -174 + 10*np.log10(self.sys_BW)
        self.max_gain = 100.
        self.max_SINR = 40  # dB
        self.min_SINR = -30  # dB
        self.max_power = max_power  # maximal power in dBm/subchannel

        self.xpr_mean = 8
        self.xpr_var = 4.

        self.asd_mean = 1.06 + 0.114*log10(gl.carrier_frequency_Hz)
        self.asd_var = 0.28

        self.asa_mean = 1.81
        self.asa_var = 0.20
        self.carrier = 30e9
        self.wave_l = speed_of_light/self.carrier

        # Default GI for some mmWave system we like
        self.guard_interval_duration = 25.78e-9
        self.symbol_duration = 651.04e-6
        self.SAMPLING_RATE = sys_bw * 3
        self.PLE = 3.3  # path loss exponent


class MP_chan_params_5GNR(MP_chan_params_LOS):
    """Config class for the 5G NR channel"""

    def __init__(self, max_BW, numerology, carrier_freq):
        """
        Creates a parameter object for 5G NR channels

        :param max_BW: Maximum bw of the entire channel
        :param numerology: Numerology (µ) of the system
        :param carrier_freq: Carrier frequency in Hz
        """
        MP_chan_params_LOS.__init__(self)

        assert 0 <= numerology <= 5, f"Numerology is outside bounds. Expected [0, 5], got: {numerology}"
        # ensure that the BW can be split into sensible blocks
        self.subchannel_BW = 2**numerology * 15e3 * 12
        assert max_BW % self.subchannel_BW == 0, "Channel max BW doesn't divide evenly into subchannels"
        self.sys_BW = max_BW
        self.max_subchannels = int(self.sys_BW/self.subchannel_BW)

        self.subchannel_gain = 10 * np.log10(self.subchannel_BW)
        self.max_power = 23 - self.subchannel_gain  # maximal power in dBm/subchannel
        self.sensitivity_level = -140 - self.subchannel_gain  # dBm/subchannel
        self.carrier = carrier_freq # Carrier frequency
        self.symbol_duration = 0.001/(numerology+1) # Symbol duration in the channel in seconds
        self.coverage_n = 2.5
