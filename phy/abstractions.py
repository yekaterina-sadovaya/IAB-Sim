
class OFDMparams:
    """
    This class stores OFD parameters
    """
    def __init__(self, SCS_Hz, RB_freq_Hz, RB_time_s, T_sym):
        self.SCS_Hz = SCS_Hz
        self.RB_freq_Hz = RB_freq_Hz
        self.RB_time_s = RB_time_s
        # symbol duration including CP (normal length)
        self.symbol_duration = T_sym


def set_params_OFDM(numerology):
    """
    Configures OFDM parameters based on
    the given numerology
    """
    params = {0: OFDMparams(15e3, 180e3, 1e-3, 71.35e-6),
              1: OFDMparams(30e3, 360e3, 0.5e-3, 35.68e-6),
              2: OFDMparams(60e3, 720e3, 0.25e-3, 17.84e-6),
              3: OFDMparams(120e3, 1440e3, 0.125e-3, 8.92e-6),
              4: OFDMparams(240e3, 2880e3, 0.0625e-3, 4.46e-6)}
    try:
        return params[numerology]
    except KeyError:
        return params[0]
