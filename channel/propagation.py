from numpy import log10, sqrt, pi, floor
import numpy as np
from warnings import warn
from scipy.constants import speed_of_light
from library.additional_functions import RATIO2DB, DB2RATIO, cart2sph
from library.vectors import vector_normalize, norm
from channel.channel_config import MP_chan_params
from numpy.random import shuffle, uniform, choice, normal, rand
from math import exp, log2
import scipy.stats as stats


from channel.antenna import antenna3gpp
from gl_vars import gl


# @jitclass(spec)
class MP_Chan_State(object):
    """Object representing a multipath channel"""

    def __init__(self):
        self.PL = 0
        self.phase_delay = 0


def generate_clusters(self, d: float, dv: list, PL_type:str):
    """Multipath cluster generator for 3GPP stochastic channel model"""

    res = MP_Chan_State()
    # Simple 3GPP-like cluster generation; return cluster_delays, cluster powers
    los_ray_pow = 0
    asd_spread = 1.06 + 0.114*log10(gl.carrier_frequency_Hz)
    asa_spread = 1.81
    zsd_spread = 0      # 10**(max(-0.0023*d + 0.81, 0))

    zsa_spread = 0.95   # 10**(max(-0.002*d + 0.83, 0))
    angle_offset = [0.0447, -0.0447,
                    0.1413, -0.1413,
                    0.2492, -0.2492,
                    0.3715, -0.3715,
                    0.5129, -0.5129,
                    0.6797, -0.6797,
                    0.8844, -0.8844,
                    1.1481, -1.1481,
                    1.5195, -1.5195,
                    2.1551, -2.1551]

    zsd_mean = 0        # max(-0.002 * d + 1.05, 4)
    zsd_var = 0.32

    zsa_mean = 0.95     # max(-0.0025 * d + 1.1, 3)
    zsa_var = 0.16

    zbd_mean = 0        # max(-0.0022 * d + 1.36, 0.6)
    zbd_var = 0.3

    zba_mean = 0        # max(-0.00172 * d + 1.09, 0.4)
    zba_var = 0.3

    # Generating rays parameters
    outgoing_ray = dv
    outgoing_ray = vector_normalize(outgoing_ray)
    los_aod, los_zod, r = cart2sph(outgoing_ray[0], outgoing_ray[1],
                                   outgoing_ray[2])
    los_aod = np.degrees(los_aod)
    los_zod = np.degrees(los_zod)

    incoming_ray = dv
    incoming_ray = vector_normalize(incoming_ray)
    los_aoa, los_zoa, r = cart2sph(-incoming_ray[0], -incoming_ray[1],
                                   incoming_ray[2])
    los_aoa = np.degrees(los_aoa)
    los_zoa = np.degrees(los_zoa)

    cluster_delays = []
    cluster_powers_init = []
    cluster_aoas = []
    cluster_aods = []
    cluster_zoas = []
    cluster_zods = []

    cluster_delays_list = []
    cluster_aoas_list = []
    cluster_aods_list = []
    cluster_zoas_list = []
    cluster_zods_list = []
    cluster_xpr_list = []
    cluster_phiVV_list = []
    cluster_phiVH_list = []
    cluster_phiHV_list = []
    cluster_phiHH_list = []

    # Generating lagre scale parameters

    normal_var_vect = [normal(loc=0., scale=1.) for i in range(0, 7)]
    cor_matrix = [
        [1.0, 0.49, 0, 0.29, 0, 0.2, -0.26],
        [0.49, 1.0, 0, 0.62, 0, 0.47, -0.16],
        [0, 0, 1.0, 0, 0.55, 0, 0.45],
        [0.29, 0.62, 0, 1.0, 0.33, 0.76, 0.17],
        [0, 0, 0.55, 0.33, 1.0, 0.33, 0.69],
        [0.2, 0.47, 0, 0.76, 0.33, 1.0, 0.39],
        [-0.26, -0.16, 0.45, 0.17, 0.69, 0.39, 1.0]]
    V, D = np.linalg.eigh(cor_matrix, UPLO='U')
    cor_multiplier = np.dot(D, np.sqrt(np.diag(V)))
    normal_var_vect = np.dot(cor_multiplier, normal_var_vect)
    # wave_length = speed_of_light/self.params.carrier
    # PL = 20.*log10(pi*4/wave_length) + (10.*self.params.PLE*log10(d))
    PL = UMa_LoS(d, self.params.carrier)

    # laplac_var_vect = laplace(size=2)
    d_spread = 10 ** ((self.params.ds_var * normal_var_vect[0]) + self.params.ds_mean)

    asd = min(10 ** ((self.params.asd_var * normal_var_vect[1]) + self.params.asd_mean), 100.0)
    asa = min(10 ** ((self.params.asa_var * normal_var_vect[2]) + self.params.asa_mean), 100.0)
    zsd = min(10 ** ((zsd_var * normal_var_vect[3]) + zsd_mean), 40.0)
    zsa = min(10 ** ((zsa_var * normal_var_vect[4]) + zsa_mean), 40.0)
    m_bias_zod = max(-10 ** (zbd_mean + zbd_var * normal_var_vect[5]), -75)
    m_bias_zoa = max(-10 ** (zba_mean + zba_var * normal_var_vect[6]), -75)
    ricean_fact = abs((self.params.ricean_fact_var * normal_var_vect[5]) + self.params.ricean_fact_mean)

    for i in range(0, self.params.N_clust):
        delay_var = uniform(0.000000001, 1.0)
        # Delays
        delay = -self.params.delay_scaling * d_spread * np.log(delay_var)
        cluster_delays.append(delay)
    cluster_delays.sort()
    if PL_type == 'NLOS':
        res.phase_delay = np.array(cluster_delays).min()
        cluster_delays -= res.phase_delay

    for i in range(0, len(cluster_delays)):
        # cluster_delays_new.append(cluster_delays[i])
        cluster_delays_new = []
        for j in range(0, self.params.N_rays):
            delay = cluster_delays[i]
            if j > 4 and j <= 7:
                delay += 5e-9
            elif j > 7:
                delay += 10e-9
            cluster_delays_new.append(delay)
        cluster_delays_list.append(cluster_delays_new)

    for i in range(0, self.params.N_clust):
        # Cluster powers
        z_clust = normal(0, self.params.per_clust_sh ** 2)
        power = (10 ** (-0.1 * z_clust)) * np.exp(
                -cluster_delays[i] * (self.params.delay_scaling - 1) / (self.params.delay_scaling * d_spread))
        cluster_powers_init.append(power)

    cluster_powers = list(np.array(cluster_powers_init) / sum(cluster_powers_init))
    # cluster_delays = cluster_delays_list

    # AoDs and AoAs
    for i in range(0, self.params.N_clust):
        xn = [-1, 1]
        uni_rand_int = choice(xn)
        norm_rand_var = normal(0, asd / 7)
        aod = 2.33 * (asd / 1.4) * np.sqrt(-np.log(cluster_powers[i] / max(cluster_powers)))
        aod = (uni_rand_int * aod) + norm_rand_var + los_aod

        uni_rand_int = choice(xn)
        norm_rand_var = normal(0, asa / 7)
        aoa = 2.33 * (asa / 1.4) * np.sqrt(-np.log(cluster_powers[i] / max(cluster_powers)))
        aoa = (uni_rand_int * aoa) + norm_rand_var + los_aoa

        uni_rand_int = choice(xn)
        norm_rand_var = normal(0, zsd / 7)
        zod = -1.01 * zsd * np.log(cluster_powers[i] / max(cluster_powers))
        zod = (uni_rand_int * zod) + norm_rand_var + los_zod + m_bias_zod

        uni_rand_int = choice(xn)
        norm_rand_var = normal(0, zsa / 7)
        zoa = -1.01 * zsa * np.log(cluster_powers[i] / max(cluster_powers))
        zoa = (uni_rand_int * zoa) + norm_rand_var + los_zoa + m_bias_zoa

        cluster_aoas.append(aoa)
        cluster_aods.append(aod)
        cluster_zoas.append(zoa)
        cluster_zods.append(zod)

        aoa_list = []
        zoa_list = []
        aod_list = []
        zod_list = []
        xpr_list = []
        phiVV_list = []
        phiVH_list = []
        phiHV_list = []
        phiHH_list = []

        for j in range(0, self.params.N_rays):
            xn = [1, -1]
            aod_ray = aod + asd_spread * xn[j % len(xn)] * angle_offset[int(floor(j / 2))]
            aoa_ray = aoa + asa_spread * xn[j % len(xn)] * angle_offset[int(floor(j / 2))]
            zoa_ray = zoa + zsa_spread * xn[j % len(xn)] * choice(xn) * angle_offset[int(floor(j / 2))]
            zod_ray = zod + zsd_spread * xn[j % len(xn)] * choice(xn) * angle_offset[int(floor(j / 2))]
            aoa_list.append(aoa_ray)
            aod_list.append(aod_ray)
            zoa_list.append(zoa_ray)
            zod_list.append(zod_ray)
            xpr = self.params.xpr_var * normal(0., 1.) + self.params.xpr_mean
            xpr = 10 ** (xpr / 10)
            xpr_list.append(xpr)
            phiVV = -pi + 2 * pi * rand()
            phiVH = -pi + 2 * pi * rand()
            phiHV = -pi + 2 * pi * rand()
            phiHH = phiVV + (rand() > 0.5) * pi
            phiVV_list.append(phiVV)
            phiVH_list.append(phiVH)
            phiHV_list.append(phiHV)
            phiHH_list.append(phiHH)
        shuffle(aoa_list)
        shuffle(aod_list)
        shuffle(zoa_list)
        shuffle(zod_list)
        cluster_aoas_list.append(aoa_list)
        cluster_aods_list.append(aod_list)
        cluster_zoas_list.append(zoa_list)
        cluster_zods_list.append(zod_list)
        cluster_xpr_list.append(xpr_list)
        cluster_phiVV_list.append(phiVV_list)
        cluster_phiVH_list.append(phiVH_list)
        cluster_phiHV_list.append(phiHV_list)
        cluster_phiHH_list.append(phiHH_list)
    if PL_type == 'LOS':
        los_ray_pow = ricean_fact  # / (1 + ricean_fact)
        cluster_powers.insert(0, los_ray_pow)
        # cluster_powers = list(np.array(cluster_powers_init) / ((ricean_fact + 1) * sum(cluster_powers_init)))
        cluster_delays = np.insert(cluster_delays, 0, 0)
        cluster_aoas.insert(0, los_aoa)
        cluster_aods.insert(0, los_aod)
        cluster_zoas.insert(0, los_zoa)
        cluster_zods.insert(0, los_zod)

        cluster_delays_list.insert(0, [0])
        cluster_aoas_list.insert(0, [los_aoa])
        cluster_aods_list.insert(0, [los_aod])
        cluster_zoas_list.insert(0, [los_zoa])
        cluster_zods_list.insert(0, [los_zod])
        phiVV = -pi + 2 * pi * rand()
        phiVH = 0  # -pi + 2 * pi * random()
        phiHV = 0  # -pi + 2 * pi * random()
        phiHH = phiVV + pi
        cluster_xpr_list.insert(0, [float('inf')])
        cluster_phiVV_list.insert(0, [phiVV])
        cluster_phiVH_list.insert(0, [phiVH])
        cluster_phiHV_list.insert(0, [phiHV])
        cluster_phiHH_list.insert(0, [phiHH])

    N_clusters = self.params.N_clust
    cluster_delays = np.array(cluster_delays)
    cluster_aods = np.array(cluster_aods_list)
    cluster_aoas = np.array(cluster_aoas_list)
    cluster_zods = np.array(cluster_zods_list)
    cluster_zoas = np.array(cluster_zoas_list)

    cluster_delays_list = np.array(cluster_delays_list)
    cluster_xpr_list = np.array(cluster_xpr_list)
    cluster_phiVV_list = np.array(cluster_phiVV_list)
    cluster_phiVH_list = np.array(cluster_phiVH_list)
    cluster_phiHV_list = np.array(cluster_phiHV_list)
    cluster_phiHH_list = np.array(cluster_phiHH_list)

    result = [res, PL, PL_type, N_clusters, cluster_powers, cluster_aoas, cluster_aods, cluster_zoas,
              cluster_zods, cluster_phiVV_list, cluster_phiVH_list, cluster_phiHV_list, cluster_phiHH_list,
              cluster_xpr_list, cluster_delays_list, los_ray_pow]
    # self.cache[str(list(src_pos)+list(dst_pos))] = result
    return result


class MP_Propagation_Model(object):
    """Cluster-based propagation model"""

    def __init__(self, params: MP_chan_params, Nant_tx, Nant_rx):
        self.params = params
        # interpolated pattern of the 3gpp antenna arrays
        self.tx_antenna = antenna3gpp(Nant_tx)
        self.rx_antenna = antenna3gpp(Nant_rx)

    def get_LOS_channel(self, src_position, dst_position):
        """Obtains LoS path"""

        dv = np.array(dst_position - src_position)
        lamb = speed_of_light / self.params.carrier
        dist = norm(dv)

        if dist < lamb:
            print("Distance = ", dv)
            if dist == 0:
                warn("Link length exactly zero, can not compute!")
            if self.params.crash_on_near_field:
                raise RuntimeError("PHY link too short")
            else:
                warn("Link too short for comfort {}", dv)
                dist = lamb

        PL = -self.params.prop_loss_function(dist, self.params.carrier, self.params.coverage_n)
        if PL > self.params.MCL:
            return None

        cs = MP_Chan_State()
        cs.PL = PL
        cs.phase_delay = dist / speed_of_light
        return cs

    def get_Cluster_channel_mmWave(self, dist, dv, tx_ant, rx_ant, PL_type):
        """Computes channel parameters such as, e.g., PL, impulse response, angles"""

        clusters = generate_clusters(self, dist, dv, PL_type)

        res, PL, PL_type, N_clusters, cluster_powers, cluster_aoas, cluster_aods, cluster_zoas, cluster_zods, \
        cluster_phiVV_list, cluster_phiVH_list, cluster_phiHV_list, cluster_phiHH_list, \
        cluster_xpr_list, cluster_delays_list, los_ray_pow = clusters

        impulse_responce = []
        cluster_tx_gains_list = []
        cluster_rx_gains_list = []
        cluster_delays = []
        for i in range(0, len(cluster_aods)):
            tx_gains_list = []
            rx_gains_list = []
            for j in range(0, len(cluster_aods[i])):
                tx_gains = self.tx_antenna(tx_ant[0] - (cluster_aods[i][j] * pi / 180),
                                           tx_ant[1] - (cluster_zods[i][j] * pi / 180))
                tx_gains = DB2RATIO(tx_gains)
                rx_gains = self.rx_antenna(rx_ant[0] - (cluster_aoas[i][j] * pi / 180),
                                           rx_ant[1] - (cluster_zoas[i][j] * pi / 180))
                rx_gains = DB2RATIO(rx_gains)

                tx_gains_list.append(tx_gains)
                rx_gains_list.append(rx_gains)
            cluster_tx_gains_list.append(tx_gains_list)
            cluster_rx_gains_list.append(rx_gains_list)
        cluster_rx_gains_list = np.array(cluster_rx_gains_list)
        cluster_tx_gains_list = np.array(cluster_tx_gains_list)

        for i in range(0, len(cluster_tx_gains_list)):
            ray_values = []
            cluster_delays.extend(cluster_delays_list[i])
            for j in range(0, len(cluster_tx_gains_list[i])):
                if PL_type == 'LOS' and i == 0:
                    pol_matrix = np.array([[np.exp(1j * cluster_phiVV_list[i][j]), 0],
                                           [0, -1 * np.exp(1j * cluster_phiHH_list[i][j])]])
                else:
                    pol_matrix = np.array([[np.exp(1j * cluster_phiVV_list[i][j]),
                                            sqrt(cluster_xpr_list[i][j] ** -1) * np.exp(
                                                1j * cluster_phiVH_list[i][j])],
                                           [sqrt(cluster_xpr_list[i][j] ** -1) * np.exp(
                                               1j * cluster_phiHV_list[i][j]),
                                            np.exp(1j * cluster_phiHH_list[i][j])]])

                tx_gains = cluster_tx_gains_list[i][j]
                rx_gains = cluster_rx_gains_list[i][j]
                f_tx = np.array([1, 0]).T
                f_rx = np.array([1, 0])

                assert tx_gains >= 0 and rx_gains >= 0
                test1 = np.dot(np.dot(f_rx, pol_matrix), f_tx)
                result = (test1 * tx_gains * rx_gains) ** 0.5
                ray_values.append(result)
            los_mult = 1
            if PL_type == 'LOS':
                los_mult = sqrt(1 / (1 + los_ray_pow))
            imp_r_val = sum(np.abs(
                    los_mult * sqrt(cluster_powers[i] / len(cluster_tx_gains_list[i])) * np.array(ray_values)) ** 2)
            impulse_responce.append(imp_r_val)
        impulse_responce = np.array(impulse_responce)
        IR_power_correction = RATIO2DB((np.sum(impulse_responce)))

        delays_for_plotting = []
        for delay in cluster_delays_list:
            delays_for_plotting.append(np.mean(np.array(delay)))

        res.PL = PL - IR_power_correction
        res.Power_correction = IR_power_correction
        if PL_type == 'LOS':
            res.PL_type = 1
        else:
            res.PL_type = 0
        return res


def UMa_LoS(dist, f_carrier, h_bs=25.0, h_ms=1.5):
    """PL formula for LoS ray"""

    d_bp = 4.0 * (h_bs - 1.0) * (h_ms - 1.0) * f_carrier / 3.0e8
    PL_MAX = 200
    if dist > 5000.0:
        pl = PL_MAX
        pl_average = pl
    else:
        if dist < 10.0:
            dist = 10.0 + (10.0 - dist)

        if dist < d_bp:
            pl = 22 * log10(dist) + 28 + 20 * log10(f_carrier / 1e9)
        else:
            pl = 40.0 * log10(dist) + 7.8 - 18.0 * log10(h_bs - 1) - 18.0 * log10(h_ms - 1) + 2.0 * log10(f_carrier / 1e9)

        pl_average = np.array(pl)
        if gl.shadow_fading is True:
            sigma = 4
            mu = 0
            pl = pl + np.random.normal(mu, sigma)

    return pl, pl_average


def UMi_street_canyon_los(dist2d, dist3d, f_carrier, h_bs, h_ut):
    d_bp = 4.0 * (h_bs - 1.0) * (h_ut - 1.0) * f_carrier / speed_of_light
    PL_MAX = 200
    if dist2d > 5000.0:
        return PL_MAX
    if dist2d < 10.0:
        dist2d = 10.0 + (10.0 - dist2d)

    if dist2d < d_bp:
        pl = 32.4 + 21 * log10(dist3d) + 20 * log10(f_carrier / 1e9)
    else:
        pl = 32.4 + 40.0 * log10(dist3d) + 20.0 * log10(f_carrier / 1e9) - 9.5 * log10(d_bp ** 2 + (h_bs - h_ut) ** 2)
    return pl


def UMi_LOS_probability(distance):

    if distance <= 18:
        probability = 1
    else:
        probability = 18/distance + (1 - 18/distance)*exp(-distance/36)
    return probability


def human_body_blockage_probability(bl_density, distance, tx_height, rx_height=1.5,
                                    bl_height=1.7, radius=0.15):
    """
    Probability of link blockage
    :param bl_density: blockers density
    :param distance: distance between the Tx and Rx
    :param tx_height: Tx height in meters
    :param rx_height: Rx height in meters
    :param bl_height: blockers' height in meters
    :param radius: blockers' radius in meters
    """

    x = distance
    probability_non_blocked = exp(-2*bl_density*radius*(x*((bl_height - rx_height)/(tx_height-rx_height)) + radius))

    return probability_non_blocked


def time_blocked(is_blocked, bl_density, distance, average_velocity,
                 bl_height=1.7, radius=0.1, tx_height=1.5, rx_height=10):
    """
    Compute the time for a link to be blocked
    :param bl_density: blockers density
    :param distance: distance between the Tx and Rx
    :param average_velocity: blockers' speed
    :param tx_height: Tx height in meters
    :param rx_height: Rx height in meters
    :param bl_height: blockers' height in meters
    :param radius: blockers' radius in meters
    """

    # first, we need to translate spatial density into time intensity of blockers per second
    blockage_zone_perimeter = ((distance * (bl_height - rx_height)) / (tx_height - rx_height) + radius) * 2 * radius
    lambda_in_time = (2 / 5) * blockage_zone_perimeter * bl_density * average_velocity
    k = 1.3
    time_interval_blocked = np.random.exponential(1/lambda_in_time)
    E_T = k * (2 * radius / average_velocity)                           # mean time to cross the LoS region
    E_n = (1 / lambda_in_time) * (np.exp(lambda_in_time * E_T) - 1)     # mean time in non-block condition
    lambda_non_blocked = 1/E_n
    time_interval_non_blocked = np.random.exponential(1/lambda_non_blocked)

    if is_blocked == 1:
        return time_interval_non_blocked
    else:
        return time_interval_blocked


def spectral_efficiency_UMi(dist2d, dist3d, f_carrier, h_bs, h_ut, P_t, B):
    """
    Spectral efficiency in case of LoS simple scenario
    (used for testing optimal solutions)
    """
    pathloss = UMi_street_canyon_los(dist2d, dist3d, f_carrier, h_bs, h_ut)
    snr = P_t + 15 - pathloss + 10 + 174 - 10 * log10(B) - 13
    i = 3
    sinr_lin = DB2RATIO(snr - i)
    s = log2(1 + sinr_lin)
    return s


def spectral_efficiency_UMa(dist3d, f_carrier, h_bs, h_ut, P_t, B):
    pathloss = UMa_LoS(dist3d, f_carrier, h_bs, h_ut)[1]
    snr = P_t + 15 - pathloss + 10 + 174 - 10 * log10(B) - 13
    i = 3
    sinr_lin = DB2RATIO(snr - i)
    s = log2(1 + sinr_lin)
    return s
