from gl_vars import gl
from stat_container import st
from mobility.configure_mobility import set_mobility_model
from lib.random_drop import drop_DgNB, drop_IAB
from iab_optimization.optimization import Optimization, OptimizationParams
from topology_formation import TopologyCreator
from phy.interpolate_ber_curves import load_BER
from phy.abstractions import set_params_OFDM
from phy.bercurve import set_params_PHY

import numpy as np
import matplotlib.pyplot as plt
from math import log10

SIM_SEED = 0
subframe_duration_s = 1e-3

BER_CURVES = load_BER()
OFDM_params = set_params_OFDM(gl.numerology_num)

st.__init__()
UE_mobility_model = set_mobility_model(SIM_SEED, subframe_duration_s)
UE_positions = next(UE_mobility_model)
# Transform according to the cell size
x = UE_positions[:, 0] - gl.cell_radius_m
y = UE_positions[:, 1] - gl.cell_radius_m
UE_positions_tr = [x, y, UE_positions[:, 2]]
UE_positions_tr = np.transpose(UE_positions_tr)
# Drop the DgNB on the edge
drop_DgNB(SIM_SEED)
drop_IAB(SIM_SEED)

topology = TopologyCreator(BER_CURVES)
topology.determine_initial_associations(UE_positions_tr)

bs_pos = gl.DgNB_pos
iab_pos = gl.IAB_pos
ue_pos = UE_positions_tr

# assume some ues active, some are not
active_ues = [0, 5, 7, 10, 8, 25, 28]
ues_belong_to_iab_nodes, ues_belong_to_DgNB = [], []

spect_eff_ue_iab, spect_eff_ue_bs = np.array([]), np.array([])
for ue_i in range(0, gl.n_UEs):
    if ue_i in active_ues:
        se = st.PHY_params['DL'][ue_i].mod_order * st.PHY_params['DL'][ue_i].code_rate
        if st.closest_bs_indices[ue_i] != 0:
            ues_belong_to_iab_nodes.append(ue_i)
            spect_eff_ue_iab = np.append(spect_eff_ue_iab, [se])
        else:
            ues_belong_to_DgNB.append(ue_i)
            spect_eff_ue_bs = np.append(spect_eff_ue_bs, se)

PL_DgNB_IAB = topology.PL_bw_DgNB_IAB
spect_eff_DgNB_IAB = np.array([])
for iab_i in range(0, gl.n_IAB):
    RSRP = gl.DgNB_tx_power_dBm - PL_DgNB_IAB[iab_i] - st.noise_power_DL - gl.interference_margin_dB
    params_DgNB_IAB = set_params_PHY(RSRP, BER_CURVES)
    spect_eff_DgNB_IAB = np.append(spect_eff_DgNB_IAB, params_DgNB_IAB.mod_order*params_DgNB_IAB.code_rate)

ue_iab = ue_pos[ues_belong_to_iab_nodes]
ue_bs = ue_pos[ues_belong_to_DgNB]

params = OptimizationParams(gl.blockers_density, bs_pos, iab_pos, ue_pos, ue_bs, ue_iab, gl.Bandwidth, gl.Bandwidth,
                            len(active_ues))
optimization = Optimization(params)
h, eps, y, x, backhaul = optimization.optimize_single_link(False, spect_eff_ue_iab, spect_eff_ue_bs, spect_eff_DgNB_IAB)

fg = plt.figure()
eps = np.around(eps, 2)
barWidth = 0.4
r1 = np.arange(1, eps.shape[0] + 1)
plt.bar(r1, eps, width=barWidth, edgecolor='black', align='center', alpha=0.5, color='b')
for i in range(eps.shape[0]):
    plt.text(i+1, eps[i], str(eps[i]))
plt.xlabel('Timeslot')
plt.ylabel('Timeslot duration')
plt.ylim([0, 0.8])
plt.show()
