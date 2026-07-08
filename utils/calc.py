"""Mockup objective calculation for locking-parameter optimization.

Computes objectives from design parameters D and locking parameters L.
In reality, this would evaluate physics performance metrics based on
the locking configuration achieved for a given design.
"""

import numpy as np
from finesse.cymath.homs import HGModes
from scipy.integrate import dblquad


# def calc_HOM_cost(out):
#     amp_itm = out["E_itm"].flatten()
#     p_itm = np.abs(amp_itm)**2
#     cost = np.sum(p_itm[1:])
#     return cost

# def calc_round_trip_clipping_loss(kat,out):
#     r_apeture = 0.17

#     HGs_ITM = HGModes(kat.ITM.p2.i.q, [00])
#     HGs_ETM = HGModes(kat.ETM.p1.i.q, [00])

#     amp_itm = out["E_itm"].flatten()[:, None]
#     amp_etm = out["E_etm"].flatten()[:, None]
#     def intensity_polar_ITM(r, theta):
#         x = np.array([r * np.cos(theta)], dtype=np.float64)
#         y = np.array([r * np.sin(theta)], dtype=np.float64)
#         a = HGs_ITM.compute_points(x,y) * amp_itm
#         E = np.sum(a, axis=0)[0]
#         I = np.abs(E)**2 * r #jacobian
#         return I
#     def intensity_polar_ETM(r, theta):
#         x = np.array([r * np.cos(theta)], dtype=np.float64)
#         y = np.array([r * np.sin(theta)], dtype=np.float64)
#         a = HGs_ETM.compute_points(x,y) * amp_etm
#         E = np.sum(a, axis=0)[0]
#         I = np.abs(E)**2 * r #jacobian
#         return I
#     loss_ITM, error_ITM = dblquad(intensity_polar_ITM, 0, 2*np.pi, lambda x: r_apeture, lambda x: 10)
#     loss_ETM, error_ETM = dblquad(intensity_polar_ETM, 0, 2*np.pi, lambda x: r_apeture, lambda x: 10)
#     loss = loss_ETM + loss_ITM
#     power_circ = out['circ']
#     loss_ppm = (loss / power_circ) * 1e6
#     return loss_ppm 

def calc_stability_cost(kat):
    g_value = kat.cavArm.g[0]
    C_stable = 10 * (g_value - 0.5)**2
    return C_stable

def calc_gain_cost(out_power_ITM, out_power_ETM, Delta_ITM, Delta_ETM):
    # separate out the powers from design_parameter-delta, design_parameter, and design_parameter+delta
    out_power_pos_ITM, out_power_D_ITM, out_power_neg_ITM = out_power_ITM
    out_power_pos_ETM, out_power_D_ETM, out_power_neg_ETM = out_power_ETM
    delta_pos_ITM, delta_neg_ITM = Delta_ITM
    delta_pos_ETM, delta_neg_ETM = Delta_ETM

    # calculate the gain difference between design_parameter-delta and design_parameter+delta
    delta_power_pos_ITM = out_power_pos_ITM - out_power_D_ITM
    delta_power_pos_ETM = out_power_pos_ETM - out_power_D_ETM
    
    delta_power_neg_ITM = out_power_neg_ITM - out_power_D_ITM
    delta_power_neg_ETM = out_power_neg_ETM - out_power_D_ETM

    ratio_power_pos_ITM = delta_power_pos_ITM / delta_pos_ITM
    ratio_power_pos_ETM = delta_power_pos_ETM / delta_pos_ETM

    ratio_power_neg_ITM = delta_power_neg_ITM / delta_neg_ITM
    ratio_power_neg_ETM = delta_power_neg_ETM / delta_neg_ETM

    cost_gain = ratio_power_pos_ITM + ratio_power_pos_ETM + ratio_power_neg_ITM + ratio_power_neg_ETM

    return cost_gain

def calc_cost(kat, out_powers_ITM, out_powers_ETM, Delta_ITM, Delta_ETM): #TODO: find appropriate weighting for each cost term
    names, powers = out_powers_ITM, out_powers_ETM
    delta_pos_ITM, delta_neg_ITM = Delta_ITM
    delta_pos_ETM, delta_neg_ETM = Delta_ETM

    port = 'p_ITM_p2_i'
    index_pick = names[0].index(port)
    print("Calculating cost with port", port, "at index", index_pick)
    out_power_pos = powers[0][index_pick]
    out_power_D = powers[1][index_pick]
    out_power_neg = powers[2][index_pick]
    
    

    # C_gain
    C_gain = calc_gain_cost((out_power_pos, out_power_D, out_power_neg), Delta_ITM, Delta_ETM)

    # C_stable
    C_stable = calc_stability_cost(kat)

    # C_HOM
    # C_HOM = calc_HOM_cost(out)

    # C_loss
    # C_loss = calc_round_trip_clipping_loss(kat)

    C_total = C_gain + C_stable
    return C_total