"""Mockup objective calculation for locking-parameter optimization.

Computes objectives from design parameters D and locking parameters L.
In reality, this would evaluate physics performance metrics based on
the locking configuration achieved for a given design.
"""

import numpy as np
from finesse.cymath.homs import HGModes
from scipy.integrate import dblquad
from scipy.signal import find_peaks

def calc_objective(D, L):
    """Compute scalar objective from design params and locking params.

    Mockup: minimize a combined metric that depends on both D and L.
    Lower is better.

    Args:
        D: Design parameters, shape (n, d)
        L: Locking parameters, shape (n, 2)

    Returns:
        obj: Objective values, shape (n,)
    """
    # Distance from target in design space
    d_cost = np.sum((D - 0.3) ** 2, axis=1) 

    # Distance from target in locking space
    l_cost = np.sum((L - 0.5) ** 2, axis=1)

    return d_cost + 2.0 * l_cost


def calc_HOM_cost(out):
    amp_itm = out["E_itm"].flatten()
    p_itm = np.abs(amp_itm)**2
    cost = np.sum(p_itm[1:])
    return cost

def calc_round_trip_clipping_loss(kat,out):
    r_apeture = 0.17

    HGs_ITM = HGModes(kat.ITM.p2.i.q, [00])
    HGs_ETM = HGModes(kat.ETM.p1.i.q, [00])

    amp_itm = out["E_itm"].flatten()[:, None]
    amp_etm = out["E_etm"].flatten()[:, None]
    def intensity_polar_ITM(r, theta):
        x = np.array([r * np.cos(theta)], dtype=np.float64)
        y = np.array([r * np.sin(theta)], dtype=np.float64)
        a = HGs_ITM.compute_points(x,y) * amp_itm
        E = np.sum(a, axis=0)[0]
        I = np.abs(E)**2 * r #jacobian
        return I
    def intensity_polar_ETM(r, theta):
        x = np.array([r * np.cos(theta)], dtype=np.float64)
        y = np.array([r * np.sin(theta)], dtype=np.float64)
        a = HGs_ETM.compute_points(x,y) * amp_etm
        E = np.sum(a, axis=0)[0]
        I = np.abs(E)**2 * r #jacobian
        return I
    loss_ITM, error_ITM = dblquad(intensity_polar_ITM, 0, 2*np.pi, lambda x: r_apeture, lambda x: 10)
    loss_ETM, error_ETM = dblquad(intensity_polar_ETM, 0, 2*np.pi, lambda x: r_apeture, lambda x: 10)
    loss = loss_ETM + loss_ITM
    power_circ = out['circ']
    loss_ppm = (loss / power_circ) * 1e6
    return loss_ppm 

def calc_stability_cost(kat):
    g_value = kat.cavArm.g[0]
    C_stable = 10 * (g_value - 0.5)**2
    return C_stable

def calc_gain_cost(out):
    gain = out['circ']
    cost_gain = 1/gain
    return cost_gain

def calc_cost(kat, out): #TODO: find appropriate weighting for each cost term
    # C_gain
    C_gain = calc_gain_cost(out)

    # C_stable
    C_stable = calc_stability_cost(kat)

    # C_HOM
    C_HOM = calc_HOM_cost(out)

    # C_loss
    C_loss = calc_round_trip_clipping_loss(kat)

    C_total = C_gain + C_stable + C_loss + C_HOM
    return C_total