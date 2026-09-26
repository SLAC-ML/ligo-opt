"""Mockup objective calculation for locking-parameter optimization.

Computes objectives from design parameters D and locking parameters L.
In reality, this would evaluate physics performance metrics based on
the locking configuration achieved for a given design.
"""

# BEGIN TEMPORARY COST RECORDING IMPORTS
import csv
import fcntl
import os
from datetime import datetime, timezone
from pathlib import Path
# END TEMPORARY COST RECORDING IMPORTS

import numpy as np
from finesse.cymath.homs import HGModes
from scipy.integrate import dblquad


# Keep PSOoptimizer's nominal and four perturbed evaluations unchanged, and select
# here which objective consumes them. This makes the current step-back test and the
# eventual full run differ by one explicit setting rather than commented code.
# Valid values: "design_power", "perturbation", "full".
COST_MODE = "full"

# Used only when COST_MODE == "full". Keeping these explicit makes it clear that the
# two terms have different units and will need physically meaningful tuning before a
# production full-cost run.
FULL_POWER_WEIGHT = 0.5
FULL_PERTURBATION_WEIGHT = 0.5


# BEGIN TEMPORARY COST RECORDING — delete this block and the call below later.
RECORD_FULL_COST = True
FULL_COST_CSV = (
    Path(__file__).resolve().parents[1] / "analysis" / "results" / "full_cost_records.csv"
)


def _record_full_cost(C_power, C_perturb, C_total, nominal_power, Delta, pd_name):
    """Append one full-cost evaluation; preserve invalid costs as 'inf'."""
    if not RECORD_FULL_COST:
        return
    row = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "pid": os.getpid(),
        "pd_name": pd_name,
        "nominal_power": nominal_power,
        "delta_itm": Delta[0],
        "delta_etm": Delta[1],
        "power_cost": C_power,
        "perturbation_cost": C_perturb,
        "power_weight": FULL_POWER_WEIGHT,
        "perturbation_weight": FULL_PERTURBATION_WEIGHT,
        "weighted_power_cost": FULL_POWER_WEIGHT * C_power,
        "weighted_perturbation_cost": FULL_PERTURBATION_WEIGHT * C_perturb,
        "total_cost": C_total,
    }
    FULL_COST_CSV.parent.mkdir(parents=True, exist_ok=True)
    with FULL_COST_CSV.open("a", newline="") as stream:
        # Prevent concurrent workers from duplicating the header/interleaving rows.
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        stream.seek(0, os.SEEK_END)
        writer = csv.DictWriter(stream, fieldnames=list(row))
        if stream.tell() == 0:
            writer.writeheader()
        writer.writerow(row)
        stream.flush()
# END TEMPORARY COST RECORDING


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

# def calc_stability_cost(kat):
#     g_value = kat.cavArm.g[0]

#     if g_value < 0.15 or g_value > 0.85:
#         C_stable = 1
#     else:
#         C_stable = 0

#     return C_stable

def _get_power_for_pd_name(pd_name, out_names, powers):
    out_names = np.asarray(out_names)
    powers = np.asarray(powers)

    idx = np.where(out_names == pd_name)[0]
    if idx.size == 0:
        raise ValueError(f"{pd_name} not found in out_names")

    return powers[idx[0]]
def calc_gain_cost(out_powers_ITM, out_powers_ETM, out_names_ITM, out_names_ETM, Delta_ITM, Delta_ETM, pd_name):
    # separate out the powers from design_parameter-delta, design_parameter, and design_parameter+delta
    out_powers_pos_ITM, out_powers_D_ITM, out_powers_neg_ITM = out_powers_ITM
    out_powers_pos_ETM, out_powers_D_ETM, out_powers_neg_ETM = out_powers_ETM
    out_names_pos_ITM, out_names_D_ITM, out_names_neg_ITM = out_names_ITM
    out_names_pos_ETM, out_names_D_ETM, out_names_neg_ETM = out_names_ETM

    # get the power for the pd_name
    out_power_pos_ITM = _get_power_for_pd_name(pd_name, out_names_pos_ITM, out_powers_pos_ITM)
    out_power_D_ITM = _get_power_for_pd_name(pd_name, out_names_D_ITM, out_powers_D_ITM)
    out_power_neg_ITM = _get_power_for_pd_name(pd_name, out_names_neg_ITM, out_powers_neg_ITM)

    out_power_pos_ETM = _get_power_for_pd_name(pd_name, out_names_pos_ETM, out_powers_pos_ETM)
    out_power_D_ETM = _get_power_for_pd_name(pd_name, out_names_D_ETM, out_powers_D_ETM)
    out_power_neg_ETM = _get_power_for_pd_name(pd_name, out_names_neg_ETM, out_powers_neg_ETM)

    # A non-positive nominal (design-point) power means the cavity is unstable /
    # unsolvable (finesse_sim zeros everything, out == -1). With all four
    # perturbation runs also zero, every |dP/dROC| term below is 0 and the raw
    # sum would score this geometry as PERFECT -- the exact opposite of reality.
    # Penalize it maximally instead, so the optimizer is pushed away from
    # unstable / lossy points. (ITM and ETM share the same D run, so a single
    # check mathematically suffices, but both are tested for clarity/safety.)
    selected_powers = np.asarray([
        out_power_pos_ITM, out_power_D_ITM, out_power_neg_ITM,
        out_power_pos_ETM, out_power_D_ETM, out_power_neg_ETM,
    ])
    if (
        not np.all(np.isfinite(selected_powers))
        or np.any(selected_powers < 0)
        or out_power_D_ITM <= 0
        or out_power_D_ETM <= 0
    ):
        return np.inf

    if not np.isfinite(Delta_ITM) or not np.isfinite(Delta_ETM) or Delta_ITM == 0 or Delta_ETM == 0:
        return np.inf

    # separate out the delta values (positive delta value is the same as negative delta value)
    delta_pos_ITM, delta_neg_ITM = Delta_ITM, Delta_ITM
    delta_pos_ETM, delta_neg_ETM = Delta_ETM, Delta_ETM

    # calculate the gain difference between design_parameter-delta and design_parameter+delta
    delta_power_pos_ITM = out_power_pos_ITM - out_power_D_ITM
    delta_power_pos_ETM = out_power_pos_ETM - out_power_D_ETM
    
    delta_power_neg_ITM = out_power_neg_ITM - out_power_D_ITM
    delta_power_neg_ETM = out_power_neg_ETM - out_power_D_ETM

    ratio_power_pos_ITM = np.abs(delta_power_pos_ITM / delta_pos_ITM)
    ratio_power_pos_ETM = np.abs(delta_power_pos_ETM / delta_pos_ETM)

    ratio_power_neg_ITM = np.abs(delta_power_neg_ITM / delta_neg_ITM)
    ratio_power_neg_ETM = np.abs(delta_power_neg_ETM / delta_neg_ETM)

    cost_gain = ratio_power_pos_ITM + ratio_power_pos_ETM + ratio_power_neg_ITM + ratio_power_neg_ETM

    return float(cost_gain) if np.isfinite(cost_gain) else np.inf

def cal_D_power_cost(ITM_powers, ITM_names,pd_name):
    # cost that only looks at the non-perturbed (nominal design-point)
    # power — the middle ("D") element, ignoring the +/- Delta perturbed runs.
    # Note: the caller (PSOoptimizer) reuses the SAME nominal D run as the
    # middle entry of both the ITM and the ETM lists, so the ITM and ETM D
    # powers are identical; we only need to read it once.
    
    _, out_powers_D_ITM, _ = ITM_powers
    _, out_names_D_ITM, _ = ITM_names

    out_power_D = _get_power_for_pd_name(pd_name, out_names_D_ITM, out_powers_D_ITM)

    if not np.isfinite(out_power_D) or out_power_D <= 0:
        # No (or zero) circulating power for the pd_name -> worst possible cost.
        return np.inf

    # Higher power is better, so the cost is simply the inverse of the power.
    return float(1.0 / out_power_D)

def calc_cost(kat, ITM_powers, ETM_powers, ITM_names, ETM_names, Delta, pd_name):

    #TODO: find appropriate weighting for each cost term
    # weight_gain = 0.3
    # weight_stable = 1 - weight_gain

    # Unpack the ITM and ETM powers and names, as well as the Delta values
    ITM_ROC_delta, ETM_ROC_delta = Delta


    
    C_power = cal_D_power_cost(ITM_powers, ITM_names, pd_name)

    if COST_MODE == "design_power":
        return C_power

    C_perturb = calc_gain_cost(
        ITM_powers, ETM_powers,
        ITM_names, ETM_names,
        ITM_ROC_delta, ETM_ROC_delta,
        pd_name,
    )

    if COST_MODE == "perturbation":
        return C_perturb

    if COST_MODE == "full":
        C_total = FULL_POWER_WEIGHT * C_power + FULL_PERTURBATION_WEIGHT * C_perturb
        C_total = float(C_total) if np.isfinite(C_total) else np.inf
        # BEGIN TEMPORARY COST RECORDING CALL
        if RECORD_FULL_COST:
            _record_full_cost(
                C_power, C_perturb, C_total,
                _get_power_for_pd_name(pd_name, ITM_names[1], ITM_powers[1]),
                Delta, pd_name,
            )
        # END TEMPORARY COST RECORDING CALL
        return C_total

    raise ValueError(
        f"Unknown COST_MODE {COST_MODE!r}; expected 'design_power', 'perturbation', or 'full'."
    )

    # C_stable
    # C_stable = calc_stability_cost(kat)

    # C_HOM
    # C_HOM = calc_HOM_cost(out)

    # C_loss
    # C_loss = calc_round_trip_clipping_loss(kat)
