"""Calculate cavity costs from nominal and perturbed detector powers.

COST_MODE selects inverse nominal power, perturbation sensitivity, or their
weighted sum. Optional CSV recording retains the full-cost components.
"""

import csv
import fcntl
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


# Select which objective consumes the nominal and perturbed evaluations.
# Valid values: "design_power", "perturbation", "full".
COST_MODE = "full"

# Used only when COST_MODE == "full". Keeping these explicit makes it clear that the
# two terms have different units and will need physically meaningful tuning before a
# production full-cost run.
FULL_POWER_WEIGHT = 0.5
FULL_PERTURBATION_WEIGHT = 0.5


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
        if RECORD_FULL_COST:
            _record_full_cost(
                C_power, C_perturb, C_total,
                _get_power_for_pd_name(pd_name, ITM_names[1], ITM_powers[1]),
                Delta, pd_name,
            )
        return C_total

    raise ValueError(
        f"Unknown COST_MODE {COST_MODE!r}; expected 'design_power', 'perturbation', or 'full'."
    )
