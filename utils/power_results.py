"""Capture original objective powers and reconstruct costs without model calls."""

import numpy as np

from utils import calc


class PowerEvaluationRecorder:
    """Keep one strictly best evaluation; equal costs preserve the first winner.

    This matches unconstrained PSO visiting particles in index order. It does not
    track particle histories or perform any inference or simulation.
    """

    def __init__(self):
        self.best = None

    def capture(self, params, cost, itm_powers, etm_powers, itm_names, etm_names,
                delta, pd_name, model_path, evaluation_mode):
        cost = float(cost)
        if self.best is not None and not cost < self.best['cost']:
            return
        powers = {
            'nominal': float(calc._get_power_for_pd_name(pd_name, itm_names[1], itm_powers[1])),
            'ITM_plus': None,
            'ITM_minus': None,
            'ETM_plus': None,
            'ETM_minus': None,
        }
        if evaluation_mode == 'perturbed':
            for key, names, values in (
                ('ITM_plus', itm_names[0], itm_powers[0]),
                ('ITM_minus', itm_names[2], itm_powers[2]),
                ('ETM_plus', etm_names[0], etm_powers[0]),
                ('ETM_minus', etm_names[2], etm_powers[2]),
            ):
                powers[key] = float(calc._get_power_for_pd_name(pd_name, names, values))
        self.best = {
            'params': np.asarray(params, dtype=float).copy(),
            'cost': cost,
            'powers': powers,
            'delta': tuple(float(value) for value in delta),
            'pd_name': pd_name,
            'model_path': str(model_path) if model_path is not None else None,
            'evaluation_mode': evaluation_mode,
            'cost_mode': calc.COST_MODE,
            'power_weight': calc.FULL_POWER_WEIGHT,
            'perturbation_weight': calc.FULL_PERTURBATION_WEIGHT,
        }

    def verified_record(self, params, cost):
        """Check the winning association and cost using only saved numbers."""
        record = self.best
        if record is None:
            raise RuntimeError('No power evaluation was captured.')
        if not np.array_equal(record['params'], params) or record['cost'] != float(np.asarray(cost).item()):
            raise RuntimeError('Captured power evaluation does not match the selected best.')
        if reconstruct_cost(record) != record['cost']:
            raise RuntimeError('Captured powers do not reproduce the recorded cost.')
        return record


def reconstruct_cost(record):
    """Rebuild a cost from a saved record, using its mode and weights.

    Calls only the arithmetic cost components; no CSV logging, GNN inference,
    Finesse simulation, or changes to global cost settings are involved.
    """
    powers = record['powers']
    nominal = powers['nominal']
    if record['evaluation_mode'] == 'nominal':
        itm = etm = [[nominal], [nominal], [nominal]]
    else:
        itm = [[powers['ITM_plus']], [nominal], [powers['ITM_minus']]]
        etm = [[powers['ETM_plus']], [nominal], [powers['ETM_minus']]]
    names = [[record['pd_name']]] * 3
    power_cost = calc.cal_D_power_cost(itm, names, record['pd_name'])
    mode = record['cost_mode']
    if mode == 'design_power':
        return power_cost
    perturbation_cost = calc.calc_gain_cost(
        itm, etm, names, names, *record['delta'], record['pd_name'],
    )
    if mode == 'perturbation':
        return perturbation_cost
    if mode == 'full':
        cost = (record['power_weight'] * power_cost
                + record['perturbation_weight'] * perturbation_cost)
        return float(cost) if np.isfinite(cost) else np.inf
    raise ValueError(f'Unknown saved cost mode: {mode!r}')
